# Copyright (c) 2022 NVIDIA CORPORATION. 
#   Licensed under the MIT license.

# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import time
import argparse
import json
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader

from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

from bigvgan.env import AttrDict, build_env
from bigvgan.models import BigVGAN, MultiPeriodDiscriminator, MultiResolutionDiscriminator
from bigvgan.models import feature_loss, generator_loss, discriminator_loss
from bigvgan.utils import plot_spectrogram, plot_spectrogram_clipped, scan_checkpoint, load_checkpoint, save_checkpoint, save_audio
from bigvgan.meldataset import MelDataset, mel_spectrogram, get_dataset_filelist, MAX_WAV_VALUE

from tqdm import tqdm


torch.backends.cudnn.benchmark = False

def train(rank, a, h):
    if h.num_gpus > 1:
        # initialize distributed
        init_process_group(backend=h.dist_config['dist_backend'],
                           init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus,
                           rank=rank)

    # set seed and device
    torch.cuda.manual_seed(h.seed)
    torch.cuda.set_device(rank)
    device = torch.device('cuda:{:d}'.format(rank))

    # define BigVGAN generator
    generator = BigVGAN(h).to(device)
    print("Generator params: {}".format(sum(p.numel() for p in generator.parameters())))

    # define discriminators. MPD is used by default
    mpd = MultiPeriodDiscriminator(h).to(device)
    print("Discriminator mpd params: {}".format(sum(p.numel() for p in mpd.parameters())))

    # define additional discriminators. BigVGAN uses MRD as default
    mrd = MultiResolutionDiscriminator(h).to(device)
    print("Discriminator mrd params: {}".format(sum(p.numel() for p in mrd.parameters())))

    # Load pretrain
    state_dict_g = load_checkpoint('./bigvgan_pretrain/g_05000000', device)
    generator.load_state_dict(state_dict_g['generator'])

    # create or scan the latest checkpoint from checkpoints directory
    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_d = scan_checkpoint(a.checkpoint_path, 'd_')

    # load the latest checkpoint if exists
    steps = 0
    if cp_g is None or cp_d is None:
        state_dict_d = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_d = load_checkpoint(cp_d, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_d['mpd'])
        mrd.load_state_dict(state_dict_d['mrd'])
        steps = state_dict_d['steps'] + 1
        last_epoch = state_dict_d['epoch']

    # initialize DDP, optimizers, and schedulers
    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        mrd = DistributedDataParallel(mrd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(mrd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_d is not None:
        optim_g.load_state_dict(state_dict_d['optim_g'])
        optim_d.load_state_dict(state_dict_d['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    # define training and validation datasets
    training_filelist, validation_filelist = get_dataset_filelist(a)

    trainset = MelDataset(training_filelist, h, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler, batch_size=h.batch_size,
                              pin_memory=True, drop_last=True)

    if rank == 0:
        validset = MelDataset(validation_filelist, h, h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device)

        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None, batch_size=1,
                                       pin_memory=True, drop_last=True)

        # Tensorboard logger
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))
        if a.save_audio: # also save audio to disk if --save_audio is set to True
            os.makedirs(os.path.join(a.checkpoint_path, 'samples'), exist_ok=True)

    # validation loop
    def validate(rank, a, h, loader, mode="seen"):
        assert rank == 0, "validate should only run on rank=0"
        generator.eval()
        torch.cuda.empty_cache()

        val_err_tot = 0

        if a.save_audio: # also save audio to disk if --save_audio is set to True
            os.makedirs(os.path.join(a.checkpoint_path, 'samples', 'gt_{}'.format(mode)), exist_ok=True)
            os.makedirs(os.path.join(a.checkpoint_path, 'samples', '{}_{:08d}'.format(mode, steps)), exist_ok=True)

        with torch.no_grad():
            print("step {} {} speaker validation...".format(steps, mode))

            # loop over validation set and compute metrics
            for j, batch in tqdm(enumerate(loader)):
                x, y, _, y_mel = batch
                y = y.to(device)
                if hasattr(generator, 'module'):
                    y_g_hat = generator.module(x.to(device))
                else:
                    y_g_hat = generator(x.to(device))
                y_mel = y_mel.to(device, non_blocking=True)
                y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                              h.sampling_rate, h.hop_size, h.win_size,
                                              h.fmin, h.fmax_for_loss)
                val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                # log audio and figures to Tensorboard
                if j % a.eval_subsample == 0:  # subsample every nth from validation set
                    if steps >= 0:
                        sw.add_audio('gt_{}/y_{}'.format(mode, j), y[0], steps, h.sampling_rate)
                        if a.save_audio: # also save audio to disk if --save_audio is set to True
                            save_audio(y[0], os.path.join(a.checkpoint_path, 'samples', 'gt_{}'.format(mode), '{:04d}.wav'.format(j)), h.sampling_rate)
                        sw.add_figure('gt_{}/y_spec_{}'.format(mode, j), plot_spectrogram(x[0]), steps)

                    sw.add_audio('generated_{}/y_hat_{}'.format(mode, j), y_g_hat[0], steps, h.sampling_rate)
                    if a.save_audio: # also save audio to disk if --save_audio is set to True
                        save_audio(y_g_hat[0, 0], os.path.join(a.checkpoint_path, 'samples', '{}_{:08d}'.format(mode, steps), '{:04d}.wav'.format(j)), h.sampling_rate)
                    # spectrogram of synthesized audio
                    y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                                 h.sampling_rate, h.hop_size, h.win_size,
                                                 h.fmin, h.fmax)
                    sw.add_figure('generated_{}/y_hat_spec_{}'.format(mode, j),
                                  plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)
                    # visualization of spectrogram difference between GT and synthesized audio
                    # difference higher than 1 is clipped for better visualization
                    spec_delta = torch.clamp(torch.abs(x[0] - y_hat_spec.squeeze(0).cpu()), min=1e-6, max=1.)
                    sw.add_figure('delta_dclip1_{}/spec_{}'.format(mode, j),
                                  plot_spectrogram_clipped(spec_delta.numpy(), clip_max=1.), steps)

            val_err = val_err_tot / (j + 1)

            # log evaluation metrics to Tensorboard
            sw.add_scalar("validation_{}/mel_spec_error".format(mode), val_err, steps)

        generator.train()

    # if the checkpoint is loaded, start with validation loop
    if steps != 0 and rank == 0:
        validate(rank, a, h, validation_loader,
                 mode="seen_{}".format(train_loader.dataset.name))

    # main training loop
    generator.train()
    mpd.train()
    mrd.train()
    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            x, y, _, y_mel = batch

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_mel = y_mel.to(device, non_blocking=True)
            y = y.unsqueeze(1)

            y_g_hat = generator(x)
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                          h.sampling_rate, h.hop_size, h.win_size,
                                          h.fmin, h.fmax_for_loss)

            optim_d.zero_grad()
            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
            # MRD
            y_ds_hat_r, y_ds_hat_g, _, _ = mrd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            # MPD + MRD
            loss_disc_all = loss_disc_s + loss_disc_f
            # whether to freeze D for initial training steps
            if steps >= a.freeze_step:
                loss_disc_all.backward()
                grad_norm_mpd = torch.nn.utils.clip_grad_norm_(mpd.parameters(), 1000.)
                grad_norm_mrd = torch.nn.utils.clip_grad_norm_(mrd.parameters(), 1000.)
                optim_d.step()
            else:
                print("WARNING: skipping D training for the first {} steps".format(a.freeze_step))
                grad_norm_mpd = 0.
                grad_norm_mrd = 0.

            # generator
            optim_g.zero_grad()
            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45
            # MPD loss
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            # MRD loss
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = mrd(y, y_g_hat)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            if steps >= a.freeze_step:
                loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
            else:
                print("WARNING: using regression loss only for G for the first {} steps".format(a.freeze_step))
                loss_gen_all = loss_mel
            loss_gen_all.backward()
            grad_norm_g = torch.nn.utils.clip_grad_norm_(generator.parameters(), 1000.)
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, mel_error, time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/d_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, 
                                    {'mpd': (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                                     'mrd': (mrd.module if h.num_gpus > 1 else mrd).state_dict(),
                                     'optim_g': optim_g.state_dict(),
                                     'optim_d': optim_d.state_dict(),
                                     'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)
                    sw.add_scalar("training/fm_loss_mpd", loss_fm_f.item(), steps)
                    sw.add_scalar("training/gen_loss_mpd", loss_gen_f.item(), steps)
                    sw.add_scalar("training/disc_loss_mpd", loss_disc_f.item(), steps)
                    sw.add_scalar("training/grad_norm_mpd", grad_norm_mpd, steps)
                    sw.add_scalar("training/fm_loss_mrd", loss_fm_s.item(), steps)
                    sw.add_scalar("training/gen_loss_mrd", loss_gen_s.item(), steps)
                    sw.add_scalar("training/disc_loss_mrd", loss_disc_s.item(), steps)
                    sw.add_scalar("training/grad_norm_mrd", grad_norm_mrd, steps)
                    sw.add_scalar("training/grad_norm_g", grad_norm_g, steps)
                    sw.add_scalar("training/learning_rate_d", scheduler_d.get_last_lr()[0], steps)
                    sw.add_scalar("training/learning_rate_g", scheduler_g.get_last_lr()[0], steps)
                    sw.add_scalar("training/epoch", epoch+1, steps)

                # validation
                if steps % a.validation_interval == 0:
                    # plot training input x so far used
                    for i_x in range(x.shape[0]):
                        sw.add_figure('training_input/x_{}'.format(i_x), plot_spectrogram(x[i_x].cpu()), steps)
                        sw.add_audio('training_input/y_{}'.format(i_x), y[i_x][0], steps, h.sampling_rate)
                    # seen and unseen speakers validation loops
                    if steps != 0:
                        validate(rank, a, h, validation_loader)

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_training_file', default='./data/files/train.txt')
    parser.add_argument('--input_validation_file', default='./data/files/valid.txt')
    parser.add_argument('--checkpoint_path', default='logs/bigvgan')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=100000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=10000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=10000, type=int)
    parser.add_argument('--freeze_step', default=0, type=int,
                        help='freeze D for the first specified steps.')
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--eval_subsample', default=5, type=int,
                        help="subsampling during evaluation loop")
    parser.add_argument('--save_audio', default=True, type=bool,
                        help="save audio of test set inference to disk")
    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
