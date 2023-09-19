# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import params
from grad import GradTTS
from grad_extend.data import TextMelDataset, TextMelBatchCollate
from grad_extend.utils import plot_tensor, save_plot
from text.symbols import symbols


def load_part(model, saved_state_dict):
    state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('framenc'):
            new_state_dict[k] = v
        else:
            new_state_dict[k] = saved_state_dict[k]
    model.load_state_dict(new_state_dict)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--checkpoint', type=str, default=None,
                        help="path of checkpoint pt file to resume training")
    args = parser.parse_args()
    resume_path = args.checkpoint

    assert torch.cuda.is_available()
    print('Numbers of GPU :', torch.cuda.device_count())

    torch.manual_seed(params.seed)
    np.random.seed(params.seed)

    print('Initializing logger...')
    logger = SummaryWriter(log_dir=params.log_dir)

    print('Initializing data loaders...')
    train_dataset = TextMelDataset(params.train_filelist_path)
    batch_collate = TextMelBatchCollate()
    loader = DataLoader(dataset=train_dataset, batch_size=params.batch_size,
                        collate_fn=batch_collate, drop_last=True,
                        num_workers=4, shuffle=False)
    test_dataset = TextMelDataset(params.valid_filelist_path)

    print('Initializing model...')
    model = GradTTS(len(symbols), 1, None, params.n_enc_channels, params.filter_channels, params.filter_channels_dp, 
                    params.n_heads, params.n_enc_layers, params.enc_kernel, params.enc_dropout, params.window_size, 
                    params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale).cuda()
    print('Number of encoder + duration predictor parameters: %.2fm' % (model.encoder.nparams/1e6))
    print('Number of decoder parameters: %.2fm' % (model.decoder.nparams/1e6))
    print('Total parameters: %.2fm' % (model.nparams/1e6))

    print('Initializing optimizer...')
    optim = torch.optim.Adam(params=model.parameters(), lr=params.learning_rate)
    # Transfer
    # checkpoint = torch.load("grad_tts_750.pt", map_location='cpu')
    # load_part(model, checkpoint['model'])

    initepoch = 1
    iteration = 0
    # Load Continue
    if resume_path is not None:
        print("Resuming from checkpoint: %s" % resume_path)
        checkpoint = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optim'])
        initepoch = checkpoint['epoch']
        iteration = checkpoint['steps']

    print('Logging test batch...')
    test_batch = test_dataset.sample_test_batch(size=params.test_size)
    for i, item in enumerate(test_batch):
        mel = item['y']
        logger.add_image(f'image_{i}/ground_truth', plot_tensor(mel.squeeze()),
                         global_step=0, dataformats='HWC')
        save_plot(mel.squeeze(), f'{params.log_dir}/original_{i}.png')

    print('Start training...')
    skip_diff_train = True
    if initepoch > params.fast_epochs:
        skip_diff_train = False
    for epoch in range(initepoch, params.full_epochs + 1):

        model.eval()
        print('Synthesis...')
        with torch.no_grad():
            for i, item in enumerate(test_batch):
                x = item['x'].to(torch.long).unsqueeze(0).cuda()
                x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
                bert = item['b'].to(torch.float32).unsqueeze(0).cuda()

                y_enc, y_dec, attn = model(x, x_lengths, bert, n_timesteps=20)
                logger.add_image(f'image_{i}/generated_enc',
                                 plot_tensor(y_enc.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/generated_dec',
                                 plot_tensor(y_dec.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                logger.add_image(f'image_{i}/alignment',
                                 plot_tensor(attn.squeeze().cpu()),
                                 global_step=iteration, dataformats='HWC')
                save_plot(y_enc.squeeze().cpu(), 
                          f'{params.log_dir}/generated_enc_{i}.png')
                save_plot(y_dec.squeeze().cpu(), 
                          f'{params.log_dir}/generated_dec_{i}.png')
                save_plot(attn.squeeze().cpu(), 
                          f'{params.log_dir}/alignment_{i}.png')


        model.train()
        dur_losses = []
        enc_losses = []
        mel_losses = []
        dec_losses = []
        with tqdm(loader, total=len(train_dataset)//params.batch_size) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                model.zero_grad()
                x, x_lengths = batch['x'].cuda(), batch['x_lengths'].cuda()
                y, y_lengths = batch['y'].cuda(), batch['y_lengths'].cuda()

                bert = batch['b'].cuda()

                dur_loss, enc_loss, mel_loss, \
                    dec_loss = model.compute_loss(x, x_lengths,
                                                  bert,
                                                  y, y_lengths,
                                                  out_size=params.out_size,
                                                  skip_diff=skip_diff_train)
                loss = sum([dur_loss, enc_loss, mel_loss, dec_loss])
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),
                                                               max_norm=1)
                mel_grad_norm = torch.nn.utils.clip_grad_norm_(model.framenc.parameters(),
                                                               max_norm=1)
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(),
                                                               max_norm=1)
                optim.step()

                logger.add_scalar('training/duration_loss', dur_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/enc_loss', enc_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/mel_loss', mel_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/dec_loss', dec_loss.item(),
                                  global_step=iteration)
                logger.add_scalar('training/enc_grad_norm', enc_grad_norm,
                                  global_step=iteration)
                logger.add_scalar('training/mel_grad_norm', mel_grad_norm,
                                  global_step=iteration)
                logger.add_scalar('training/dec_grad_norm', dec_grad_norm,
                                  global_step=iteration)
                
                dur_losses.append(dur_loss.item())
                enc_losses.append(enc_loss.item())
                mel_losses.append(mel_loss.item())
                dec_losses.append(dec_loss.item())
                
                if batch_idx % 5 == 0:
                    msg = f'Epoch: {epoch}, iteration: {iteration} | '
                    msg = msg + f'dur_loss: {dur_loss.item():.3f}, '
                    msg = msg + f'enc_loss: {enc_loss.item():.3f}, '
                    msg = msg + f'mel_loss: {mel_loss.item():.3f}, '
                    msg = msg + f'dec_loss: {dec_loss.item():.3f}'
                    progress_bar.set_description(msg)

                iteration += 1

        log_msg = 'Epoch %d: duration loss = %.3f ' % (epoch, np.mean(dur_losses))
        log_msg += '| enc loss = %.3f ' % np.mean(enc_losses)
        log_msg += '| mel loss = %.3f ' % np.mean(mel_losses)
        log_msg += '| dec loss = %.3f\n' % np.mean(dec_losses)
        with open(f'{params.log_dir}/train.log', 'a') as f:
            f.write(log_msg)

        if epoch > params.fast_epochs:
            skip_diff_train = False
        if epoch % params.save_every > 0:
            continue

        save_path = f"{params.log_dir}/grad_tts_{epoch}.pt"
        torch.save({
            'model': model.state_dict(),
            'optim': optim.state_dict(),
            'epoch': epoch,
            'steps': iteration,
        }, save_path)
        print("Saved checkpoint to: %s" % save_path)
