# Adapted from https://github.com/jik876/hifi-gan under the MIT license.
#   LICENSE is in incl_licenses directory.

from __future__ import absolute_import, division, print_function, unicode_literals

import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob
import argparse
import json
import torch
import librosa

from scipy.io.wavfile import write
from bigvgan.env import AttrDict
from bigvgan.models import BigVGAN as Generator


h = None
device = None
torch.backends.cudnn.benchmark = False

# from meldataset import mel_spectrogram, MAX_WAV_VALUE
from grad_extend.spec import mel_spectrogram, MAX_WAV_VALUE


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    # return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
    return mel_spectrogram(x)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a, h):
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    filelist = os.listdir(a.input_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    with torch.no_grad():
        for i, filname in enumerate(filelist):
            # load the ground truth audio and resample if necessary
            wav, sr = librosa.load(os.path.join(a.input_wavs_dir, filname), h.sampling_rate, mono=True)
            wav = torch.FloatTensor(wav).to(device)
            # compute mel spectrogram from the ground truth audio
            x = get_mel(wav.unsqueeze(0))
            print(filname)
            print(x.shape)

            y_g_hat = generator(x)

            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated.wav')
            write(output_file, h.sampling_rate, audio)
            print(output_file)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')

    a = parser.parse_args()
    a.checkpoint_file = os.path.join('./bigvgan_pretrain', 'g_05000000')

    config_file = os.path.join('./bigvgan_pretrain', 'config.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a, h)


if __name__ == '__main__':
    main()

