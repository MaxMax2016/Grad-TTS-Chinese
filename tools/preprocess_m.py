import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
import librosa

from tqdm import tqdm
from grad_extend.spec import mel_spectrogram


def extract_mel(wav_in, mel_out):
    device = torch.device('cpu')
    # load the ground truth audio and resample if necessary
    wav, sr = librosa.load(wav_in, sr=24000, mono=True)
    wav = torch.FloatTensor(wav).to(device)
    # compute mel spectrogram from the ground truth audio
    mel = mel_spectrogram(wav.unsqueeze(0))
    mel = torch.squeeze(mel, 0)
    torch.save(mel, mel_out)


def process_files(wavPath, outPath):
    files = [f for f in os.listdir(f"./{wavPath}") if f.endswith(".wav")]
    for file in tqdm(files, desc="Extract Mel"):
        file = file[:-4]
        extract_mel(f"{wavPath}/{file}.wav", f"{outPath}/{file}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", help="wav", dest="wav", required=True)
    parser.add_argument("--out", help="out", dest="out", required=True)

    args = parser.parse_args()
    print(args.wav)
    print(args.out)

    os.makedirs(args.out, exist_ok=True)
    wavPath = args.wav
    outPath = args.out

    process_files(wavPath, outPath)
