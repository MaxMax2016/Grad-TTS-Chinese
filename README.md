# Huawei Grad-TTS for Chinese, integrated Bert for rhyme and integrated BigVGAN as vocoder

#### 用于学习的TTS算法项目，如果您在寻找直接用于生产的TTS，本项目并不适合您！
<div align="center">

![grad_tts](assets/grad_tts.jpg)

![bert_grad_tts](assets/bert_grad_tts.jpg)
Framework
</div>

## Acoustic Model

### Install and Test

download [bigvgan_base_24khz_100band](https://drive.google.com/drive/folders/1e9wdM29d-t3EHUpBb8T4dcHrkYGAXTgq) from https://github.com/NVIDIA/BigVGAN

download [prosody_model](https://github.com/Executedone/Chinese-FastSpeech2) from [Executedone/Chinese-FastSpeech2](https://github.com/Executedone/Chinese-FastSpeech2)

download [grad_tts.pt](https://github.com/PlayVoice/Bert-Grad-Vocos-TTS/releases/tag/release) from release page

put [g_05000000]() To ./bigvgan_pretrain/g_05000000

rename best_model.pt to prosody_model.pt

put [prosody_model.pt]() To ./bert/prosody_model.pt

put [grad_tts.pt]() To ./grad_tts.pt

> pip install -r requirements.txt

```
> cd ./grad/monotonic_align
> python setup.py build_ext --inplace
> cd -
```

> python inference.py --file test.txt --checkpoint grad_tts.pt --timesteps 4 --temperature 1.15

the waves infered will be saved in `./inference_out`

**if `timesteps` is set to 0, then diffusion will be skipped.**

### Data

download [baker](https://aistudio.baidu.com/datasetdetail/36741) data: https://www.data-baker.com/data/index/TNtts/

put `Waves` to ./data/Waves

put `000001-010000.txt` to ./data/000001-010000.txt

1, resample

> python tools/preprocess_a.py -w ./data/Wave/ -o ./data/wavs -s `24000`

2, extract mel

> python tools/preprocess_m.py --wav data/wavs/ --out data/mels/

3, extract bert, and generate train files by the way

> python tools/preprocess_b.py

output contains `data/berts/` and `data/files`

注意：打印信息，是在剔除`儿化音`（项目为算法演示，不做生产）

Raw label
``` c
000001	卡尔普#2陪外孙#1玩滑梯#4。
	ka2 er2 pu3 pei2 wai4 sun1 wan2 hua2 ti1
000002	假语村言#2别再#1拥抱我#4。
	jia2 yu3 cun1 yan2 bie2 zai4 yong1 bao4 wo3
```
Cleaned label
``` c
000001	卡尔普陪外孙玩滑梯。
	ka2 er2 pu3 pei2 wai4 sun1 wan2 hua2 ti1
	sil k a2 ^ er2 p u3 p ei2 ^ uai4 s uen1 ^ uan2 h ua2 t i1 sp sil
000002	假语村言别再拥抱我。
	jia2 yu3 cun1 yan2 bie2 zai4 yong1 bao4 wo3
	sil j ia2 ^ v3 c uen1 ^ ian2 b ie2 z ai4 ^ iong1 b ao4 ^ uo3 sp sil
```
Train files
```
./data/wavs/000001.wav|./data/mels/000001.pt|./data/berts/000001.npy|sil k a2 ^ er2 p u3 p ei2 ^ uai4 s uen1 ^ uan2 h ua2 t i1 sp sil
./data/wavs/000002.wav|./data/mels/000002.pt|./data/berts/000002.npy|sil j ia2 ^ v3 c uen1 ^ ian2 b ie2 z ai4 ^ iong1 b ao4 ^ uo3 sp sil
```
Error
```
002365	这图#2难不成#2是#1Ｐ过的#4？
	zhe4 tu2 nan2 bu4 cheng2 shi4 P IY1 guo4 de5
```
### Train

debug train

> python tools/preprocess_d.py

start train

> python train.py

resume train

> python train.py -p logs/new_exp/grad_tts_***.pt

### Inference

> python inference.py --file test.txt --checkpoint ./logs/new_exp/grad_tts_***.pt --timesteps 20 --temperature 1.15

### Code sources and references

https://github.com/huawei-noah/Speech-Backbones/blob/main/Grad-TTS

https://github.com/thuhcsi/LightGrad

https://github.com/Executedone/Chinese-FastSpeech2

https://github.com/PlayVoice/vits_chinese

# Raw Grad-TTS information

Official implementation of the Grad-TTS model based on Diffusion Probabilistic Modelling. For all details check out our paper accepted to ICML 2021 via [this](https://arxiv.org/abs/2105.06337) link.

**Authors**: Vadim Popov\*, Ivan Vovk\*, Vladimir Gogoryan, Tasnima Sadekova, Mikhail Kudinov.

<sup>\*Equal contribution.</sup>

## Abstract

**Demo page** with voiced abstract: [link](https://grad-tts.github.io/).

Recently, denoising diffusion probabilistic models and generative score matching have shown high potential in modelling complex data distributions while stochastic calculus has provided a unified point of view on these techniques allowing for flexible inference schemes. In this paper we introduce Grad-TTS, a novel text-to-speech model with score-based decoder producing mel-spectrograms by gradually transforming noise predicted by encoder and aligned with text input by means of Monotonic Alignment Search. The framework of stochastic differential equations helps us to generalize conventional diffusion probabilistic models to the case of reconstructing data from noise with different parameters and allows to make this reconstruction flexible by explicitly controlling trade-off between sound quality and inference speed. Subjective human evaluation shows that Grad-TTS is competitive with state-of-the-art text-to-speech approaches in terms of Mean Opinion Score.

## References

* HiFi-GAN model is used as vocoder, official github repository: [link](https://github.com/jik876/hifi-gan).
* Monotonic Alignment Search algorithm is used for unsupervised duration modelling, official github repository: [link](https://github.com/jaywalnut310/glow-tts).
* Phonemization utilizes CMUdict, official github repository: [link](https://github.com/cmusphinx/cmudict).


## Vocoder Model

project link: https://github.com/NVIDIA/BigVGAN

### Infer Test

dowdload pretrain model [bigvgan_base_24khz_100band](https://drive.google.com/drive/folders/19WyD7wN3BeIwBtr9ei1bBcdNEuiq_Avr)

```shell
python bigvgan/inference.py \
--input_wavs_dir bigvgan_debug \
--output_dir bigvgan_out
```

## Train with baker

> python bigvgan/train.py --config bigvgan_pretrain/config.json

## References
* [HiFi-GAN](https://github.com/jik876/hifi-gan) (for generator and multi-period discriminator)

* [Snake](https://github.com/EdwardDixon/snake) (for periodic activation)

* [Alias-free-torch](https://github.com/junjun3518/alias-free-torch) (for anti-aliasing)

* [Julius](https://github.com/adefossez/julius) (for low-pass filter)

* [UnivNet](https://github.com/mindslab-ai/univnet) (for multi-resolution discriminator)
