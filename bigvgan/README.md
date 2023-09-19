## BigVGAN: A Universal Neural Vocoder with Large-Scale Training
#### Sang-gil Lee, Wei Ping, Boris Ginsburg, Bryan Catanzaro, Sungroh Yoon

<center><img src="https://user-images.githubusercontent.com/15963413/218609148-881e39df-33af-4af9-ab95-1427c4ebf062.png" width="800"></center>


### [Paper](https://arxiv.org/abs/2206.04658)
### [Audio demo](https://bigvgan-demo.github.io/)


## Pretrained Models
We provide the [pretrained models](https://drive.google.com/drive/folders/1e9wdM29d-t3EHUpBb8T4dcHrkYGAXTgq).
One can download the checkpoints of generator (e.g., g_05000000) and discriminator (e.g., do_05000000) within the listed folders.

|Folder Name|Sampling Rate|Mel band|fmax|Params.|Dataset|Fine-Tuned|
|------|---|---|---|---|------|---|
|bigvgan_24khz_100band|24 kHz|100|12000|112M|LibriTTS|No|
|bigvgan_base_24khz_100band|24 kHz|100|12000|14M|LibriTTS|No|
|bigvgan_22khz_80band|22 kHz|80|8000|112M|LibriTTS + VCTK + LJSpeech|No|
|bigvgan_base_22khz_80band|22 kHz|80|8000|14M|LibriTTS + VCTK + LJSpeech|No|

The paper results are based on 24kHz BigVGAN models trained on LibriTTS dataset.
We also provide 22kHz BigVGAN models with band-limited setup (i.e., fmax=8000) for TTS applications.
Note that, the latest checkpoints use ``snakebeta`` activation with log scale parameterization, which have the best overall quality.


## TODO
Current codebase only provides a plain PyTorch implementation for the filtered nonlinearity. We are working on a fast CUDA kernel implementation, which will be released in the future. 


## References
* [HiFi-GAN](https://github.com/jik876/hifi-gan) (for generator and multi-period discriminator)

* [Snake](https://github.com/EdwardDixon/snake) (for periodic activation)

* [Alias-free-torch](https://github.com/junjun3518/alias-free-torch) (for anti-aliasing)

* [Julius](https://github.com/adefossez/julius) (for low-pass filter)

* [UnivNet](https://github.com/mindslab-ai/univnet) (for multi-resolution discriminator)