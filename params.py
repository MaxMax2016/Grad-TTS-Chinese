from grad.utils import fix_len_compatibility


seed = 37
# data parameters
train_filelist_path = './data/files/train.txt'
valid_filelist_path = './data/files/valid.txt'

n_spks = 1
n_feats = 100
spk_emb_dim = 64

#feature_extractor:
#  class_path: vocos.feature_extractors.MelSpectrogramFeatures
#  init_args:
#    sample_rate: 24000
#    n_fft: 1024
#    hop_length: 256
#    n_mels: 100
#    padding: center

# encoder parameters
n_enc_channels = 192
filter_channels = 768
filter_channels_dp = 256
n_enc_layers = 6
enc_kernel = 3
enc_dropout = 0.1
n_heads = 2
window_size = 4

# decoder parameters
dec_dim = 64
beta_min = 0.05
beta_max = 20.0
pe_scale = 1000  # 1 for `grad-tts-old.pt` checkpoint

# training parameters
full_epochs = 900
fast_epochs = 100

log_dir = 'logs/new_exp'
test_size = 4
batch_size = 8
learning_rate = 1e-4

save_every = 5
out_size = fix_len_compatibility(2*24000//256)
