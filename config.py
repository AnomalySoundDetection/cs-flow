'''This file configures the training procedure because handling arguments in every single function is so exhaustive for
research purposes. Don't try this code if you are a software engineer.'''
# TODO: be clear
model_directory = "./model"
mode = True
latent_size = 2048

# feature 
n_mels = 64
sample_rate = 16000
sr_list = [16000, 32000, 64000]
sr = sample_rate
n_fft = 1024
hop_length = 256
fmin = 50
fmax = [8000, 16000, 32000]
frame_length = 25
shift_length = 10
# device settings
device = 'cuda:0'  # or 'cpu'

dataset_path = "/mnt/HDD2/ASD_team/dev_data"
# evl_dataset_path = "/mnt/HDD2/ASD_team/dev_data"

modelname = "dummy_test"  # export evaluations/logs with this name

dev_directory = "/mnt/HDD2/ASD_team/dev_data"
eval_directory = "/mnt/HDD2/ASD_team/test_data"
model_directory = "/mnt/HDD2/ASD_team/ting/cs-flow/model"
result_directory = "/mnt/HDD2/ASD_team/ting/cs-flow/result"
checkpoint_directory = "/mnt/HDD2/ASD_team/ting/cs-flow/checkpoint"
score_export_dir = "./viz/scores"
machine_type = [ "slider", "pump", "valve", "fan"]
# pre_extracted = False  # were feature preextracted with extract_features?
pre_extracted = True  # were feature preextracted with extract_features?

# frame_size = (96000, 0)
# # shape: 320000, 640000, 960000

# # img_dims = [3] + list(img_size)
# img_dims = [3] + list(frame_size)

# transformation settings
norm_mean, norm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# network hyperparameters
n_scales = 3  # number of scales at which features are extracted, img_size is the highest - others are //2, //4,...
clamp = 3  # clamping parameter
max_grad_norm = 1e0  # clamp gradients to this norm
n_coupling_blocks = 4  # higher = more flexible = more unstable
fc_internal = 1024  # * 4 # number of neurons in hidden layers of s-t-networks
lr_init = 2e-4  # inital learning rate
use_gamma = True
max_fpr = 0.1
# extractor = "effnetB5"  # feature dataset name (which was used in 'extract_features.py' as 'export_name')

# FIXME: n_feat = 1024 (PANNs ver)
# n_feat = 512
n_feat = 256
# n_feat = {"effnetB5": 304}[extractor]  # dependend from feature extractor
# map_size = (img_size[0] // 32, img_size[1] // 32)
# map_size = (frame_size[0] // 32, frame_size[1] // 32)
map_size = [32, 32]

# dataloader parameters
batch_size = 4  # actual batch size is this value multiplied by n_transforms(_test)
kernel_sizes = [3] * (n_coupling_blocks - 1) + [5]

# total epochs = meta_epochs * sub_epochs
# evaluation after <sub_epochs> epochs
meta_epochs = 20  # total epochs = meta_epochs * sub_epochs
# sub_epochs = 60  # evaluate after this number of epochs

# output settings
verbose = True
hide_tqdm_bar = True
save_model = True

# audio config
audio_conf = {'num_mel_bins': 64, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'dcase', 
                'mode': 'train', 'mean': -4.2677393, 'std': 4.5689974, 'noise': False, 'sample_rate': 16000}

val_audio_conf = {'num_mel_bins': 64, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'dcase', 
                'mode': 'train', 'mean': -4.2677393, 'std': 4.5689974, 'noise': False, 'sample_rate': 16000}

test_audio_conf = {'num_mel_bins': 64, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'dcase', 
                'mode': 'test', 'mean': -4.2677393, 'std': 4.5689974, 'noise': False, 'sample_rate': 16000}