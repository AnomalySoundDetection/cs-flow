import os
import torch
from torchvision import datasets, transforms
import config as c
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import numpy as np
import librosa

import os
import torchaudio
from torch.utils.data import Dataset


# class AudioFolder(Dataset):
#     def __init__(self, root, transform=None, target_transform=None, n_scales=c.n_scales):
#         self.root = root
#         self.transform = transform
#         self.target_transform = target_transform
#         self.extensions = torchaudio.get_audio_backend_extensions()
#         self.file_list = self._get_file_list()
#         self.n_scales = n_scales

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, index):
#         file_path = self.file_list[index]
#         waveform, sample_rate = torchaudio.load(file_path)
#         if self.transform is not None:
#             waveform = self.transform(waveform)
#         if self.target_transform is not None:
#             target = self.target_transform(os.path.dirname(file_path))
#         else:
#             target = os.path.dirname(file_path)
#         return waveform, target

#     def _get_file_list(self):
#         file_list = []
#         for root, _, files in sorted(os.walk(self.root)):
#             for file_name in sorted(files):
#                 file_path = os.path.join(root, file_name)
#                 if torchaudio.get_audio_backend(file_path).lower() in self.extensions:
#                     file_list.append(file_path)
#         return file_list


def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def flat(tensor):
    return tensor.reshape(tensor.shape[0], -1)


def concat_maps(maps):
    flat_maps = list()
    for m in maps:
        flat_maps.append(flat(m))
    return torch.cat(flat_maps, dim=1)[..., None]


def get_loss(z, jac):
    z = torch.cat([z[i].reshape(z[i].shape[0], -1) for i in range(len(z))], dim=1)
    jac = sum(jac)
    return torch.mean(0.5 * torch.sum(z ** 2, dim=(1,)) - jac) / z.shape[1]


def cat_maps(z):
    return torch.cat([z[i].reshape(z[i].shape[0], -1) for i in range(len(z))], dim=1)


def load_datasets(dataset_path, class_name):
    # def target_transform(target):
    #     return class_perm[target]
    # trainset = []
    # testset = []
    # # print("pre_extracted is ", c.pre_extracted)
    # if c.pre_extracted:
    trainset = AudioDataset(train=True, root=dataset_path, machine=class_name)
    testset = AudioDataset(train=False, root=dataset_path, machine=class_name)
    # else:
    #     data_dir_train = os.path.join(dataset_path, class_name, 'train')
    #     data_dir_test = os.path.join(dataset_path, class_name, 'test')

    #     classes = os.listdir(data_dir_test)
    #     classes.sort()
    #     class_perm = list()
    #     class_idx = 1
    #     for cl in classes:
    #         if 'normal' in cl:
    #             class_perm.append(0)
    #         else:
    #             class_perm.append(class_idx)

    #     file_paths, targets = [], []
    #     for root, dirs, files in os.walk(data_dir_train):
    #         for f in files:
    #             file_path = os.path.join(root, file_path)
    #             label = class_perm[int('normal' not i file_path)]
    #             file_paths.append(file_path)
    #             targets.append(label)

    #     # TODO: resize -> change frames size
    #     tfs = [transforms.Resize(c.img_size), transforms.ToTensor(), transforms.Normalize(c.norm_mean, c.norm_std)]
    #     transform_train = transforms.Compose(tfs)

    #     trainset = ImageFolder(data_dir_train, transform=transform_train)
    #     testset = ImageFolder(data_dir_test, transform=transform_train, target_transform=target_transform)
    # # print(len(trainset))
    return trainset, testset



class AudioDataset(Dataset):
    def __init__(self, root="data/features/", machine="fan", n_scales=c.n_scales, train=False, n_mfcc=20, n_fft=512, hop_length=256):
        super(AudioDataset, self).__init__()
        self.data = list()
        self.n_scales = n_scales
        self.train = train
        self.machine = machine
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.dataset_path = os.path.join(root, machine)
        suffix = 'train' if train else 'test'
    
    def _load_files(self):
        data_dir_train = os.path.join(self.dataset_path, self.class_name, 'train')
        data_dir_test = os.path.join(self.dataset_path, self.class_name, 'test')

        classes = os.listdir(data_dir_test)
        classes.sort()
        class_perm = list()
        class_idx = 1
        for cl in classes:
            if 'normal' in cl:
                class_perm.append(0)
            else:
                class_perm.append(class_idx)

        self.file_paths, self.targets = [], []
        for root, dirs, files in os.walk(data_dir_train):
            for file in files:
                file_path = os.path.join(root, file)
                label = class_perm[int('normal' not in file_path)]
                self.file_paths.append(file_path)
                self.targets.append(label)

        return self.file_paths, self.targets

        # self.feature_root = os.path.join(root, machine)
        # os.makedirs(self.feature_root, exist_ok=True)
        
        # for s in range(c.n_scales):
        #     self.data.append(np.load(root + c.class_name + "_scale_" + str(s) + "_" + suffix + ".npy"))

        # audio_folder = os.path.join(root, machine, suffix)
        # # audio_files = os.listdir(audio_folder, '.wav')
        # audio_files = [os.path.join(audio_folder, f) for f in os.listdir(audio_folder) if os.path.isfile(os.path.join(audio_folder, f))]
        # print("files: ", audio_files[:4])
        # features = []
        # for audio_file in audio_files[:4]:
        #     y, sr = librosa.load(audio_file)

        #     # Extract three different scales of MFCCs
        #     for n_mfcc in [10, 20, 30]:
        #         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        #         features.append(mfccs)

        # features = np.array(features, dtype=object)
        # print(os.path.join(self.feature_root, f"{machine}_{suffix}.npy"))
        # np.save(os.path.join(self.feature_root, f"{machine}_{suffix}.npy"), features)
        # self.paths = np.array(audio_files)
        # self.labels = [audio_path.split('_')[-2] for audio_path in self.paths]
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        # out = list()
        # for d in self.data:
        #     y, sr = librosa.load(d[index])
        #     # Extract three different scales of MFCCs
        #     for n_mfcc in [10, 20, 30]:
        #         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        #         mfccs = torch.FloatTensor(mfccs)
        #         out.append(mfccs)
        # # print("result:", out[:5])
        # return out, self.labels[index]
        file_path = self.file_paths[index]
        target = self.targets[index]

        signal, sr = librosa.load(file_path, sr=None, mono=True)
        mfccs = librosa.feature.mfcc(signal, sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length)
        mfccs = mfccs.astype('float32')
        return mfccs, target
    
class FeatureDataset(Dataset):
    def __init__(self, root="data/features/" + c.class_name + '/', n_scales=c.n_scales, train=False):
        super(FeatureDataset, self).__init__()
        self.data = list()
        self.n_scales = n_scales
        self.train = train
        suffix = 'train' if train else 'test'

        for s in range(c.n_scales):
            self.data.append(np.load(root + c.class_name + '_scale_' + str(s) + '_' + suffix + '.npy'))

        self.labels = np.load(os.path.join(root, c.class_name + '_labels.npy')) if not train else np.zeros(
            [len(self.data[0])])
        self.paths = np.load(os.path.join(root, c.class_name + '_image_paths.npy'))
        self.class_names = [img_path.split('/')[-2] for img_path in self.paths]

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        out = list()
        for d in self.data:
            sample = d[index]
            sample = torch.FloatTensor(sample)
            out.append(sample)
        return out, self.labels[index]


def make_dataloaders(trainset, testset):
    trainloader = torch.utils.data.DataLoader(trainset, pin_memory=True, batch_size=c.batch_size, shuffle=True,
                                              drop_last=False)
    testloader = torch.utils.data.DataLoader(testset, pin_memory=True, batch_size=c.batch_size, shuffle=False,
                                             drop_last=False)
    return trainloader, testloader


def preprocess_batch(data):
    '''move data to device and reshape image'''
    if c.pre_extracted:
        inputs, labels = data
        for i in range(len(inputs)):
            inputs[i] = inputs[i].to(c.device)
        labels = labels.to(c.device)
    else:
        inputs, labels = data
        inputs, labels = inputs.to(c.device), labels.to(c.device)
        inputs = inputs.view(-1, *inputs.shape[-3:])
    return inputs, labels


class Score_Observer:
    '''Keeps an eye on the current and highest score so far'''

    def __init__(self, name):
        self.name = name
        self.max_epoch = 0
        self.max_score = None
        self.min_loss_epoch = 0
        self.min_loss_score = 0
        self.min_loss = None
        self.last = None
    def update(self, score, epoch, print_score=False):
        self.last = score
        if self.max_score == None or score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
        if print_score:
            self.print_score()

    def print_score(self):
        print('{:s}: \t last: {:.4f} \t max: {:.4f} \t epoch_max: {:d} \t epoch_loss: {:d}'.format(self.name, self.last,
                                                                                                   self.max_score,
                                                                                                   self.max_epoch,
                                                                                                   self.min_loss_epoch))