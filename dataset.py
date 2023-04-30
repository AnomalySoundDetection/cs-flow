import itertools
import re
from torch.utils.data import Dataset
from torchvision import transforms
import librosa
import torch
import config as c
import os
import glob
from utils import file_list_generator, test_file_list_generator
import numpy as np


def load_audio(audio_path, sample_rate):
    audio_data, _ = librosa.load(audio_path, sr=sample_rate)
    audio_data = (audio_data - audio_data.mean()) / audio_data.std()
    audio_data = torch.FloatTensor(audio_data)
    return audio_data


class AudioDataset(Dataset):
    def __init__(self, _id=None, root=None, sample_rate=c.sample_rate, n_scales=c.n_scales, train=True):
        self.labels = []
        if train:
            self.files, self.labels = file_list_generator(
                                        target_dir=root,
                                        id=_id,
                                        dir_name="train",
                                        mode=True
                                    )
        else:
            # root = root + "/test"
            self.files, self.labels = test_file_list_generator(
                                        target_dir=root,
                                        id_name=_id,
                                        dir_name="test",
                                        mode=True
                                    )
        self.train = train
        print("data len", len(self.files))
        # print("files:", self.files[0], self.files[-1])
        self.sample_rate = sample_rate
        self.n_scales = n_scales
        
    def __getitem__(self, index):
        file_path = self.files[index]
        a_audio = []

        for i in range(0, self.n_scales):
            audio_data = load_audio(f"{file_path}", sample_rate=c.sr_list[i])
            a_audio.append(audio_data)
        # a_audio = load_audio(f"{file_path}", sample_rate=self.sample_rate)

        if  self.train:
            return a_audio
        else:
            # return a_audio, self.labels[index]   
            return {'audio': a_audio, 'label': self.labels[index]}
        # return audio_data
    def __len__(self):
        return len(self.files)
