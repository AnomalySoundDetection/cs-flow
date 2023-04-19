import itertools
import re
from torch.utils.data import Dataset
from torchvision import transforms
import librosa
import torch
import config as c
import os
import glob
import logging
from utils import file_list_generator
import numpy as np


logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')

def load_audio(audio_path, sample_rate):
    # audio_data, _ = librosa.core.load(audio_path, sr=sample_rate)
    audio_data, _ = librosa.load(audio_path, sr=sample_rate)
    # mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_fft=c.n_fft, hop_length=c.hop_length,
    #                                        n_mels=c.n_mels, fmin=c.fmin, fmax=c.fmax)


    audio_data = torch.FloatTensor(audio_data)
    # mel_spec = torch.FloatTensor(mel_spec)
    # mel_spec = mel_spec.unsqueeze(0)
    # S_dB = librosa.power_to_db(mel_spec, ref=np.max)
    # print("mel shape is:", mel_spec.shape)
    # print("shape is:", S_dB.shape)

    # time_steps = mel_spec.shape[1]
    # mel_spec = mel_spec[:, :time_steps//32*32]
    # mel_spec = mel_spec.reshape(c.n_mels, -1, 32)
    # print("shape is:", mel_spec.shape)
    # print("audio_data shape is:", audio_data.shape)
    return audio_data


class AudioDataset(Dataset):
    def __init__(self, data=None, _id=None, root=None, sample_rate=c.sample_rate, n_scales=c.n_scales, train=True):
        if data:
            self.files = data
        elif train:
            # root = root + "/train"
            self.files, self.labels = file_list_generator(
                target_dir=root,
                id=_id,
                dir_name="train",
                mode=True
            )
        else:
            # root = root + "/test"
            self.files, self.labels = file_list_generator(
            target_dir=root,
            id=_id,
            dir_name="test",
            mode=True
            )
        self.train = train

        # self.data = [sample for sample in data if _id in sample]
        print("data len", len(self.files))
        self.sample_rate = sample_rate
        self.n_scales = n_scales
        
    def __getitem__(self, index):
        file_path = self.files[index]
        a_audio = []

        for i in range(0, self.n_scales):
            audio_data = load_audio(f"{file_path}", sample_rate=c.sr_list[i])
            a_audio.append(audio_data)
        # a_audio = load_audio(f"{file_path}", sample_rate=self.sample_rate)

        # a_audio=torch.cat(a_audio, dim=0)
        # label_list = torch.cat(label_list , dim=0)
        if  self.train:
            return a_audio
        else:
            if self.labels[index] == 0:
                target = torch.zeros([1,3])
            else:
                target = torch.ones([1,3])
            return a_audio, target   
    
        # return audio_data
    def __len__(self):
        return len(self.files)
    
def get_machine_id_list(target_dir,
                        dir_type="test",
                        ext="wav"):
    """
    target_dir : str
        base directory path of "dev_data" or "eval_data"
    test_dir_name : str (default="test")
        directory containing test data
    ext : str (default="wav)
        file extension of audio files
    return :
        machine_id_list : list [ str ]
            list of machine IDs extracted from the names of test files
            ex. id_00, id_02
    """
    # create test files
    dir_path = os.path.abspath("{dir}/{dir_type}/*.{ext}".format(dir=target_dir, dir_type=dir_type, ext=ext))
    #print(dir_path)
    file_paths = sorted(glob.glob(dir_path))
    #print(file_paths)
    # extract id
    machine_id_list = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    
    return machine_id_list

def select_dirs(machine, mode=True, dir_type="train"):
    """
    param : dict
        baseline.yaml data

    return :
        if active type the development :
            dirs :  list [ str ]
                load base directory list of dev_data
        if active type the evaluation :
            dirs : list [ str ]
                load base directory list of eval_data
    """
    if mode:
        logger.info("load_directory <- development")
        dir_path = os.path.abspath("{base}/{machine}/{dir_type}/*".format(base=c.dataset_path, machine=machine, dir_type=dir_type))
        dirs = sorted(glob.glob(dir_path))
    else:
        # FIXME: no evl dataset
        logger.info("load_directory <- evaluation")
        dir_path = os.path.abspath("{base}/{machine}/{dir_type}/*".format(base=c.dataset_path, machine=machine, dir_type=dir_type))
        dirs = sorted(glob.glob(dir_path))
    return dirs
