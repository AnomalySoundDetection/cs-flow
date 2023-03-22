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


logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')

def load_audio(audio_path, sample_rate):
    audio_data, _ = librosa.core.load(audio_path, sr=sample_rate)
    audio_data = torch.FloatTensor(audio_data)
    return audio_data

class AudioDataset(Dataset):
    def __init__(self, data, _id, root, sample_rate=c.sample_rate, n_scales=c.n_scales, train=True):
        if train:
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
        # label_list = []
        for i in range(1, self.n_scales+1):
            audio_data = load_audio(f"{file_path}", sample_rate=self.sample_rate * i)
            a_audio.append(audio_data)
            # label_list.append(self.labels[index])
        # a_audio = transforms.Compose(a_audio)
        a_audio=torch.cat(a_audio, dim=0)
        # label_list = torch.cat(label_list , dim=0)
        if  self.train:
            return a_audio
        else:
            if self.labels[index] == 0:
                target = torch.zeros([1,3])
            else:
                target = torch.ones([1,3])
            return audio_data, target   
    
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
