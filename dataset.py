import itertools
import re
from torch.utils.data import Dataset
import librosa
import torch
import config as c
import os
import glob
import logging

logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')

def load_audio(audio_path, sample_rate):
    audio_data, _ = librosa.core.load(audio_path, sr=sample_rate)
    audio_data = torch.FloatTensor(audio_data)

    return audio_data

class AudioDataset(Dataset):
    def __init__(self, data, _id, root, sample_rate=c.sample_rate, n_scales=c.n_scales, train=True):
        if train:
            root = root + "/train"
        else:
            root = root + "/test"
        self.data = [sample for sample in data if _id in sample]
        print("data len", len(self.data))
        self.root = root
        self.sample_rate = sample_rate
        self.n_scales = n_scales
        self.data_list = list()
        

    def __getitem__(self, index):
        file_path = self.data[index]
        print("file_path:", file_path)
        a_audio = list()
        for i in range(1, self.n_scales+1):
            audio_data = load_audio(f"{file_path}", sample_rate=self.sample_rate * i)
            a_audio.append(audio_data)

        self.data_list.append(a_audio)
            
        return a_audio
        # return audio_data
    def __len__(self):
        return len(self.data_list)
    
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



'''
root_path = c.dataset_path + "/" + c.class_name
print(root_path)
id_list = get_machine_id_list(target_dir=root_path, dir_type="train")
data = select_dirs(machine=c.machine_type[0])
# print("dataset_path:", data)
print(len(id_list))
for _id in id_list:
    train_dataset = AudioDataset(data=data, _id=_id, root=root_path, sample_rate=c.sample_rate, train=True)
    print(c.machine_type[0], _id, len(train_dataset))
    print(train_dataset[0][0])

    print("========================")
    print(len(train_dataset), len(train_dataset[0]))
'''
