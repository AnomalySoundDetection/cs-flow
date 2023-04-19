import glob
import os
import torch
from torchvision import datasets, transforms
import config as c
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import numpy as np
import librosa
import logging
import os
import torchaudio
from torch.utils.data import Dataset

logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')

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


# def preprocess_batch(data):
#     '''move data to device and reshape image'''
#     if c.pre_extracted:
#         inputs, labels = data
#         for i in range(len(inputs)):
#             inputs[i] = inputs[i].to(c.device)
#         labels = labels.to(c.device)
#     else:
#         inputs, labels = data
#         inputs, labels = inputs.to(c.device), labels.to(c.device)
#         inputs = inputs.view(-1, *inputs.shape[-3:])
#     return inputs, labels
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

########################################################################
# get the list of wave file paths
########################################################################
def file_list_generator(target_dir,
                        id,
                        dir_name,
                        mode,
                        prefix_normal="normal",
                        prefix_anomaly="anomaly",
                        ext="wav"):
    """
    target_dir : str
        base directory path
    section_name : str
        section name of audio file in <<dir_name>> directory
    dir_name : str
        sub directory name
    prefix_normal : str (default="normal")
        normal directory name
    prefix_anomaly : str (default="anomaly")
        anomaly directory name
    ext : str (default="wav")
        file extension of audio files

    return :
        if the mode is "development":
            files : list [ str ]
                audio file list
            labels : list [ boolean ]
                label info. list
                * normal/anomaly = 0/1
        if the mode is "evaluation":
            files : list [ str ]
                audio file list
    """
    logger.info("target_dir : {}".format(target_dir+'\\'))

    # print(dir_name, target_dir)
    query = os.path.abspath("{target_dir}/{dir_name}/{prefix_normal}_{id}_*.{ext}".format(target_dir=target_dir,
                                                                                dir_name=dir_name,
                                                                                id=id,
                                                                                prefix_normal=prefix_normal,
                                                                                ext=ext))
    print(query)
    normal_files = sorted(glob.glob(query))
    print('normal files #:', len(normal_files))
    normal_labels = np.zeros(len(normal_files))

    query = os.path.abspath("{target_dir}/{dir_name}/{prefix_normal}_{id}_*.{ext}".format(target_dir=target_dir,
                                                                                                    dir_name=dir_name,
                                                                                                    id=id,
                                                                                                    prefix_normal=prefix_anomaly,
                                                                                                    ext=ext))
    anomaly_files = sorted(glob.glob(query))
    print('anomaly files #:', len(anomaly_files))
    anomaly_labels = np.ones(len(anomaly_files))

    files = np.concatenate((normal_files, anomaly_files), axis=0)
    labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
    
    logger.info("#files : {num}".format(num=len(files)))
    if len(files) == 0:
        logger.exception("no_wav_file!!")
    print("\n========================================")

    return files, labels