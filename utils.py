import glob
import os
import torch
import config as c
import numpy as np
import librosa
import logging
import os
import torchaudio
import csv

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

def test_file_list_generator(target_dir,
                             id_name,
                             dir_name="test",
                             prefix_normal="normal",
                             prefix_anomaly="anomaly",
                             ext="wav",
                             mode=True):
    # development
    if mode:
        normal_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                 dir_name=dir_name,
                                                                                 prefix_normal=prefix_normal,
                                                                                 id_name=id_name,
                                                                                 ext=ext)))
        normal_labels = np.zeros(len(normal_files))
        anomaly_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                  dir_name=dir_name,
                                                                                  prefix_anomaly=prefix_anomaly,
                                                                                  id_name=id_name,
                                                                                  ext=ext)))
        anomaly_labels = np.ones(len(anomaly_files))
        files = np.concatenate((normal_files, anomaly_files), axis=0)
        labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
        if len(files) == 0:
            print("no_wav_file!!")
        print("\n========================================")

    # evaluation
    else:
        files = sorted(
            glob.glob("{dir}/{dir_name}/*{id_name}*.{ext}".format(dir=target_dir,
                                                                  dir_name=dir_name,
                                                                  id_name=id_name,
                                                                  ext=ext)))
        labels = None

        if len(files) == 0:
            print("no_wav_file!!")
        print("\n=========================================")

    return files, labels

def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)