"""
 @file   01_test.py
 @brief  Script for test
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import default python-library
########################################################################
import os
import glob
import csv
import re
import itertools
import sys
import torch
import gc

########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy as np
#import cupy as cp
# from import
from tqdm import tqdm
from sklearn import metrics
# original lib
import config as c
from torchsummary import summary
from PANN_model import load_extractor
from model import get_cs_flow_model
from utils import *
from dataset import *
import logging
from torch.utils.data import DataLoader
from copy import deepcopy
import matplotlib.pyplot as plt


logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
########################################################################

########################################################################
# 
########################################################################
def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


def get_machine_id_list_for_test(target_dir,
                                 dir_name="test",
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
    """
    # create test files
    dir_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(dir_path))
    # extract id
    machine_id_list = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return machine_id_list

def compare_histogram(scores, classes, machine_type, _id, thresh=2.5, n_bins=64):
    classes = deepcopy(classes)
    scores = deepcopy(scores)
    classes[classes > 0] = 1
    scores[scores > thresh] = thresh
    bins = np.linspace(np.min(scores), np.max(scores), n_bins)
    scores_norm = scores[classes == 0]
    scores_ano = scores[classes == 1]

    plt.clf()
    plt.hist(scores_norm, bins, alpha=0.5, density=True, label='non-defects', color='cyan', edgecolor="black")
    plt.hist(scores_ano, bins, alpha=0.5, density=True, label='defects', color='crimson', edgecolor="black")

    ticks = np.linspace(0.5, thresh, 5)
    labels = [str(i) for i in ticks[:-1]] + ['>' + str(thresh)]
    plt.xticks(ticks, labels=labels)
    plt.xlabel(r'$-log(p(z))$')
    plt.ylabel('Count (normalized)')
    plt.legend()
    plt.grid(axis='y')
    plt.savefig(os.path.join(c.score_export_dir, machine_type + _id + '_score_histogram.png'), bbox_inches='tight', pad_inches=0)
########################################################################


########################################################################
# main 01_test.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False

    # FIXME: cuda:0
    device = torch.device('cuda:1')
    # device = c.device

    mode = False
    if mode is None:
        sys.exit(-1)

    # make output result directory
    os.makedirs(c.model_directory, exist_ok=True)

    # # load base directory
    # dirs = select_dirs(machine="fan", mode=mode)
    machine_list = c.machine_type[:1]

    # initialize lines in csv for AUC and pAUC
    csv_lines = []

    # init extractor
    extractors = feature_extractor = load_extractor(sample_rate=c.sr_list,
                                                    window_size=c.n_fft,
                                                    hop_size=c.hop_length,
                                                    mel_bins=c.n_mels,
                                                    fmin=c.fmin,
                                                    fmax=c.fmax)
    extractor, extractor1, extractor2 = extractors[0].to(device=device), extractors[1].to(device=device), extractors[2].to(device=device)
    extractor.eval()
    extractor1.eval()
    extractor2.eval()
    for param in extractor.parameters():
        param.requires_grad = False
    for param in extractor1.parameters():
        param.requires_grad = False
    for param in extractor2.parameters():
        param.requires_grad = False


    # loop of the base directory
    for idx, machine_type in enumerate(machine_list):
        root_path = os.path.join(c.dev_directory, machine_type)
        id_list = get_machine_id_list(target_dir=root_path, dir_type="test")
        for _id in id_list:
            print("\n===========================")
            print("[{idx}/{total}] {target_dir}".format(target_dir=machine_type, idx=idx+1, total=len(machine_list)))

            print("============== MODEL LOAD ==============")
            # set model path
            '''
            model_file change to .pt
            '''
            print("machine id:", _id)
            model_file = "{model}/model_{machine_type}_{id}.pt".format(model=c.model_directory,
                                                                    machine_type=machine_type, id=_id)
            # load model file
            if not os.path.exists(model_file):
                logger.error("{} model not found ".format(machine_type))
                sys.exit(-1)
                # continue
            logger.info("model path: {}".format(model_file))
                
            model = get_cs_flow_model()
            model.load_state_dict(torch.load(model_file))
            
            model = model.to(device)

        # if mode:
            # results by type
            csv_lines.append([machine_type])
            csv_lines.append(["id", "AUC", "pAUC"])
            performance = []

            target_dir = select_dirs(machine_type, mode=False, dir_type="test")

        # for id_str in machine_id_list:
            # load test file
            # test_files, y_true = test_file_list_generator(c.dev_directory+"/"+machine_type, _id)

            # setup anomaly score file path
            anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{id_str}.csv".format(
                                                                                    result=c.result_directory,
                                                                                    machine_type=machine_type,
                                                                                    id_str=_id)
            anomaly_score_list = []
            test_labels = []

            print("\n============== BEGIN TEST FOR A MACHINE ID ==============")
            # y_pred = [0. for k in test_files]
            # print(test_files[0], root_path)
            # print("in dir", target_dir[500])
            test_dataset = AudioDataset(_id=_id, root=root_path, sample_rate=c.sample_rate, train=False)
            test_dl = DataLoader(dataset=test_dataset, batch_size=c.batch_size, shuffle=False)

            model.eval()
            with torch.no_grad():
                for batch_dict in tqdm(test_dl):
                    batch = batch_dict['audio']
                    labels  = batch_dict['label']

                    batch0 = batch[0].to(device)
                    batch1 = batch[1].to(device)
                    batch2 = batch[2].to(device)

                    f0 = extractor(batch0).to(device)
                    f1 = extractor1(batch1).to(device)
                    f2 = extractor2(batch2).to(device)
                    features = [f2, f1, f0]
                    z = model(features)
                    # z shape =  [[4, 256, 8, 8], [4, 256, 16, 16], [4, 256, 32, 32]]

                    # (z_concat): Merge multiple feature maps into one matrix (aggreate and flat)
                    # get the L2 norm of Z * 0.5
                    z_concat = t2np(concat_maps(z))
                    nll_score = np.mean(z_concat ** 2 / 2, axis=(1, 2))

                    anomaly_score_list.append(nll_score)
                    test_labels.append(t2np(labels))

                    del batch_dict, batch, labels, batch0, batch1, batch2, f0, f1, f2, z
                    gc.collect()
                    torch.cuda.empty_cache()
                    # if localize:
                    #     z_grouped = list()
                    #     likelihood_grouped = list()
                    #     for i in range(len(z)):
                    #         z_grouped.append(z[i].view(-1, *z[i].shape[1:]))
                    #         likelihood_grouped.append(torch.mean(z_grouped[-1] ** 2, dim=(1,)))
                    #     all_maps.extend(likelihood_grouped[0])
                    #     for i_l, l in enumerate(t2np(labels)):
                    #         # viz_maps([lg[i_l] for lg in likelihood_grouped], c.modelname + '_' + str(c.viz_sample_count), label=l, show_scales = 1)
                    #         c.viz_sample_count += 1
                #     sys.exit(-1)
                # sys.exit(-1)
            anomaly_score_list = np.concatenate(anomaly_score_list)
            # save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list.astype(np.float32))
            test_labels = np.concatenate(test_labels)
            # compare_histogram(anomaly_score_list, test_labels, machine_type=machine_type)

            # AUC
            auc = metrics.roc_auc_score(test_labels, anomaly_score_list)
            p_auc = metrics.roc_auc_score(test_labels, anomaly_score_list, max_fpr=c.max_fpr)
            print("AUC : {}".format(auc))
            print("pAUC : {}".format(p_auc))
            csv_lines.append([_id.split("_", 1)[1], auc, p_auc])
            performance.append([auc, p_auc])

            np.savetxt(anomaly_score_csv, anomaly_score_list.reshape(-1, anomaly_score_list.shape[-1]), delimiter=',')
            # save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
            
            compare_histogram(anomaly_score_list, test_labels, machine_type=machine_type, _id=_id)
            sys.exit(-1)