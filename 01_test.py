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
from AST_model import load_extractor
from model import get_cs_flow_model
from utils import *
# from dataset import *
from dataset_v2 import *
import logging
from torch.utils.data import DataLoader
from copy import deepcopy
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG, filename="baseline.log")
logger = logging.getLogger(' ')
########################################################################
# function
########################################################################
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
    plt.savefig(os.path.join(c.score_export_dir, machine_type + "_" + _id + '_score_histogram.png'), bbox_inches='tight', pad_inches=0)

########################################################################
# main 01_test.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False

    # FIXME: cuda:0
    device = c.device

    mode = False
    if mode is None:
        sys.exit(-1)

    # make output result directory
    os.makedirs(c.model_directory, exist_ok=True)

    machine_list = c.machine_type

    # initialize lines in csv for AUC and pAUC
    csv_lines = []
    AUC_csv = "{result}/AUC_record.csv".format(result=c.result_directory)

    # init extractor
    # extractors = feature_extractor = load_extractor(sample_rate=c.sr_list,
    #                                                 window_size=c.n_fft,
    #                                                 hop_size=c.hop_length,
    #                                                 mel_bins=c.n_mels,
    #                                                 fmin=c.fmin,
    #                                                 fmax=c.fmax)
    # extractor = feature_extractor = load_extractor(tdim=1024, fdim=64, target_size=8).to(device=device)
    # extractor1 = feature_extractor = load_extractor(tdim=1024, fdim=128, target_size=16).to(device=device)
    # extractor2 = feature_extractor = load_extractor(tdim=1024, fdim=256, target_size=32).to(device=device)
    # extractor.eval()
    # extractor1.eval()
    # extractor2.eval()
    # for param in extractor.parameters():
    #     param.requires_grad = False
    # for param in extractor1.parameters():
    #     param.requires_grad = False
    # for param in extractor2.parameters():
    #     param.requires_grad = False


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
                # print("{} model not found ".format(machine_type))
                # sys.exit(-1)
                continue
            logger.info("model path: {}".format(model_file))
            
            test_model = torch.load(model_file)
            model = get_cs_flow_model()
            model.load_state_dict(test_model['flow_state_dict'])
            model = model.to(device)

            extractor = load_extractor(tdim=1024, fdim=64, target_size=8)
            extractor1 = load_extractor(tdim=1024, fdim=128, target_size=16)
            extractor2 = load_extractor(tdim=1024, fdim=256, target_size=32)
            extractor.load_state_dict(test_model['ast0_state_dict'])
            extractor1.load_state_dict(test_model['ast1_state_dict'])
            extractor2.load_state_dict(test_model['ast2_state_dict'])
            extractor, extractor1, extractor2 = extractor.to(device=device), extractor1.to(device=device), extractor2.to(device=device)
            extractor.eval()
            extractor1.eval()
            extractor2.eval()
            for param in extractor.parameters():
                param.requires_grad = False
            for param in extractor1.parameters():
                param.requires_grad = False
            for param in extractor2.parameters():
                param.requires_grad = False

        # if mode:
            # results by type
            csv_lines.append([machine_type])
            csv_lines.append(["id", "AUC", "pAUC"])
            performance = []

            target_dir = select_dirs(machine_type, mode=False, dir_type="test")

            anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{id_str}.csv".format(
                                                                                    result=c.result_directory,
                                                                                    machine_type=machine_type,
                                                                                    id_str=_id)
            anomaly_score_list = []
            # test_labels = []

            print("\n============== BEGIN TEST FOR A MACHINE ID ==============")
            data_list, test_labels = test_file_list_generator(target_dir=root_path, id_name=_id, dir_name="test", mode=True)
            test_dataset = AudioDataset(data=data_list, _id=_id, root=root_path, frame_length=c.frame_length, shift_length=c.shift_length, audio_conf=c.audio_conf)
            test_dl = DataLoader(dataset=test_dataset, batch_size=c.batch_size, shuffle=False)
            # print(len(y_true), print(len(test_dl)))
            model.eval()
            with torch.no_grad():
                for batch in tqdm(test_dl):
                    # print("batch", len(batch))
                    # sys.exit(1)
                    # batch = batch_dict['audio']
                    # labels  = batch_dict['label']

                    batch0 = batch[0].to(device)
                    batch1 = batch[1].to(device)
                    batch2 = batch[2].to(device)

                    f0 = extractor(batch0).to(device)
                    f1 = extractor1(batch1).to(device)
                    f2 = extractor2(batch2).to(device)
                    features = [f2, f1, f0]
                    z = model(features)

                    # get the L2 norm of Z * 0.5
                    z_concat = t2np(concat_maps(z))

                    z0 = z[0]
                    z1 = z[1]
                    z2 = z[2]
                    z0 = t2np(flat(z0))
                    z1 = t2np(flat(z1))
                    z2 = t2np(flat(z2))
                    # flat_maps = [z0, z1, z2]
                    # zz = t2np(torch.cat(flat_maps, dim=1)[..., None])
                    # nll_score = np.mean(zz ** 2 / 2, axis=(1, 2))
                    z0_score = np.mean(z0 ** 2 / 2, axis=(1))
                    z1_score = np.mean(z1 ** 2 / 2, axis=(1))
                    z2_score = np.mean(z2 ** 2 / 2, axis=(1))
                    # print("z0_score", z0_score.shape, z1_score.shape)
                    # nll_score = (z0_score + z1_score + z2_score) / 3
                    nll_score = np.mean(z_concat ** 2 / 2, axis=(1, 2))
                    # nll_score = z2_score
                    # print(z0_score, z1_score, z2_score, nll_score)
                    # sys.exit(1)
                    anomaly_score_list.append(nll_score)
                    # test_labels.append(t2np(labels))

                    # print("score", nll_score)

                    del batch, batch0, batch1, batch2, f0, f1, f2, z
                    gc.collect()
                    torch.cuda.empty_cache()
                    # sys.exit(1)
            anomaly_score_list = np.concatenate(anomaly_score_list)
            # test_labels = np.concatenate(y_true)

            # AUC
            auc = metrics.roc_auc_score(test_labels, anomaly_score_list)
            p_auc = metrics.roc_auc_score(test_labels, anomaly_score_list, max_fpr=c.max_fpr)
            print("AUC : {}".format(auc))
            print("pAUC : {}".format(p_auc))
            csv_lines.append([_id.split("_", 1)[1], auc, p_auc])
            performance.append([auc, p_auc])

            np.savetxt(anomaly_score_csv, anomaly_score_list.reshape(-1, anomaly_score_list.shape[-1]), delimiter=',')
            save_csv(save_file_path=AUC_csv, save_data=csv_lines)

            del test_dataset, test_dl, model
            gc.collect()
            torch.cuda.empty_cache()
            
            compare_histogram(anomaly_score_list, test_labels, machine_type=machine_type, _id=_id)
            sys.exit(1)