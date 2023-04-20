"""
 @file   00_train.py
 @brief  Script for training
 @author Toshiki Nakamura, Yuki Nikaido, and Yohei Kawaguchi (Hitachi Ltd.)
 Copyright (C) 2020 Hitachi, Ltd. All right reserved.
"""

########################################################################
# import default python-library
########################################################################
import os
import glob
import sys
import time
import gc
########################################################################


########################################################################
# import additional python-library
########################################################################
import numpy as np
from tqdm import tqdm
import config as c
import random
from PANN_model import load_extractor
from model import get_cs_flow_model, save_model, nf_forward
from utils import *
from dataset import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.cuda.memory
import pandas as pd

if torch.cuda.is_available():
    # set CUDA allocator
    torch.cuda.memory.set_per_process_memory_fraction(0.9)
    # set cudnn flags
    torch.backends.cudnn.benchmark = True 
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
########################################################################

########################################################################
# visualizer
########################################################################
class visualizer(object):
    def __init__(self):
        import matplotlib.pyplot as plt
        self.plt = plt
        self.fig = self.plt.figure(figsize=(30, 10))
        self.plt.subplots_adjust(wspace=0.3, hspace=0.3)

    def loss_plot(self, loss, val_loss):
        """
        Plot loss curve.

        loss : list [ float ]
            training loss time series.
        val_loss : list [ float ]
            validation loss time series.

        return   : None
        """
        ax = self.fig.add_subplot(1, 1, 1)
        ax.cla()
        ax.plot(loss)
        ax.plot(val_loss)
        ax.set_title("Model loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(["Train", "Validation"], loc="upper right")

    def save_figure(self, name):
        """
        Save figure.

        name : str
            save png file path.

        return : None
        """
        self.plt.savefig(name)

########################################################################
# main 00_train.py
########################################################################
if __name__ == "__main__":
    
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = c.mode
    if mode is None:
        sys.exit(-1)

    # param = com.load_yaml()
        
    # make output directory
    os.makedirs(c.model_directory, exist_ok=True)

    # initialize the visualizer
    visualizer = visualizer()

    # training device
    # device = torch.device('cuda:0')  # default device_ids[0]
   

    # training info 
    device = c.device
    epochs = int(c.meta_epochs)
    batch_size = int(c.batch_size)
    latent_size = int(c.latent_size)
    # num_layers = int(param["NF_Layers"])
    
    # load base_directory list
    machine_list = c.machine_type
    print("=====================================")
    print("Train Machine List: ", machine_list)
    print("=====================================")

    # load pre-trained feature extractor
    extractors = feature_extractor = load_extractor(sample_rate=c.sr_list,
                                                    window_size=c.n_fft,
                                                    hop_size=c.hop_length,
                                                    mel_bins=c.n_mels,
                                                    fmin=c.fmin,
                                                    fmax=c.fmax)

    #extractor = nn.DataParallel(extractor, device_ids=[0, 1])
    # extractor, extractor1, extractor2 = extractor.to(device=device), extractor1.to(device=device), extractor2.to(device=device)
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
    for idx, machine in enumerate(machine_list):
        print("\n===========================")
        print("[{idx}/{total}] {machine}".format(machine=machine, idx=idx+1, total=len(machine_list)))
        
        root_path = os.path.join(c.dev_directory, machine)
        # param["dev_directory"] + "/" + machine

        # data_list = select_dirs(param=param, machine=machine)
        data_list = select_dirs(machine, mode=True, dir_type="train")

        id_list = get_machine_id_list(target_dir=root_path, dir_type="train")

        print("Current Machine: ", machine)
        print("Machine ID List: ", id_list)

        train_list = []
        val_list = []

        # FIXME: modify the dataset!!!
        for path in data_list[:200]:
        # for path in data_list:
            if random.random() < 0.85:
                train_list.append(path)
            else:
                val_list.append(path)
        
        for _id in id_list:
            # generate dataset

            model_file_path = "{model}/model_{machine}_{_id}.pt".format(model=c.model_directory, machine=machine, _id=_id)

            if os.path.exists(model_file_path):
                logger.info("model exists")
                continue
            print("\n----------------")
            print("Generating Dataset of Current ID: ", _id)

            train_dataset = AudioDataset(data=train_list, _id=_id, root=root_path, sample_rate=c.sample_rate)
            val_dataset = AudioDataset(data=val_list, _id=_id, root=root_path, sample_rate=c.sample_rate)
            train_dl = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            val_dl = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

            print("------ DONE -------")

            history_img = "{model}/history_{machine}_{_id}.png".format(model=c.model_directory, machine=machine, _id=_id)
            
            # train model
            print("\n----------------")
            print("Start Model Training...")

            train_loss_list = []
            val_loss_list = []
            flow_model = get_cs_flow_model()
            flow_model = flow_model.to(device)


            # optimizer = torch.optim.Adam(flow_model.parameters(), lr=2e-4, eps=1e-04, weight_decay=1e-5)
            optimizer = torch.optim.Adam(flow_model.parameters(), lr=2e-4, eps=1e-04, weight_decay=1e-5)

            for epoch in range(1, epochs+1):
                train_loss = 0.0
                val_loss = 0.0
                print("Epoch: {}".format(epoch))   

                flow_model.train()
                ccc = 0

                # Training part
                for batch in tqdm(train_dl):
                    # optimizer.zero_grad()
                    batch0 = batch[0].to(device)
                    batch1 = batch[1].to(device)
                    batch2 = batch[2].to(device)

                    # torch.cuda.empty_cache()
                    f0 = extractor(batch0).to(device)
                    f1 = extractor1(batch1).to(device)
                    f2 = extractor2(batch2).to(device)
                    # print("feature shape", f0.shape, f1.shape, f2.shape)
                    # exit(1)
                    # features = [f2.detach(), f1.detach(), f0.detach()]
                    features = [f2, f1, f0]
                    # features = [f2.requires_grad_(True), f1.requires_grad_(True), f0.requires_grad_(True)]
                    # feature0: torch.Size([4, 512, 32, 32])
                    # feature1: torch.Size([4, 512, 16, 16])
                    # feature2: torch.Size([4, 512, 8, 8])
                    z, jac = nf_forward(flow_model, features)
                    print("z, jac", len(z), len(jac))
                    loss = get_loss(z, jac)
                    print("loss", loss)
                    # sys.exit(-1)

                    # loss.detach_()
                    # loss.detach()
                    
                    del batch0, batch1, batch2, features, z, jac, batch, f0, f1, f2
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    ccc+=1
                    if ccc >= 10:
                        sys.exit(-1)

                    optimizer.zero_grad()
                    with torch.autograd.set_detect_anomaly(True):
                        loss.backward()

                        train_loss += loss.item()
                        loss = None
                        del loss
                        gc.collect()
                        torch.cuda.empty_cache()

                    optimizer.step()

                    gc.collect()
                    torch.cuda.empty_cache()
                    # exit(1)
                    # for param in flow_model.parameters():
                    #     b = param.data
                    #     break

                # exit(1)
                train_loss /= len(train_dl)
                train_loss_list.append(train_loss)

                # Validation part
                flow_model.eval()
                with torch.no_grad():
                    for batch in tqdm(val_dl):

                        batch0 = batch[0].to(device)
                        batch1 = batch[1].to(device)
                        batch2 = batch[2].to(device)

                        f0 = extractor(batch0)
                        f1 = extractor1(batch1)
                        f2 = extractor2(batch2)

                        f0 = f0.to(device)
                        f1 = f1.to(device)
                        f2 = f2.to(device)
                        features = [f2, f1, f0]

                        z, jac = nf_forward(flow_model, features)

                        loss = get_loss(z, jac)
                        val_loss += loss.item()

                        del batch0, batch1, batch2, features, z, jac, batch, f0, f1, f2, loss
                        gc.collect()
                        torch.cuda.empty_cache()

                val_loss /= len(val_dl)
                val_loss_list.append(val_loss)
                
                print("Train Loss: {train_loss}, Validation Loss: {val_loss}".format(train_loss=train_loss, val_loss=val_loss))
                # for param in flow_model.parameters():
                #     print(param.grad, end=" ")
            visualizer.loss_plot(train_loss_list, val_loss_list)
            visualizer.save_figure(history_img)

            torch.save(flow_model.state_dict(), model_file_path)
            print("save_model -> {}".format(model_file_path))
            # com.logger.info("save_model -> {}".format(model_file_path))

            del train_dataset, val_dataset, train_dl, val_dl, flow_model
            gc.collect()
            torch.cuda.empty_cache()
            exit(1)
            time.sleep(5)