import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn
import config as c
from freia_funcs import *

WEIGHT_DIR = './weights'
MODEL_DIR = './models/tmp'


def get_cs_flow_model(input_dim=c.n_feat):
    nodes = list()
    nodes.append(InputNode(input_dim, c.map_size[0], c.map_size[1], name='input'))
    nodes.append(InputNode(input_dim, c.map_size[0] // 2, c.map_size[1] // 2, name='input2'))
    nodes.append(InputNode(input_dim, c.map_size[0] // 4, c.map_size[1] // 4, name='input3'))

    for k in range(c.n_coupling_blocks):
        if k == 0:
            node_to_permute = [nodes[-3].out0, nodes[-2].out0, nodes[-1].out0]
        else:
            node_to_permute = [nodes[-1].out0, nodes[-1].out1, nodes[-1].out2]

        nodes.append(Node(node_to_permute, ParallelPermute, {'seed': k}, name=F'permute_{k}'))
        nodes.append(Node([nodes[-1].out0, nodes[-1].out1, nodes[-1].out2], parallel_glow_coupling_layer,
                          {'clamp': c.clamp, 'F_class': CrossConvolutions,
                           'F_args': {'channels_hidden': c.fc_internal,
                                      'kernel_size': c.kernel_sizes[k], 'block_no': k}},
                          name=F'fc1_{k}'))
    nodes.append(OutputNode([nodes[-1].out0], name='output_end0'))
    nodes.append(OutputNode([nodes[-2].out1], name='output_end1'))
    nodes.append(OutputNode([nodes[-3].out2], name='output_end2'))
    nf = ReversibleGraphNet(nodes, n_jac=3)
    return nf

def nf_forward(model, inputs):
    return model(inputs), model.jacobian(run_forward=False)

def save_model(model, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(model, os.path.join(MODEL_DIR, filename))


def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    model = torch.load(path)
    return model
