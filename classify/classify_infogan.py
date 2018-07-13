# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import sys

sys.path.append('/home/patrick/repositories/hyperspec')
from used_methods.hsgan.data_loader import DataLoader
from used_methods.hsgan.discriminator import InfoGAN_Discriminator as Discriminator, d_load_from_state_dict

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='leaf | in')
parser.add_argument('--semisup', action="store_true", help='True: semi-supervised | False: unsupervised')
parser.add_argument('--sup_ratio', default=1.0, required=False, help='ratio of semi-supervised labels')
parser.set_defaults(semisup=False)
opt = parser.parse_args()

if opt.dataset == "leaf":
    from used_methods.hsgan.config.multilabel.config_leaf_infogan_multilabel import *
else:
    from used_methods.hsgan.config.multilabel.config_leaf_infogan_multilabel import *  # TODO
    # from config_in_infogan import *

try:
    os.makedirs(OUTF)
except OSError:
    pass
try:
    os.makedirs(OUTF + "/classification")
except OSError:
    pass

sup_ratio = float(opt.sup_ratio)
NETD_classify = NETD_classify.format("_ratio-{}".format(int(sup_ratio * 100)) if opt.semisup else "")
print(NETD_classify)


def classify_InfoGAN(InfoGAN_Dis, dataloader, use_gpu=False):
    """train InfoGAN and print out the losses for D and G"""

    # get next batch
    data = dataloader.data_original  # get the inputs from true distribution
    inputs_x, inputs_y, _, _ = zip(*data)
    inputs_x, inputs_y = np.array(inputs_x), np.array(inputs_y)

    batch_size = np.minimum(300, len(inputs_x))
    outputs_total = np.empty([0, 4], dtype=float)
    for idx in range(len(inputs_x) // batch_size + 1):

        inputs_x_batch = inputs_x[idx * batch_size:(idx + 1) * batch_size]
        inputs_x_batch = Variable(torch.FloatTensor(inputs_x_batch))
        if use_gpu:
            inputs_x_batch = inputs_x_batch.cuda()

        # Discriminator
        outputs = InfoGAN_Dis(inputs_x_batch)
        outputs = outputs.cpu().detach().numpy()
        outputs_total = np.vstack((outputs_total, outputs))

    outputs_total = np.array(outputs_total)
    labels = np.argmax(outputs_total, axis=1)
    labels[labels == 3] = 0
    # labels[labels == 0] = 4
    # labels[labels == 2] = 0
    plt.figure(figsize=(15, 15))
    plt.subplot(1, 1, 1)
    plt.axis('off')
    plt.imshow(labels.reshape(123, 147), cmap="tab10")
    # plt.subplot(1, 2, 2)
    # plt.imshow(inputs_y.reshape(123, 147), cmap="tab10")
    plt.show()
    print("-" * 42)
    # print(inputs_y)


def run_InfoGAN(n_conti=2, n_discrete=1, num_category=3,
                batch_size=1, use_gpu=False):
    # loading data
    from_dataset = False if FROM_DATASET is None else FROM_DATASET
    dataloader = DataLoader(batch_size=batch_size, dataset=opt.dataset, from_dataset=from_dataset)

    D_featmap_dim = 1024
    # initialize models
    InfoGAN_Dis = Discriminator(n_conti, n_discrete, num_category,
                                D_featmap_dim,
                                NDF, NGF, classification=True)
    d_load_from_state_dict(InfoGAN_Dis, torch.load(NETD_classify))

    if use_gpu:
        InfoGAN_Dis = InfoGAN_Dis.cuda()

    classify_InfoGAN(InfoGAN_Dis, dataloader, use_gpu)


if __name__ == '__main__':
    # run_InfoGAN(n_conti=2, n_discrete=1, D_featmap_dim=64, G_featmap_dim=128, use_gpu=True,
    #            n_epoch=10000, batch_size=32, update_max=2000)
    run_InfoGAN(num_category=NC, n_conti=N_CONTI, n_discrete=N_DISCRETE,
                use_gpu=True, batch_size=100)
