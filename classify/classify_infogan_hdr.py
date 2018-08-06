# -*- coding: utf-8 -*-

# n_classes4-n_discrete1-n_conti3-n_noise10_3/model/config.p
# n_classes2-n_discrete1-n_conti2-n_noise10/model/config.p
# n_classes1-n_discrete0-n_conti3-n_noise10/model/config.p
# n_classes2-n_discrete1-n_conti3-n_noise0/model/config.p


import argparse
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import sys

sys.path.append("/home/patrick/repositories/hyperspectral_phenotyping_gan")
from data_loader_hdr_pytorch import Hdr_dataset, save_model, save_image
from models.networks import FrontEnd, D, Q, weights_init
from models.networks import G_with_fc as G
from config.config_hdr import config_dict as config
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--nc', default=3, required=False, help='dim of category code or number of classes')
parser.add_argument('--n_conti', default=2, required=False, help='')
parser.add_argument('--n_dis', default=1, required=False, help='')
parser.add_argument('--n_noise', default=10, required=False, help='')
parser.add_argument('--dataset', default="", required=True, help='set to hdr to load files from bonn experiment')

parser.add_argument('--outf_suffix', default="", required=False, help='')

parser.add_argument('--epoch', required=True, help='epoch...')
parser.add_argument('--semisup', action="store_true", help='True: semi-supervised | False: unsupervised')
parser.add_argument('--sup_ratio', default=1.0, required=False, help='ratio of semi-supervised labels')
parser.set_defaults(semisup=False)
opt = parser.parse_args()

sup_ratio = float(opt.sup_ratio)  # 0.1


def save_config(path_config, data):
    pickle.dump(data, open(path_config, "wb"))
    print("Saved config to:", path_config)


def load_config(path_config):
    print("Using pretrained model")
    print("Loading config from:", path_config)
    return pickle.load(open(path_config, "rb"))


class log_gaussian:
    def __call__(self, x, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - \
                (x - mu).pow(2).div(var.mul(2.0) + 1e-6)

        return logli.sum(1).mean().mul(-1)


class Classifier:
    def __init__(self, FE, Q, config):

        self.FE = FE
        self.Q = Q

        self.dim_noise = config["NOISE"]
        self.dim_code_conti = config["N_CONTI"]
        self.dim_code_disc = config["NC"] * config["N_DISCRETE"]
        self.num_categories = config["NC"]
        self.size_total = self.dim_noise + self.dim_code_conti + self.dim_code_disc
        self.dim_signature = config["NDF"]

        self.batch_size = 128  # 40 * self.num_categories
        self.dataset = Hdr_dataset(load_to_mem=False, train=False)
        self.num_batches = len(self.dataset) // self.batch_size
        print("Num samples:", len(self.dataset), ", Num batches:", self.num_batches)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                     shuffle=False, num_workers=8, drop_last=False)

    def classify(self):

        real_x = torch.FloatTensor(self.batch_size, self.dim_signature).cuda()
        real_x = Variable(real_x)

        classification = np.empty([0])
        for i_batch, x in tqdm(enumerate(self.dataloader)):
            real_x.data.resize_(x.size())
            real_x.data.copy_(x)
            fe_out = self.FE(real_x)

            q_logits, _, _ = self.Q(fe_out)
            q_classification = np.argmax(q_logits.data.cpu().numpy(), axis=1)

            classification = np.hstack((classification, q_classification))

        self.dataset.classification(classification)
        print(classification.shape)

    print("Finished classifying with GAN")


if __name__ == '__main__':

    out_path = "generated_leaf_infogan-n_classes{}-n_discrete{}-n_conti{}-n_noise{}{}".format(opt.nc,
                                                                                              opt.n_dis,
                                                                                              opt.n_conti,
                                                                                              opt.n_noise,
                                                                                              opt.outf_suffix)

    if opt.semisup:
        out_path += "_supervised"

    config_path = "/home/patrick/repositories/hyperspectral_phenotyping_gan/trained_models{}/{}".format(opt.dataset,
                                                                                                        out_path)
    # config_path += "/model{}".format("_ratio-{}".format(int(sup_ratio * 100)) if opt.semisup else "")
    config_path += "/model{}".format("")
    config = load_config(os.path.join(config_path, "config.p"))

    size_total = config["NOISE"] + config["N_CONTI"] + (config["NC"] * config["N_DISCRETE"])

    fe = FrontEnd()
    q = Q(dim_conti=config["N_CONTI"], dim_disc=config["NC"] * config["N_DISCRETE"])

    NETFE = "/home/patrick/repositories/hyperspectral_phenotyping_gan/trained_models{}/{}/model{}/netFE_epoch_{}{}.pth".format(
        opt.dataset, out_path, "{}", "{}", "-crossval-0")
    NETFE = NETFE.format("", opt.epoch)
    NETQ = "/home/patrick/repositories/hyperspectral_phenotyping_gan/trained_models{}/{}/model{}/netQ_epoch_{}{}.pth".format(
        opt.dataset, out_path, "{}", "{}", "-crossval-0")
    NETQ = NETQ.format("", opt.epoch)

    fe.load_state_dict(torch.load(NETFE))
    fe.eval()

    q.load_state_dict(torch.load(NETQ))
    q.eval()

    size_total = config["NOISE"] + config["N_CONTI"] + (config["NC"] * config["N_DISCRETE"])
    fe = FrontEnd()
    q = Q(dim_conti=config["N_CONTI"], dim_disc=config["NC"] * config["N_DISCRETE"])

    for i in [fe, q]:
        i.cuda()
        i.apply(weights_init)

    classifier = Classifier(fe, q, config)
    classifier.classify()
