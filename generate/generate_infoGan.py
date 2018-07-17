# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import sys

sys.path.append("/home/patrick/repositories/hyperspectral_phenotyping_gan")
from data_loader import DataLoader
from models.networks import G_with_fc as G
import pickle
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--nc', default=3, required=False, help='dim of category code or number of classes')
parser.add_argument('--n_conti', default=2, required=False, help='')
parser.add_argument('--n_dis', default=1, required=False, help='')
parser.add_argument('--n_noise', default=10, required=False, help='')

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


def plot_fake(fakes, num_category, label, mean, x_real):
    # PLOT FAKE DATA
    current_plot = label * 2 + 1
    ax = plt.subplot(num_category, 2, current_plot)
    ax.set_title("Label {} Fake".format(label))
    ax.grid(True)
    linestyle = "r-"
    if mean is not None and num_category <= 3:
        _, = plt.plot(mean, linestyle, linewidth=8)
    plt.ylim(0, 1.0)
    # plt.yscale('symlog')
    plt.yticks(np.arange(0, 1.0, .1))

    for idx, x in enumerate(fakes):
        plt.subplot(num_category, 2, current_plot)
        x = x.flatten()
        linestyle = "--"
        _, = plt.plot(x, linestyle, linewidth=2, alpha=0.3)
        # plt.legend([l1], ["fake sample" + str(label[idx])])

    # PLOT REAL DATA
    if x_real is not None:
        current_plot += 1
        ax = plt.subplot(num_category, 2, current_plot)
        ax.set_title("Label {} Real".format(label))
        ax.grid(True)
        linestyle = "r-"
        _, = plt.plot(mean, linestyle, linewidth=8)
        plt.ylim(0, 1.0)
        # plt.yscale('symlog')
        plt.yticks(np.arange(0, 1.0, .1))

        for idx, x in enumerate(x_real):
            plt.subplot(num_category, 2, current_plot)
            x = x.flatten()
            linestyle = "--"
            _, = plt.plot(x, linestyle, linewidth=2, alpha=0.3)
            # plt.legend([l1], ["fake sample" + str(label[idx])])


class log_gaussian:
    def __call__(self, x, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - \
                (x - mu).pow(2).div(var.mul(2.0) + 1e-6)

        return logli.sum(1).mean().mul(-1)


class Generator:
    def __init__(self, G, config):

        self.G = G

        self.dim_noise = config["NOISE"]
        self.dim_code_conti = config["N_CONTI"]
        self.num_categories = config["NC"]
        self.size_total = self.dim_noise + self.dim_code_conti + self.num_categories

        self.batch_size = 30 * self.num_categories
        self.dataloader = DataLoader(batch_size=self.batch_size, sup_ratio=sup_ratio)

    def _noise_sample(self, dis_c, con_c, noise, bs):

        idx = np.random.randint(self.num_categories, size=bs)
        c = np.zeros((bs, self.num_categories))
        c[range(bs), idx] = 1.0

        dis_c.data.copy_(torch.Tensor(c))
        con_c.data.uniform_(-1.0, 1.0)
        noise.data.uniform_(-1.0, 1.0)

        z = torch.cat([noise, dis_c, con_c], 1).view(-1, self.size_total)

        return z, idx

    def generate(self):

        batch_size_eval = self.batch_size

        batch_x_mean, _ = self.dataloader.fetch_samples_mean()
        batch_x_mean = batch_x_mean.numpy()
        batch_x_real, batch_y_real = self.dataloader.fetch_samples(
            num_sample_each_class=batch_size_eval * 10 // self.num_categories)
        batch_x_real, batch_y_real = batch_x_real.numpy(), batch_y_real.numpy()

        label = torch.FloatTensor(self.batch_size).cuda()
        dis_c = torch.FloatTensor(self.batch_size, self.num_categories).cuda()
        con_c = torch.FloatTensor(self.batch_size, self.dim_code_conti).cuda()
        noise = torch.FloatTensor(self.batch_size, self.dim_noise).cuda()

        label = Variable(label, requires_grad=False)
        dis_c = Variable(dis_c)
        con_c = Variable(con_c)
        noise = Variable(noise)

        # fixed random variables
        c = np.linspace(-1, 1, batch_size_eval // self.num_categories).reshape(1, -1)
        c = np.repeat(c, self.num_categories, 0).reshape(-1, 1)

        c1 = np.hstack([c, np.zeros_like(c)])
        c2 = np.hstack([np.zeros_like(c), c])
        c12 = np.hstack([c, c])
        c_all = np.hstack([c, c])

        if self.dim_code_conti > 3:
            raise ValueError("Continuous code of dim > 3 not implemented")
        if self.dim_code_conti == 3:
            c1 = np.hstack([c, np.zeros_like(c), np.zeros_like(c)])
            c2 = np.hstack([np.zeros_like(c), c, np.zeros_like(c)])
            c12 = np.hstack([c, -c, np.zeros_like(c)])
            c3 = np.hstack([np.zeros_like(c), np.zeros_like(c), c])
            c13 = np.hstack([c, np.zeros_like(c), c])
            c23 = np.hstack([np.zeros_like(c), c, c])
            c_all = np.hstack([c, c, c])
        cx = c12

        idx = np.arange(self.num_categories).repeat(batch_size_eval // self.num_categories)
        one_hot = np.zeros((batch_size_eval, self.num_categories))
        one_hot[range(batch_size_eval), idx] = 1
        # fix_noise = torch.Tensor(batch_size_eval, self.dim_noise).uniform_(-1, 1)
        fix_noise = torch.Tensor(1, self.dim_noise).uniform_(-1, 1).repeat(batch_size_eval, 1)
        # print(fix_noise)
        # print(fix_noise.shape)
        # 1 / 0
        label.data.resize_(self.batch_size, 1)
        dis_c.data.resize_(self.batch_size, self.num_categories)
        con_c.data.resize_(self.batch_size, self.dim_code_conti)
        noise.data.resize_(self.batch_size, self.dim_noise)

        noise.data.copy_(fix_noise)
        dis_c.data.copy_(torch.Tensor(one_hot))

        # con_c.data.copy_(torch.from_numpy(cx))
        con_c.data.uniform_(-1.0, 1.0)
        z = torch.cat([noise, dis_c, con_c], 1).view(-1, self.size_total)
        x_save = self.G(z)
        x_save = x_save.data.cpu().numpy()

        num_samples = np.arange(0, batch_size_eval + 1, batch_size_eval // self.num_categories)
        for label in range(self.num_categories):
            mean_of_category = None if label > 2 else batch_x_mean[label]
            x_real_of_category = None if label > 2 else batch_x_real[batch_y_real == label]

            plot_fake(x_save[num_samples[label]:num_samples[label + 1]],
                      self.num_categories, label,
                      mean_of_category, x_real_of_category)

        plt.show()


if __name__ == '__main__':

    out_path = "generated_leaf_infogan-n_classes{}-n_discrete{}-n_conti{}-n_noise{}{}".format(opt.nc,
                                                                                                   opt.n_dis,
                                                                                                   opt.n_conti,
                                                                                                   opt.n_noise,
                                                                                                   opt.outf_suffix)

    config_path = "/home/patrick/repositories/hyperspectral_phenotyping_gan/trained_models/{}".format(out_path)
    config_path += "/model{}".format("_ratio-{}".format(int(sup_ratio * 100)) if opt.semisup else "")
    config = load_config(os.path.join(config_path, "config.p"))

    size_total = config["NOISE"] + config["N_CONTI"] + config["NC"]
    g = G(size_total)

    NETG_generate = "/home/patrick/repositories/hyperspectral_phenotyping_gan/trained_models/{}/model{}/netG_epoch_{}{}.pth".format(
        out_path, "{}", "{}", "-crossval-0")
    NETG_generate = NETG_generate.format("_ratio-{}".format(int(sup_ratio * 100)) if opt.semisup else "", opt.epoch)

    g.load_state_dict(torch.load(NETG_generate))
    g.eval()

    for i in [g]:
        i.cuda()

    generator = Generator(g, config)
    generator.generate()
