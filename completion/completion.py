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
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--nc', default=3, required=False, help='dim of category code or number of classes')
parser.add_argument('--n_conti', default=2, required=False, help='')
parser.add_argument('--n_dis', default=1, required=False, help='')
parser.add_argument('--n_noise', default=10, required=False, help='')

parser.add_argument('--outf_suffix', default="", required=False, help='')

parser.add_argument('--epoch', required=True, help='epoch...')
parser.add_argument('--semisup', action="store_true", help='True: semi-supervised | False: unsupervised')
parser.add_argument('--sup_ratio', default=1.0, required=False, help='ratio of semi-supervised labels')
parser.add_argument('--interpolate', action="store_true", help='')
parser.set_defaults(semisup=False)
parser.set_defaults(interpolate=False)
opt = parser.parse_args()

sup_ratio = float(opt.sup_ratio)  # 0.1


def load_config(path_config):
    print("Using pretrained model")
    print("Loading config from:", path_config)
    return pickle.load(open(path_config, "rb"))


def plot_signatures(fakes, num_category, num_col, label, x_real, legend=None, alpha=0.3, linewidth_fake=2):
    # PLOT FAKE DATA
    current_plot = label * num_col + 1
    ax = plt.subplot(num_category, num_col, current_plot)
    ax.grid(True)
    plt.ylim(0, 1.0)
    plt.yticks(np.arange(0, 1.0, .1))

    for idx, x in enumerate(fakes):
        plt.subplot(num_category, num_col, current_plot)
        x = x.flatten()
        linestyle = "--"
        _, = plt.plot(x, linestyle, linewidth=linewidth_fake, alpha=alpha)
        if legend is not None:
            legend = legend.squeeze()
            plt.legend(legend, fontsize=16)

    if x_real is not None:
        # PLOT REAL DATA
        current_plot += 1
        ax = plt.subplot(num_category, num_col, current_plot)
        ax.grid(True)
        plt.ylim(0, 1.0)
        # plt.yscale('symlog')
        plt.yticks(np.arange(0, 1.0, .1))

        plt.subplot(num_category, num_col, current_plot)
        x = x_real.flatten()
        linestyle = "--"
        _, = plt.plot(x, linestyle, linewidth=2, alpha=0.3)
        # plt.legend([l1], ["fake sample" + str(label[idx])])


class Generator:
    def __init__(self, G, config):
        self.G = G

        self.dim_noise = config["NOISE"]
        self.dim_code_conti = config["N_CONTI"]
        self.dim_code_disc = (config["NC"] * config["N_DISCRETE"])
        self.num_categories = config["NC"]
        self.size_total = self.dim_noise + self.dim_code_conti + (self.num_categories * config["N_DISCRETE"])

        self.batch_size = 1 * 3

        self.dataloader = DataLoader(batch_size=self.batch_size, sup_ratio=sup_ratio)

    def _set_noise(self, noise, dis_c, con_c):
        z_ = []  # [noise, dis_c, con_c]
        if self.dim_noise != 0:
            z_.append(noise)
        if self.dim_code_disc != 0:
            z_.append(dis_c)
        if self.dim_code_conti != 0:
            z_.append(con_c)
        z = torch.cat(z_, 1).view(-1, self.size_total)

        return z

    def _inpaint(self, x_inputs, code=None, noise=None, batch_size=None):
        """
        z = torch.zeros([self.batch_size, self.dim_noise]).cuda()
        z = Variable(z, requires_grad=True)
        z.data.resize_(self.batch_size, self.size_total)
        """

        if code is None:
            code = torch.zeros([self.batch_size, self.dim_code_conti + self.dim_code_disc]).cuda()
        if noise is None:
            noise = torch.cuda.FloatTensor(1, self.dim_noise).uniform_(-1, 1).repeat(self.batch_size, 1)
        if batch_size is None:
            batch_size = self.batch_size

        code = Variable(code, requires_grad=True)
        noise = Variable(noise, requires_grad=False)

        # z = torch.cat([noise, code], 1)
        # z = Variable(z, requires_grad=False)

        data_shape = [batch_size, 160]
        mask = np.ones(data_shape)
        mask = torch.FloatTensor(mask)
        mask = mask.cuda()
        # l = 45
        # u = 120
        # mask[:, l:u] = 0.0

        optimizer = optim.Adam([code], lr=0.01, betas=(0.5, 0.999))  # [code, noise]

        n_epoch = 100
        fakes = None
        z = None
        for i in range(n_epoch):
            z = self._set_noise(noise, None, code)
            fakes = self.G(z)

            # make a minibatch of labels
            """labels = np.ones(batch_size)
            labels = torch.from_numpy(labels.astype(np.float32))
            if use_gpu:
                labels = labels.cuda()
            labels = Variable(labels)

            # Discriminator
            #outputs = InfoGAN_Dis(fakes)
            #contextual_loss = torch.mean(torch.abs((torch.mul(mask, fakes) - torch.mul(mask, x_inputs))))

            # Update Generator
            G_loss = G_criterion(outputs[:, 0],
                                 labels[:])
            perceptual_loss = G_loss"""

            # contextual_loss = torch.mean(torch.abs((torch.mul(mask, fakes) - torch.mul(mask, x_inputs))))
            contextual_loss = torch.mean(torch.abs((torch.mul(mask, fakes) - torch.mul(mask, x_inputs))))
            # complete_loss = contextual_loss + 0.01 * perceptual_loss
            complete_loss = contextual_loss

            optimizer.zero_grad()
            complete_loss.backward()
            optimizer.step()

        return fakes, z

    def generate_code(self):
        #x_inputs, y_inputs = self.dataloader.fetch_samples(num_sample_each_class=1)
        data_x, data_y = zip(*self.dataloader.data_total)
        data_x, data_y = np.array(data_x), np.array(data_y)

        batch_size = 1
        code = torch.zeros([batch_size, self.dim_code_conti + self.dim_code_disc]).cuda()
        noise = torch.cuda.FloatTensor(1, self.dim_noise).uniform_(-1, 1).repeat(batch_size, 1)

        data = dict()
        data["x"] = list()
        data["y"] = list()
        data["fake"] = list()
        data["z"] = list()
        current_index = 0
        while current_index < len(data_x):
            batch_x = data_x[current_index:current_index + batch_size]
            batch_y = data_y[current_index:current_index + batch_size]
            current_index += batch_size

            fakes, z = self._inpaint(torch.cuda.FloatTensor(batch_x), code=code, noise=noise, batch_size=batch_size)
            fakes = fakes.cpu().detach().numpy()
            z = z.cpu().detach().numpy()
            #plot_signatures(fakes[:1], 3, 2,
            #                label=0, x_real=batch_x[0])
            #plt.show()
            data["x"] += batch_x.tolist()
            data["y"] += batch_y.tolist()
            data["fake"] += fakes.tolist()
            data["z"] += z.tolist()

            if current_index % 100 == 0:
                print(current_index, "/", len(data_y))

        pickle.dump(data, open("./experiments/generated_code_noise{}_disc{}_conti{}.p".format(self.dim_noise,
                                                                                              self.dim_code_disc,
                                                                                              self.dim_code_conti),
                               "wb"))

    def inpaint(self):
        x_inputs, y_inputs = self.dataloader.fetch_samples(num_sample_each_class=1)
        x_inputs = x_inputs.cuda()
        # l = 45
        # u = 120
        # mask[:, l:u] = 0.0

        fakes, z = self._inpaint(x_inputs)

        plot_signatures(fakes.cpu().detach().numpy()[:1], 3, 2,
                        label=0, x_real=x_inputs.cpu().detach().numpy()[0])
        plot_signatures(fakes.cpu().detach().numpy()[:2], 3, 2,
                        label=1, x_real=x_inputs.cpu().detach().numpy()[1])
        plot_signatures(fakes.cpu().detach().numpy()[:3], 3, 2,
                        label=2, x_real=x_inputs.cpu().detach().numpy()[2])
        plt.show()
        print(z.cpu().detach().numpy())

    def show_real(self, label=None):
        plt.figure(figsize=(32, 16))
        if label is None:
            data = np.vstack((self.dataloader.x_train,
                              self.dataloader.x_test))
        else:
            data = np.vstack((self.dataloader.x_train[self.dataloader.y_train.squeeze() == label],
                              self.dataloader.x_test[self.dataloader.y_test.squeeze() == label]))

        plot_signatures(data, 1, 1,
                        label=0, x_real=None)
        plt.show()

    def generate(self, z):
        plt.figure(figsize=(32, 16))

        batch_size = 5
        z_ = z.copy()
        c = np.linspace(-2, 2, batch_size).reshape(1, -1)
        c_plot = c.copy()
        z = z.repeat(batch_size, 0).reshape(-1, self.size_total)
        z[:, -3] = c
        # z = np.vstack((z_, z))
        z = torch.cuda.FloatTensor(z)
        z_ = torch.cuda.FloatTensor(z_)
        fakes = self.G(z)
        fakes_ = self.G(z_).unsqueeze(0)
        plot_signatures(fakes.cpu().detach().numpy(), 3, 2,
                        label=0, x_real=None, legend=c_plot)

        plot_signatures(fakes_.cpu().detach().numpy(), 3, 2,
                        label=1, x_real=None)

        plt.show()


if __name__ == '__main__':

    out_path = "generated_leaf_infogan-n_classes{}-n_discrete{}-n_conti{}-n_noise{}{}".format(opt.nc,
                                                                                              opt.n_dis,
                                                                                              opt.n_conti,
                                                                                              opt.n_noise,
                                                                                              opt.outf_suffix)

    if opt.semisup:
        out_path += "_supervised"

    config_path = "/home/patrick/repositories/hyperspectral_phenotyping_gan/trained_models/{}".format(out_path)
    # config_path += "/model{}".format("_ratio-{}".format(int(sup_ratio * 100)) if opt.semisup else "")
    config_path += "/model{}".format("")
    config = load_config(os.path.join(config_path, "config.p"))

    size_total = config["NOISE"] + config["N_CONTI"] + (config["NC"] * config["N_DISCRETE"])
    g = G(size_total)

    NETG_generate = "/home/patrick/repositories/hyperspectral_phenotyping_gan/trained_models/{}/model{}/netG_epoch_{}{}.pth".format(
        out_path, "{}", "{}", "-crossval-0")
    # NETG_generate = NETG_generate.format("_ratio-{}".format(int(sup_ratio * 100)) if opt.semisup else "", opt.epoch)
    NETG_generate = NETG_generate.format("", opt.epoch)

    g.load_state_dict(torch.load(NETG_generate))
    g.eval()

    for i in [g]:
        i.cuda()

    generator = Generator(g, config)
    #generator.inpaint()

    z = np.array([[0.01107442, -0.848209, -0.7097479, -0.7720386, -0.2420094, 0.40806794, -0.02045506,
                   0.87952113, -0.6333749, 0.09759343, -0.50705755, -0.8597003, -0.05239939]])
    #generator.generate(z)

    #generator.show_real(label=2)
    generator.generate_code()
