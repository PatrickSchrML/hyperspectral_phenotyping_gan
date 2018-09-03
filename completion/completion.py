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
from data_loader_hdr import Hdr_dataset
from data_loader_mat import Mat_dataset
# from models.networks import G_with_fc as G
from models.networks import G_with_fc_nopadding as G
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from matplotlib import style

from sklearn.neighbors import KDTree

style.use("ggplot")

cmap_for_plt = plt.get_cmap("winter")

parser = argparse.ArgumentParser()
parser.add_argument('--nc', default=3, required=False, help='dim of category code or number of classes')
parser.add_argument('--n_conti', default=2, required=False, help='')
parser.add_argument('--n_dis', default=1, required=False, help='')
parser.add_argument('--n_noise', default=10, required=False, help='')

parser.add_argument('--outf_suffix', default="", required=False, help='')
parser.add_argument('--dataset', default="mat", help='mat or hdr')

parser.add_argument('--epoch', required=True, help='epoch...')
parser.add_argument('--semisup', action="store_true", help='True: semi-supervised | False: unsupervised')
parser.add_argument('--sup_ratio', default=1.0, required=False, help='ratio of semi-supervised labels')
parser.add_argument('--interpolate', action="store_true", help='')
parser.add_argument('--func_to_call', default=0, required=True)
parser.set_defaults(semisup=False)
parser.set_defaults(interpolate=False)
opt = parser.parse_args()

sup_ratio = float(opt.sup_ratio)  # 0.1


def load_config(path_config):
    print("Using pretrained model")
    print("Loading config from:", path_config)
    return pickle.load(open(path_config, "rb"))


class Generator:
    def __init__(self, G, config, train=False, eval=False):
        self.G = G

        self.dim_noise = config["NOISE"]
        self.dim_code_conti = config["N_CONTI"]
        self.dim_code_disc = (config["NC"] * config["N_DISCRETE"])
        self.num_categories = config["NC"]
        self.size_total = self.dim_noise + self.dim_code_conti + (self.num_categories * config["N_DISCRETE"])
        self.size_signature = config["NDF"]
        self.batch_size = 128

        if opt.dataset == "mat":
            print("SMALL Dataset")
            self.dataset = Mat_dataset(train=train, eval=eval)
        else:
            self.dataset = Hdr_dataset(load_to_mem=True, train=train)

        self.num_batches = len(self.dataset) // self.batch_size
        print("Num samples:", len(self.dataset), ", Num batches:", self.num_batches)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                     shuffle=False, num_workers=1)

        self.complete_data, self.complete_labels, _ = self.dataset.get_complete_data()
        self.kdt = None
        self.get_nearest_neighbor()

    def _plot_signatures(self, fakes, num_rows=3, num_col=2, current_row=0, current_col=1,
                         x_real=None, legend=None, alpha=0.5, linewidth_fake=6, title="",
                         ylim=(0, 1.0), color=None, with_nearest_neighbor=False):
        current_plot = current_row * num_col + current_col


        if x_real is not None:
            # PLOT REAL DATA
            ax = plt.subplot(num_rows, num_col, current_plot)
            # ax.grid(True)
            plt.ylim(ylim[0], ylim[1])
            plt.yticks(np.arange(0, ylim[1], .2))
            plt.xticks(np.arange(0, 180, 60))
            plt.title("Real", fontsize=20)

            x = x_real.flatten()[:-3]  # hack to hide problem of conv at end of signature
            linestyle = "--"
            _, = plt.plot(x, linestyle, linewidth=6, alpha=0.5, color="b")
            # plt.legend([l1], ["fake sample" + str(label[idx])])
            current_plot += 1

        # PLOT FAKE DATA
        for idx, x in enumerate(fakes):
            clipping = -3
            ax = plt.subplot(num_rows, num_col, current_plot)
            # ax.grid(True)
            plt.ylim(ylim[0], ylim[1])
            plt.yticks(np.arange(0, ylim[1], .2))
            plt.xticks(np.arange(0, 180, 60))
            plt.title(title, fontsize=20)

            x = x.flatten()  # hack to hide problem of conv at end of signature
            linestyle = "-"

            if color is None:
                color = list(cmap_for_plt(current_col / num_col))
                color[3] = alpha

            _, = plt.plot(x[:clipping], linestyle, linewidth=linewidth_fake, color=color, alpha=alpha)

            if with_nearest_neighbor:
                neighbor_idx = self.get_nearest_neighbor([x])
                neighbor = self.complete_data[neighbor_idx.squeeze()]
                neighbor_label = self.complete_labels[neighbor_idx.squeeze()]
                _, = plt.plot(neighbor[:clipping], linestyle, linewidth=linewidth_fake, color="r",
                              alpha=min(alpha-0.5, 0.3))
                #plt.title("NN-label: {}".format(neighbor_label.item()), fontsize=20)

            if legend is not None:
                legend = legend.squeeze()
                plt.legend(legend, fontsize=16)

        frame = plt.gca()
        frame.axes.xaxis.set_ticklabels([])
        frame.axes.yaxis.set_ticklabels([])

    def _set_noise(self, noise, code_disc, code_conti):
        z_ = []  # [noise, dis_c, con_c]
        if self.dim_noise != 0 and noise is not None:
            z_.append(noise)
        if self.dim_code_disc != 0 and code_disc is not None:
            z_.append(code_disc)
        if self.dim_code_conti != 0 and code_conti is not None:
            z_.append(code_conti)

        z = torch.cat(z_, 1).view(-1, self.size_total)
        return z

    # generates code of inputs by optimizing z
    def _generate_representation(self, x_inputs, code_conti=None, code_disc=None, noise=None,
                                 batch_size=None, n_epochs=5000, fixed_noise=False):
        """
        z = torch.zeros([self.batch_size, self.dim_noise]).cuda()
        z = Variable(z, requires_grad=True)
        z.data.resize_(self.batch_size, self.size_total)
        """
        if batch_size is None:
            batch_size = self.batch_size
        if code_conti is None:
            code_conti = torch.zeros([batch_size, self.dim_code_conti]).cuda()
        if code_disc is None:
            # code_disc = torch.ones([1, self.dim_code_disc]).cuda()
            code_disc = torch.zeros([1, self.dim_code_disc]).uniform_(0, 1).cuda()
            code_disc = code_disc / torch.sum(code_disc)
            code_disc = code_disc.repeat(batch_size, 1)
        if noise is None:
            noise = torch.cuda.FloatTensor(1, self.dim_noise).uniform_(-1, 1).repeat(batch_size, 1)

        x_inputs = x_inputs.cuda()
        code_conti = Variable(code_conti, requires_grad=True)
        code_disc = Variable(code_disc, requires_grad=True)
        noise = Variable(noise, requires_grad=(not fixed_noise and self.dim_noise > 0))

        variables = []
        if not fixed_noise and self.dim_noise > 0:
            variables.append(noise)
        if self.dim_code_disc > 0:
            variables.append(code_disc)
        if self.dim_code_conti > 0:
            variables.append(code_conti)

        print("Optimizing {} variables".format(len(variables)))

        data_shape = [batch_size, self.size_signature]
        mask = np.ones(data_shape)
        mask = torch.cuda.FloatTensor(mask)
        # l = 45
        # u = 120
        # mask[:, l:u] = 0.0

        # optimizer = optim.Adam([z], lr=0.01, betas=(0.5, 0.999))  # [code, noise]
        optimizer = optim.Adam(variables, lr=0.01, betas=(0.5, 0.999))

        # fakes = None
        # z = None

        for i in tqdm(range(n_epochs)):
            code_disc_ = code_disc  # torch.nn.functional.softmax(code_disc, dim=1)
            # if i == 0 or i == n_epochs -1 or i == n_epochs // 2:
            #    print(code_disc_)
            z = self._set_noise(noise, code_disc_, code_conti)
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
            complete_loss = contextual_loss  # + 0.1*((1 - torch.sum(code_disc_))**2) #+ 0.1 * (torch.sum(code_disc_)**2)

            optimizer.zero_grad()
            complete_loss.backward()
            optimizer.step()

        # z = self._set_noise(noise, code_disc_, code_conti)
        # fakes = self.G(z)

        return fakes, z

    # generates code for the whole dataset (test or train) and dumps the data to storage
    def generate_code(self, n_epochs):
        # x_inputs, y_inputs = self.dataloader.fetch_samples(num_sample_each_class=1)

        code_conti_init = torch.zeros([1, self.dim_code_conti]).cuda()
        code_disc_init = torch.zeros([1, self.dim_code_disc]).uniform_(0, 1).cuda()
        code_disc_init = code_disc_init / torch.sum(code_disc_init)
        noise_init = torch.cuda.FloatTensor(1, self.dim_noise).uniform_(-1, 1)

        data = dict()
        data["x"] = list()
        data["y"] = list()
        data["fake"] = list()
        data["z"] = list()
        data["origin_indices_in_img"] = self.dataset.indices
        data["meta"] = self.dataset.meta

        for idx, (batch_x, batch_y, _) in tqdm(enumerate(self.dataloader)):
            batch_y = batch_y.cpu().detach().numpy()  # key of lu_table to file
            code_conti = code_conti_init.repeat(batch_x.size(0), 1)
            code_disc = code_disc_init.repeat(batch_x.size(0), 1)
            noise = noise_init.repeat(batch_x.size(0), 1)
            fakes, z = self._generate_representation(batch_x.float(),
                                                     code_conti=code_conti, code_disc=code_disc, noise=noise,
                                                     batch_size=batch_x.size(0), fixed_noise=False, n_epochs=n_epochs)
            fakes = fakes.cpu().detach().numpy()
            z = z.cpu().detach().numpy()
            # plot_signatures(fakes[:1], 3, 2,
            #                label=0, x_real=batch_x[0])
            # plt.show()
            data["x"] += batch_x.tolist()
            data["y"] += batch_y.tolist()
            data["fake"] += fakes.tolist()
            data["z"] += z.tolist()

        pickle.dump(data,
                    open(
                        "./experiments/generated_code_dataset_{}_classes{}_disc{}_conti{}_noise{}_epoch{}{}.p".format(
                            opt.dataset,
                            self.num_categories,
                            self.dim_code_disc,
                            self.dim_code_conti,
                            self.dim_noise,
                            opt.epoch,
                            opt.outf_suffix),
                        "wb"))

    def generate_and_interpolate(self, num_plt_rows=3, n_epochs=5000,
                                 dim_to_interpolate=0, num_interpolations=9,
                                 start_interpolation=-2.,
                                 end_interpolation=2., fixed_noise=True):

        plt.figure(figsize=(32, 20))
        np.random.seed(2)
        #x_inputs, _ = self.dataset.fetch_samples(num_sample_each_class=1, shuffle=True)
        x_inputs, _ = self.dataset.fetch_row_col()
        x_inputs = x_inputs.cuda()
        # l = 45
        # u = 120
        # mask[:, l:u] = 0.0

        fakes, z = self._generate_representation(x_inputs, batch_size=x_inputs.shape[0],
                                                 n_epochs=n_epochs, fixed_noise=fixed_noise)
        fakes = fakes.cpu().detach().numpy()
        z = z.cpu().detach().numpy()
        x_inputs = x_inputs.cpu().detach().numpy()
        num_plt_col = 2 + num_interpolations
        for idx in range(len(fakes)):
            self._plot_signatures(fakes[idx:idx + 1], num_rows=num_plt_rows, num_col=num_plt_col,
                                  current_row=idx, x_real=x_inputs[idx],
                                  ylim=(0.0, 0.9),
                                  title="Fake c_{}={:.2f}".format(dim_to_interpolate,
                                                                  z[idx][-(
                                                                      self.dim_code_conti - dim_to_interpolate)]))
            # for dim_to_interpolate in
            self.interpolate_continuous_code(z[idx],
                                             num_interpolations=num_interpolations,
                                             num_cols=2,
                                             num_rows=num_plt_rows,
                                             current_row=idx,
                                             dim_to_interpolate=dim_to_interpolate,
                                             start_interpolation=start_interpolation,
                                             end_interpolation=end_interpolation)
        plt.show()

    def generate_and_interpolate_between(self, n_epochs=5000, sig_class1=0, sig_class2=1,
                                         num_interpolations=4, with_nearest_neighbor=True):
        plt.figure(figsize=(32, 20))
        np.random.seed(2)  # 2
        x_inputs, _ = self.dataset.fetch_samples(num_sample_each_class=1)
        x_inputs = x_inputs.cuda()
        # l = 45
        # u = 120
        # mask[:, l:u] = 0.0

        fakes, z = self._generate_representation(x_inputs, batch_size=x_inputs.shape[0],
                                                 n_epochs=n_epochs, fixed_noise=True)
        fakes = fakes.cpu().detach().numpy()
        z = z.cpu().detach().numpy()
        x_inputs = x_inputs.cpu().detach().numpy()
        # for dim_to_interpolate in
        self.interpolate_between_to_representation(z[sig_class1], z[sig_class2],
                                                   num_interpolations=num_interpolations,
                                                   sig_class1=sig_class1, sig_class2=sig_class2,
                                                   with_nearest_neighbor=with_nearest_neighbor)
        plt.show()

    def show_real(self, label=None):
        plt.figure(figsize=(32, 16))
        if label is None:
            data = self.dataset.data
        else:
            data = self.dataset.data[self.dataset.labels.squeeze() == label]

        self._plot_signatures(data, num_rows=1, num_col=1,
                              current_row=0, x_real=None, alpha=0.4, color="b")
        plt.show()

    def generate(self, z):
        plt.figure(figsize=(32, 16))

        batch_size = 5
        z_ = z.copy()
        c = np.linspace(-2, 2, batch_size).reshape(1, -1)
        c_plot = c.copy()
        z = z.repeat(batch_size, 0).reshape(-1, self.size_total)
        z[:, -self.dim_code_conti] = c
        # z = np.vstack((z_, z))
        z = torch.cuda.FloatTensor(z)
        z_ = torch.cuda.FloatTensor(z_)
        fakes = self.G(z)
        fakes_ = self.G(z_).unsqueeze(0)
        self._plot_signatures(fakes.cpu().detach().numpy(), num_rows=3, num_col=2,
                              current_row=0, x_real=None, legend=c_plot)

        self._plot_signatures(fakes_.cpu().detach().numpy(), num_rows=3, num_col=2,
                              current_row=1, x_real=None)

        plt.show()

    def generate_from_representation(self, data_dict):
        code = np.array(data_dict["z"])
        labels = np.array(data_dict["y"]).squeeze()
        real = np.array(data_dict["x"])
        tmp = code.copy()
        # tmp[:, 1] = np.random.uniform(0., -1., size=[tmp.shape[0]])
        # tmp[:, 0] = np.random.uniform(1., 1., size=[tmp.shape[0]])  # tmp[:, 2] = np.random.uniform(0., .5, size=[tmp.shape[0]])
        #mask = np.vstack((tmp[:, 0] < 0.,)).T
        mask = np.vstack((tmp[:, 0] < 0., tmp[:, 2] < 0.37)).T
        mask = np.all(mask, axis=1)
        #tmp[mask, 2] *= -1 #np.random.uniform(.3, .8, size=[tmp[tmp[:, 0] < 0., 0].shape[0]])
        tmp[mask, 0] *= -1. #np.random.uniform(.3, .8, size=[tmp[tmp[:, 0] < 0., 0].shape[0]])

        tmp = tmp.clip(min=-1, max=1)
        # tmp[:, 0] = np.random.uniform(-0., -.25, size=[tmp.shape[0]])  # epoch 500 - sick to healthy
        # tmp[:, 2] = np.random.uniform(-0.5, -1., size=[tmp.shape[0]])  # epoch 500 - sick to vein
        # code[:, 2] = 1
        #code[labels == 1] = tmp[labels == 1]
        code = tmp
        code = torch.cuda.FloatTensor(code)
        fakes = self.G(code)
        fakes = fakes.cpu().detach().numpy()

        plt.figure(figsize=(32, 16))

        self._plot_signatures(fakes[labels == 1][:100], num_rows=1, num_col=2,
                              current_row=0, x_real=None, alpha=0.2)
        self._plot_signatures(real[labels == 1][:100], num_rows=1, num_col=2,
                              current_row=0, x_real=None, current_col=2, alpha=0.2)

        plt.show()

        data_dict["fakes_manipulated"] = fakes
        pickle.dump(data_dict,
                    open(
                        "./experiments/generated_signatures_from_representation_dataset_{}_classes{}_disc{}_conti{}_noise{}_epoch{}{}.p".format(
                            opt.dataset,
                            self.num_categories,
                            self.dim_code_disc,
                            self.dim_code_conti,
                            self.dim_noise,
                            opt.epoch,
                            opt.outf_suffix),
                        "wb"))

    def interpolate_continuous_code(self, z, dim_to_interpolate, num_interpolations=5,
                                    num_cols=2, num_rows=3, current_row=0,
                                    start_interpolation=-2., end_interpolation=2.):

        # plt.figure(figsize=(32, 16))
        if len(z.shape) == 1:
            z = np.expand_dims(z, axis=0)
        c = np.linspace(start_interpolation, end_interpolation, num_interpolations).reshape(1, -1)
        z = z.repeat(num_interpolations, 0)  # .reshape(-1, self.size_total)

        print(z[0])
        print("- " * 10)
        z[:, -(self.dim_code_conti - dim_to_interpolate)] = c
        print(z[0])
        print("--" * 10)
        # z[:, -1] = c

        z = torch.cuda.FloatTensor(z)
        fakes = self.G(z)
        fakes = fakes.cpu().detach().numpy()
        for idx, signature in enumerate(fakes):
            legend = None
            current_col = idx + 1 + 2
            self._plot_signatures(np.expand_dims(signature, axis=0),
                                  num_rows=num_rows,
                                  num_col=num_interpolations + num_cols,
                                  current_col=current_col,
                                  current_row=current_row, x_real=None, legend=legend, alpha=.8,
                                  ylim=(0.0, 0.9),
                                  title="c{}={}".format(dim_to_interpolate, str(c.squeeze()[idx])),
                                  with_nearest_neighbor=True)

    def interpolate_between_to_representation(self, z1, z2, num_interpolations=5,
                                              sig_class1=0, sig_class2=1, with_nearest_neighbor=True):

        # check if representations are from same domain
        assert z1.shape == z2.shape

        # just interpolate continuous code
        z1_conti = z1.squeeze()[-self.dim_code_conti:]
        z2_conti = z2.squeeze()[-self.dim_code_conti:]

        z_diff = np.abs(z1_conti - z2_conti)
        c_dim_max_diff = np.argmax(z_diff, axis=0)
        print("z1:", z1_conti)
        print("z2:", z2_conti)
        print("Diff:", z_diff)
        print("Max diff dim:", c_dim_max_diff)

        # interpolate between representations
        c = []
        for idx in range(len(z1_conti)):
            c.append([np.linspace(z1_conti[idx], z2_conti[idx], num_interpolations)])

        # reshaping
        c = np.array(c).squeeze()
        c = np.transpose(c, axes=(1, 0))

        # generate samples of interpolation - could also be done in one step but who cares
        z_ = z1.copy()

        #plt.suptitle("Difference of representation along continuous " +
        #             "code of classes {}/{}: {} \n Max diff dim: {}".format(sig_class1, sig_class2,
        #                                                                   str(z_diff),
        #                                                                   str(c_dim_max_diff)))

        for idx, conti_code in enumerate(c):
            z_[-self.dim_code_conti:] = conti_code
            z = torch.cuda.FloatTensor(np.expand_dims(z_, axis=0))
            fakes = self.G(z)
            fakes = fakes.cpu().detach().numpy()
            legend = None

            # plot
            self._plot_signatures(np.expand_dims(fakes, axis=0),
                                  num_rows=1,
                                  num_col=num_interpolations,
                                  current_col=idx + 1,
                                  current_row=0, x_real=None, legend=legend, alpha=.9, ylim=(0, 0.8),
                                  with_nearest_neighbor=with_nearest_neighbor)

    def get_nearest_neighbor(self, query_samples=None):
        """

        :return: nearest neigbor
        """
        if self.kdt is None:
            self.kdt = KDTree(self.complete_data, leaf_size=30, metric='manhattan') # euclidean

        result = []
        if query_samples is not None:
            result = self.kdt.query(query_samples, k=1, return_distance=False)

        return result


if __name__ == '__main__':

    out_path = "generated_leaf_infogan-n_classes{}-n_discrete{}-n_conti{}-n_noise{}{}".format(opt.nc,
                                                                                              opt.n_dis,
                                                                                              opt.n_conti,
                                                                                              opt.n_noise,
                                                                                              opt.outf_suffix)

    if opt.semisup:
        out_path += "_supervised"

    config_path = "/home/patrick/repositories/hyperspectral_phenotyping_gan/trained_models_{}/{}".format(opt.dataset,
                                                                                                         out_path)
    # config_path += "/model{}".format("_ratio-{}".format(int(sup_ratio * 100)) if opt.semisup else "")
    config_path += "/model{}".format("")
    config = load_config(os.path.join(config_path, "config.p"))

    size_total = config["NOISE"] + config["N_CONTI"] + (config["NC"] * config["N_DISCRETE"])
    g = G(size_total, config["NDF"])

    NETG_generate = "/home/patrick/repositories/hyperspectral_phenotyping_gan/trained_models_{}/{}/model{}/netG_epoch_{}{}.pth".format(
        opt.dataset, out_path,
        "{}", "{}", "-crossval-0")
    # NETG_generate = NETG_generate.format("_ratio-{}".format(int(sup_ratio * 100)) if opt.semisup else "", opt.epoch)
    NETG_generate = NETG_generate.format("", opt.epoch)

    g.load_state_dict(torch.load(NETG_generate))
    g.eval()

    for i in [g]:
        i.cuda()

    # call
    if int(opt.func_to_call) == 0:
        generator = Generator(g, config, eval=True)
        # generator = Generator(g, config)
        # generate z and corresponding signatures, then interpolate intervall of specific continuous code dim
        # dim_to_interpolate =: 0 for first conti_dim, 1 for second conti dim etc...
        generator.generate_and_interpolate(n_epochs=300, dim_to_interpolate=0, num_interpolations=9,
                                           start_interpolation=-1.5, end_interpolation=1.5, fixed_noise=True)
    elif int(opt.func_to_call) == 1:
        generator = Generator(g, config)
        # generate z and corresponding signatures, then interpolate between their representations
        generator.generate_and_interpolate_between(n_epochs=1000,
                                                   sig_class1=2, sig_class2=1,
                                                   num_interpolations=5,
                                                   with_nearest_neighbor=False)
    elif int(opt.func_to_call) == 2:
        generator = Generator(g, config)
        z = np.random.uniform(-1, 1, [size_total, 1])
        # np.array([[0.01107442, -0.848209, -0.7097479, -0.7720386, -0.2420094, 0.40806794, -0.02045506,
        # 0.87952113, -0.6333749, 0.09759343, -0.50705755, -0.8597003, -0.05239939]])
        generator.generate(z)
    elif int(opt.func_to_call) == 3:
        generator = Generator(g, config)
        generator.show_real(label=0)
    elif int(opt.func_to_call) == 4:
        generator = Generator(g, config, eval=True)
        # generate code for test_dataset
        generator.generate_code(n_epochs=300)
    elif int(opt.func_to_call) == 5:
        generator = Generator(g, config, eval=True)
        data_dict = pickle.load(
            open(
                "./experiments/generated_code_dataset_{}_classes{}_disc{}_conti{}_noise{}_epoch{}{}.p".format(
                    opt.dataset,
                    generator.num_categories,
                    generator.dim_code_disc,
                    generator.dim_code_conti,
                    generator.dim_noise,
                    opt.epoch,
                    opt.outf_suffix),
                "rb"))
        # generate code for test_dataset
        generator.generate_from_representation(data_dict)
