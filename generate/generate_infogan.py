# -*- coding: utf-8 -*-
# @Author: aaronlai

import argparse
import numpy as np

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import sys
sys.path.append("/home/patrick/repositories/hyperspectral_phenotyping_gan")
from data_loader import DataLoader
from models.generator import InfoGAN_Generator as Generator
from config.config import *

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', required=True, help='epoch...')
parser.add_argument('--semisup', action="store_true", help='True: semi-supervised | False: unsupervised')
parser.add_argument('--sup_ratio', default=1.0, required=False, help='ratio of semi-supervised labels')
parser.set_defaults(dataset="leaf")
parser.set_defaults(semisup=False)

opt = parser.parse_args()
sup_ratio = float(opt.sup_ratio)


NETG_generate = NETG_generate.format("_ratio-{}".format(int(sup_ratio*100)) if opt.semisup else "", opt.epoch)


def activate_dropout(m):
    classname = m.__class__.__name__
    # if classname.find('Dropout') != -1:
    # m.train()


def gen_noise_same(n_instance, n_dim=2):
    """generate n-dim uniform random noise"""
    rand_noise = np.random.uniform(low=-1.0, high=1.0,
                      size=(1, n_dim))

    noise = np.repeat(rand_noise, n_instance, axis=0)
    return torch.Tensor(noise)


def gen_noise(n_instance, n_dim=2):
    """generate n-dim uniform random noise"""
    return torch.Tensor(np.random.uniform(low=-1.0, high=1.0,
                                          size=(n_instance, n_dim)))


def gen_conti_codes(n_instance, n_conti, mean=0, std=1):
    """generate gaussian continuous codes with specified mean and std"""
    if n_conti == 0:
        return torch.Tensor([])
    codes = np.random.randn(n_instance, n_conti) * std + mean

    return torch.Tensor(codes)


def gen_conti_codes_with_value(n_instance, n_conti, mean=0, std=1, value=1):
    """generate gaussian continuous codes with specified mean and std"""
    if n_conti == 0:
        return torch.Tensor([])
    codes = []
    for c in range(n_conti):
        if c == 2 or c == 1 or c== 0:
            code = np.ones((n_instance, 1)) * 8 + mean
        else:
            #code = np.ones((n_instance, 1)) * 0 + mean
            code = np.random.randn(n_instance, 1) * std + mean
        codes.append(code)
    # code = np.ones((n_instance, 1)) * 0 + mean
    # codes.append(code)
    codes = np.concatenate(codes, 1)

    return torch.Tensor(codes)


def gen_discrete_code_balanced(n_instance, n_discrete, num_category=10):
    """generate discrete codes with n categories"""
    codes = []
    cate_instance = n_instance // num_category
    for i in range(n_discrete):
        code = np.zeros((n_instance, num_category))
        for c in range(num_category):
            random_cate = np.random.randint(c, c + 1, n_instance // num_category)
            code[range(c * cate_instance, c * cate_instance + cate_instance), random_cate] = 1.
        codes.append(code)

    codes = np.concatenate(codes, 1)
    return torch.Tensor(codes)


def gen_discrete_code_for_single_label(n_instance, num_category=10, label=0):
    """generate discrete codes with n categories"""
    codes = []
    code = np.zeros((n_instance, num_category))
    #code[range(n_instance), 1] = 1  # combine with desease
    code[range(n_instance), label] = 1
    codes.append(code)
    codes = np.concatenate(codes, 1)
    return torch.Tensor(codes)


def run_InfoGAN(InfoGAN_Gen,
                  n_conti, n_discrete, mean, std,
                  num_category, batch_size, noise_dim,
                  use_gpu=False, label=0):
    """train InfoGAN and print out the losses for D and G"""  # get inputs (noises and codes) for Generator
    noises = Variable(gen_noise(batch_size, n_dim=noise_dim))
    #conti_codes = Variable(gen_conti_codes_with_value(batch_size, n_conti,
    #                                      mean, std, 0))
    conti_codes = Variable(gen_conti_codes(batch_size, n_conti,
                                                     mean, std))
    discr_codes = Variable(gen_discrete_code_for_single_label(batch_size, num_category, label=label))
    if use_gpu:
        noises = noises.cuda()
        conti_codes = conti_codes.cuda()
        discr_codes = discr_codes.cuda()

    # generate fake images
    gen_inputs = torch.cat((noises, conti_codes, discr_codes), 1)
    fakes = InfoGAN_Gen(gen_inputs)

    return fakes.cpu().detach().numpy()


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


def run_InfoGAN_conditioned_label(noise_dim=10,
                                  n_conti=2, n_discrete=1, mean=0.0, std=1., num_category=3,
                                  batch_size=50, use_gpu=False):
    # loading data
    InfoGAN_Gen = Generator(noise_dim, n_conti, n_discrete, num_category,
                            NGF)

    InfoGAN_Gen.load_state_dict(torch.load(NETG_generate))
    InfoGAN_Gen.eval()
    InfoGAN_Gen.apply(activate_dropout)

    if use_gpu:
        InfoGAN_Gen = InfoGAN_Gen.cuda()

    dataloader = DataLoader(batch_size=batch_size)
    batch_x_mean, _ = dataloader.fetch_samples_mean()
    batch_x_mean = batch_x_mean.numpy()
    batch_x_real, batch_y_real = dataloader.fetch_samples(num_sample_each_class=batch_size)
    batch_x_real, batch_y_real = batch_x_real.numpy(), batch_y_real.numpy()

    plt.figure(figsize=(32, 16))
    for category in range(num_category):
        fakes = run_InfoGAN(InfoGAN_Gen, n_conti, n_discrete, mean, std, num_category, batch_size, noise_dim, use_gpu,
                              label=category)

        #category = 1 if category == 0 else 0 if category == 1 else 2
        #category = 1 if category == 2 else 2 if category == 0 else 0
        #category = 1 if category == 2 else 2 if category == 0 else 0
        #category = 0 if category == 2 else 2 if category == 0 else 1
        mean_of_category = None if category > 2 else batch_x_mean[category]
        x_real_of_category = None if category > 2 else batch_x_real[batch_y_real == category]
        plot_fake(fakes, num_category, category, mean_of_category, x_real_of_category)



if __name__ == '__main__':
    # test conditional label
    print(NC)
    run_InfoGAN_conditioned_label(noise_dim=NOISE, n_conti=N_CONTI, n_discrete=N_DISCRETE, num_category=NC,
                                  mean=CONTI_MEAN, std=CONTI_STD,
                                  use_gpu=True, batch_size=80 * NC)
    plt.show()
