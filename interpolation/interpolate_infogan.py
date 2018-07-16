# -*- coding: utf-8 -*-
# @Author: aaronlai

import argparse
import numpy as np

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import sys
import pickle
import os
sys.path.append('/home/patrick/repositories/hyperspectral_phenotyping_gan')
from models.generator import InfoGAN_Generator as Generator

parser = argparse.ArgumentParser()
parser.add_argument('--nc', default=3, required=False,  help='dim of category code or number of classes')
parser.add_argument('--n_conti', default=2, required=False,  help='')
parser.add_argument('--n_dis', default=1, required=False,  help='')
parser.add_argument('--n_noise', default=10, required=False,  help='')
parser.add_argument('--epoch', required=True, help='epoch...')
parser.add_argument('--semisup', action="store_true", help='True: semi-supervised | False: unsupervised')
parser.add_argument('--sup_ratio', default=1.0, required=False, help='ratio of semi-supervised labels')
parser.set_defaults(semisup=False)

opt = parser.parse_args()
sup_ratio = float(opt.sup_ratio)


def load_config(path_config):
    print("Using pretrained model")
    print("Loading config from:", path_config)
    return pickle.load(open(path_config, "rb"))


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
        if c == 2 or c == 1 or c == 0:
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
                  noises, conti_codes, discr_codes,
                  use_gpu=False):
    """train InfoGAN and print out the losses for D and G"""  # get inputs (noises and codes) for Generator

    if use_gpu:
        noises = noises.cuda()
        conti_codes = conti_codes.cuda()
        discr_codes = discr_codes.cuda()

    # generate fake images
    gen_inputs = torch.cat((noises, conti_codes, discr_codes), 1)

    fakes = InfoGAN_Gen(gen_inputs)

    return fakes.cpu().detach().numpy()


def plot_fake(fakes, num_rows, num_cols, current_plot):
    # PLOT FAKE DATA
    ax = plt.subplot(num_rows, num_cols, current_plot)
    #ax.set_title("Label {} Fake".format(label))
    ax.grid(True)
    linestyle = "r-"

    plt.ylim(0, 1.0)
    # plt.yscale('symlog')
    plt.yticks(np.arange(0, 1.0, .1))

    for idx, x in enumerate(fakes):
        plt.subplot(num_rows, num_cols, current_plot)
        x = x.flatten()
        linestyle = "--"
        _, = plt.plot(x, linestyle, linewidth=2, alpha=0.8)
        # plt.legend([l1], ["fake sample" + str(label[idx])])


def run_InfoGAN_conditioned_label(config, noise_dim=10, n_conti=2, n_discrete=1, num_category=3, use_gpu=False):
    # loading data
    InfoGAN_Gen = Generator(noise_dim, n_conti, n_discrete, num_category,
                            config["NGF"])

    model_path = "generated_leaf_infogan-n_classes{}-n_discrete{}-n_conti{}-n_noise{}".format(opt.nc,
                                                                                              opt.n_dis,
                                                                                              opt.n_conti,
                                                                                              opt.n_noise)
    NETG_generate = "/home/patrick/repositories/hyperspectral_phenotyping_gan/trained_models/{}/model{}/netG_epoch_{}{}.pth".format(
        model_path, "{}", "{}", "-crossval-0")

    NETG_generate = NETG_generate.format("_ratio-{}".format(int(sup_ratio * 100)) if opt.semisup else "", opt.epoch)

    InfoGAN_Gen.load_state_dict(torch.load(NETG_generate))
    InfoGAN_Gen.eval()

    if use_gpu:
        InfoGAN_Gen = InfoGAN_Gen.cuda()

    num_rows = num_category
    interpolated_code = np.array([-3., -2., -1., 0., 1., 2., 3.])
    num_cols = len(interpolated_code)
    plt.figure(figsize=(32, 16))

    # fixed noise
    #np.random.seed(142)
    n_intances = 10
    noises = Variable(gen_noise(n_intances, n_dim=noise_dim))

    # fixed conti codes
    conti_codes = np.zeros([n_intances, n_conti])
    for category in range(num_category):
        # code for discrete code / cluster
        discr_codes = Variable(gen_discrete_code_for_single_label(n_intances, num_category, label=category))

        for interpolated_idx, interpolated_value in enumerate(interpolated_code):
            # change 1 code param
            #conti_codes[:, 0] = interpolated_value
            conti_codes[:, 1] = interpolated_value
            #conti_codes[:, 2] = interpolated_value
            conti_codes = Variable(torch.Tensor(conti_codes))
            #conti_codes = Variable(gen_conti_codes(1, n_conti,
            #                                       0, 1))
            # generate fakes from code and noise
            fakes = run_InfoGAN(InfoGAN_Gen,
                                noises=noises, conti_codes=conti_codes, discr_codes=discr_codes,
                                use_gpu=use_gpu)

            plt_idx = 1 + interpolated_idx + category * num_cols
            print(plt_idx)
            # plt
            plot_fake(fakes, num_rows, num_cols, plt_idx)

if __name__ == '__main__':
    # test conditional label
    config_path = "/home/patrick/repositories/hyperspectral_phenotyping_gan/trained_models/"
    config_path += "generated_leaf_infogan-n_classes{}-n_discrete{}-n_conti{}-n_noise{}".format(opt.nc,
                                                                                                opt.n_dis,
                                                                                                opt.n_conti,
                                                                                                opt.n_noise)
    config_path += "/model{}".format("_ratio-{}".format(int(sup_ratio * 100)) if opt.semisup else "")
    config = load_config(os.path.join(config_path, "config.p"))

    run_InfoGAN_conditioned_label(config, noise_dim=config["NOISE"], n_conti=config["N_CONTI"],
                                  n_discrete=config["N_DISCRETE"], num_category=config["NC"],
                                  use_gpu=True)
    plt.show()
