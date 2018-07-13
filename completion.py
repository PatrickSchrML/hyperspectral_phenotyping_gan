# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.autograd import Variable

from data_loader import DataLoader
from models.discriminator import InfoGAN_Discriminator as Discriminator
from models.generator import InfoGAN_Generator as Generator

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='leaf | in')
opt = parser.parse_args()

if opt.dataset == "leaf":
    from used_methods.hsgan.config.config_leaf_infogan import *
else:
    from used_methods.hsgan.config.config_leaf_infogan import *  # TODO
    # from config_in_infogan import *

try:
    os.makedirs(OUTF)
except OSError:
    pass
try:
    os.makedirs(OUTF + "/samples")
except OSError:
    pass
try:
    os.makedirs(OUTF + "/model")
except OSError:
    pass


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


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


def gen_discrete_code(n_instance, cate, num_category=10):
    """generate discrete codes with n categories"""
    codes = []
    code = np.zeros((n_instance, num_category))
    code[range(n_instance), cate] = 1
    codes.append(code)

    codes = np.concatenate(codes, 1)
    return torch.Tensor(codes)


def train_InfoGAN(InfoGAN_Dis, InfoGAN_Gen, c_criterion, G_criterion, info_reg_discrete, info_reg_conti,
                  n_conti, n_discrete, mean, std, num_category, dataloader,
                  n_epoch, batch_size, noise_dim,
                  n_update_dis=1, n_update_gen=1, use_gpu=False,
                  print_every=50, update_max=None):
    """train InfoGAN and print out the losses for D and G"""

    # define number of batches
    start = 0

    # get one example
    x_inputs, y_inputs = dataloader.fetch_samples(batch_size//3)
    #x_inputs = x_inputs[:1]
    if use_gpu:
        x_inputs = x_inputs.cuda()
    #y_inputs = y_inputs[:1]
    data_shape = [batch_size, 160]

    mask = np.ones(data_shape)

    l = 45
    u = 120
    mask[:, l:u] = 0.0
    l = 0
    u = 20
    mask[:, l:u] = 0.0
    #mask[:, l:u:10] = 1.0
    #l = 100
    #u = 120
    #mask[:, l:u] = 0.0

    mask = torch.FloatTensor(mask)
    #print(mask)
    if use_gpu:
        mask = mask.cuda()
    # init noise
    noise_z = Variable(gen_noise(batch_size, n_dim=noise_dim))
    conti_codes = Variable(gen_conti_codes(batch_size, n_conti,
                                           mean, std))
    discr_codes = Variable(gen_discrete_code(batch_size, y_inputs, num_category))
    gen_inputs = torch.cat((noise_z, conti_codes, discr_codes), 1)

    if use_gpu:
        gen_inputs = gen_inputs.cuda()

    z = Variable(gen_inputs, requires_grad=True)

    optimizer = optim.Adam([z], lr=0.01, betas=(0.9, 0.999))

    for i in range(start, n_epoch):

        G_running_loss = 0.0  # get the inputs from true distribution

        #if use_gpu:
            #conti_codes = conti_codes.cuda()
            #discr_codes = discr_codes.cuda()

        fakes = InfoGAN_Gen(z)

        # make a minibatch of labels
        labels = np.ones(batch_size)
        labels = torch.from_numpy(labels.astype(np.float32))
        if use_gpu:
            labels = labels.cuda()
        labels = Variable(labels)

        # Discriminator
        outputs = InfoGAN_Dis(fakes)

        # calculate mutual information lower bound L(G, Q)
        #
        # of discrete code
        """for j in range(n_discrete):
            shift = (j * num_category)
            start = 1 + n_conti + shift
            end = start + num_category
            Q_cx_discr = outputs[batch_size:, start:end]  # for all fakes - category noise
            codes = discr_codes[:, shift:(shift + num_category)]
            condi_entro = -torch.mean(torch.sum(Q_cx_discr * codes, 1))

            if j == 0:
                L_discrete = -condi_entro
            else:
                L_discrete -= condi_entro
        L_discrete /= n_discrete

        # of continuous code
        Q_cx_conti = outputs[batch_size:, 1:(1 + n_conti)]
        L_conti = torch.mean(-(((Q_cx_conti - mean) / std) ** 2))"""

        # Update noise
        #contextual_loss = c_criterion(torch.mul(mask, inputs), torch.mul(mask, x_inputs))
        contextual_loss = torch.mean(torch.abs((torch.mul(mask, fakes) - torch.mul(mask, x_inputs))))

        # Update Generator
        G_loss = G_criterion(outputs[:, 0],
                                 labels[:])
        perceptual_loss = G_loss

        """if n_discrete > 0:
            G_loss = G_loss - info_reg_discrete * L_discrete

        if n_conti > 0:
            G_loss = G_loss - info_reg_conti * L_conti"""

        #lr_p = 0.01 if i > n_epoch - 300 else 0.1
        complete_loss = contextual_loss + 0.01 * perceptual_loss

        optimizer.zero_grad()
        complete_loss.backward()
        optimizer.step()
        # print statistics
        #G_running_loss += G_loss.data.item()  # [0]
        if i % print_every == (print_every - 1):
            print('[%5d], c_loss %.3f, p_loss: %.3f' %
                  (i + 1 / print_every, contextual_loss, perceptual_loss))

    ax = plt.subplot(1, 1, 1)
    #ax.set_title("Label {} Fake".format(label))
    ax.grid(True)
    #plt.ylim(0, 1.0)
    #plt.yscale('symlog')
    #plt.yticks(np.arange(0, 1.0, .1))

    #gen = fakes.cpu().detach().numpy()[0]
    mask_inv = torch.abs(mask - 1)
    completed_inputs = torch.mul(mask_inv, fakes) + torch.mul(mask, x_inputs)
    completed_inputs = completed_inputs.cpu().detach().numpy()

    reals = x_inputs.cpu().detach().numpy()
    for completed_input in completed_inputs[:]:
        _, = plt.plot(completed_input, "r-", linewidth=4, alpha=0.7)
    for real in reals[:]:
        _, = plt.plot(real, "g--", linewidth=4, alpha=0.7)
    #_, = plt.plot(real, "g--", linewidth=4)
    plt.show()

    print('Finished Training')


def run_InfoGAN(info_reg_discrete=1., info_reg_conti=0.5, noise_dim=10,
                n_conti=2, n_discrete=1, mean=0.0, std=0.5, num_category=3,
                n_epoch=2, batch_size=50, use_gpu=False, dis_lr=1e-4,
                gen_lr=1e-3, n_update_dis=1, n_update_gen=1, update_max=None):
    # loading data
    dataloader = DataLoader(batch_size=batch_size, dataset=opt.dataset)

    D_featmap_dim = 1024
    # initialize models
    InfoGAN_Dis = Discriminator(n_conti, n_discrete, num_category,
                                D_featmap_dim,
                                NDF, NGF)
    InfoGAN_Dis.load_state_dict(torch.load(NETD_completion))
    InfoGAN_Dis.eval()

    InfoGAN_Gen = Generator(noise_dim, n_conti, n_discrete, num_category,
                            NGF)
    InfoGAN_Gen.load_state_dict(torch.load(NETG_completion))
    InfoGAN_Gen.eval()
    if use_gpu:
        InfoGAN_Dis = InfoGAN_Dis.cuda()
        InfoGAN_Gen = InfoGAN_Gen.cuda()

        # assign loss function and optimizer (Adam) to D and G
    c_criterion = torch.nn.L1Loss()

    G_criterion = torch.nn.BCELoss()


    train_InfoGAN(InfoGAN_Dis, InfoGAN_Gen, c_criterion, G_criterion, info_reg_discrete, info_reg_conti,
                  n_conti, n_discrete, mean, std, num_category, dataloader,
                  n_epoch, batch_size, noise_dim,
                  n_update_dis, n_update_gen, use_gpu, update_max=update_max)


if __name__ == '__main__':
    # run_InfoGAN(n_conti=2, n_discrete=1, D_featmap_dim=64, G_featmap_dim=128, use_gpu=True,
    #            n_epoch=10000, batch_size=32, update_max=2000)
    run_InfoGAN(info_reg_discrete=1., info_reg_conti=0.5,
                num_category=NC,
                noise_dim=NOISE, n_conti=N_CONTI, n_discrete=N_DISCRETE,
                mean=CONTI_MEAN, std=CONTI_STD,
                n_update_dis=2, n_update_gen=1,
                use_gpu=True,
                dis_lr=1e-4, gen_lr=1e-4,  # <- on pretrained with n_update_dis=1, n_update_gen=1
                # dis_lr=1e-4, gen_lr=1e-3,  #  <- not pretrained with n_update_dis=2, n_update_gen=1
                n_epoch=3000, batch_size=6, update_max=2000)
