# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os

import torch
import torch.optim as optim
from torch.autograd import Variable
import sys
sys.path.append("/home/patrick/repositories/hyperspectral_phenotyping_gan")
from data_loader import DataLoader
from models.discriminator import InfoGAN_Discriminator as Discriminator
from models.generator import InfoGAN_Generator as Generator
from config.config import config_dict as config
import time
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--semisup', action="store_true", help='True: semi-supervised | False: unsupervised')
parser.add_argument('--sup_ratio', default=1.0, required=False, help='ratio of semi-supervised labels')
parser.add_argument('--pretrained', help='use pretrained model for initialization', action="store_true")
parser.add_argument('--pretrained_epoch', default=0, help='epoch of pretrained model')
parser.set_defaults(pretrained=False)
parser.set_defaults(semisup=False)
opt = parser.parse_args()

sup_ratio = float(opt.sup_ratio)  # 0.1

try:
    os.makedirs(config["OUTF"])
except OSError:
    pass

OUTF_samples = config["OUTF"] + "/samples" + ("_ratio-{}".format(int(sup_ratio * 100)) if opt.semisup else "")
OUTF_model = config["OUTF"] + "/model" + ("_ratio-{}".format(int(sup_ratio * 100)) if opt.semisup else "")

config["NETG"] = config["NETG"].format("_ratio-{}".format(int(sup_ratio * 100)) if opt.semisup else "", opt.pretrained_epoch)
config["NETD"] = config["NETD"].format("_ratio-{}".format(int(sup_ratio * 100)) if opt.semisup else "", opt.pretrained_epoch)

try:
    os.makedirs(OUTF_samples)
except OSError:
    pass
    #raise ValueError("OUT-dir already exists")
try:
    os.makedirs(OUTF_model)
except OSError:
    pass
    #raise ValueError("OUT-dir already exists")


def save_config(path_config, data):
    pickle.dump(data, open(path_config, "wb"))
    print("Saved config to:", path_config)


def load_config(path_config):
    print("Using pretrained model")
    print("Loading config from:", path_config)
    return pickle.load(open(path_config, "rb"))


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


def gen_discrete_code(n_instance, n_discrete, num_category=10):
    """generate discrete codes with n categories"""
    codes = []
    for i in range(n_discrete):
        code = np.zeros((n_instance, num_category))
        random_cate = np.random.randint(0, num_category, n_instance)
        code[range(n_instance), random_cate] = 1
        codes.append(code)

    codes = np.concatenate(codes, 1)
    return torch.Tensor(codes)


def train_InfoGAN(InfoGAN_Dis, InfoGAN_Gen, dis_lr, gen_lr, info_reg_discrete, info_reg_conti,
                  n_conti, n_discrete, mean, std, num_category, dataloader,
                  n_epoch, batch_size, noise_dim,
                  n_update_dis=1, n_update_gen=1, use_gpu=False):
    """train InfoGAN and print out the losses for D and G"""

    # define number of batches
    num_batches = dataloader.size_train // batch_size

    # setup training from earlier checkpoint
    start = 0
    scheduler_milestones = [15000]  # reduce learning rate after 15000 steps

    print("GAN balanced train size:", dataloader.size_train)
    print("-"*42)
    print("Starting training")
    print("-"*42)

    indices_disc_fake = torch.LongTensor(range(batch_size, batch_size * 2))
    if opt.pretrained:
        start = opt.pretrained_epoch + 1
        scheduler_milestones = [15000 - opt.pretrained_epoch]
        if opt.pretrained_epoch >= 15000:
            gen_lr = 1e-4
            scheduler_milestones = []

    #
    # assign loss function and optimizer (Adam) to D and G
    D_criterion = torch.nn.BCELoss()
    D_optimizer = optim.Adam(InfoGAN_Dis.parameters(), lr=dis_lr,
                             betas=(0.5, 0.999))

    G_criterion = torch.nn.BCELoss()
    G_optimizer = optim.Adam(InfoGAN_Gen.parameters(), lr=gen_lr,
                             betas=(0.5, 0.999))

    # D_optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(D_optimizer, [15000], gamma=10., last_epoch=0)
    #G_optimizer_scheduler = torch.optim.lr_scheduler.MultiStepLR(G_optimizer, scheduler_milestones,
    #                                                             gamma=0.1, last_epoch=-1)

    for epoch in range(start, n_epoch):
        # D_optimizer_scheduler.step()
        #G_optimizer_scheduler.step()

        D_running_loss = 0.0
        G_running_loss = 0.0

        for i in range(num_batches):
            # get next batch
            start_time1 = time.process_time()
            true_inputs, true_labels, real_sup_indices = dataloader.fetch_batch(onehot=True, num_classes=num_category)
            end_time1 = time.process_time()
            if epoch == 0 and i == 0:
                print("Data loading time 1/{} steps per epoch:".format(num_batches),
                      (end_time1 - start_time1) * 1000, "ms")

            start_time2 = time.process_time()

            # get the inputs from true distribution
            if use_gpu:
                true_inputs = true_inputs.cuda()
                true_labels = true_labels.cuda()
                real_sup_indices = real_sup_indices.cuda()
                indices_disc_fake = indices_disc_fake.cuda()

            # get inputs (noises and codes) for Generator
            noises = Variable(gen_noise(batch_size, n_dim=noise_dim))
            conti_codes = Variable(gen_conti_codes(batch_size, n_conti,
                                                   mean, std))
            discr_codes = Variable(gen_discrete_code(batch_size, n_discrete,
                                                     num_category))
            if use_gpu:
                noises = noises.cuda()
                conti_codes = conti_codes.cuda()
                discr_codes = discr_codes.cuda()

            # generate fake signatures
            gen_inputs = torch.cat((noises, conti_codes, discr_codes), 1)
            fake_inputs = InfoGAN_Gen(gen_inputs)

            inputs = torch.cat([true_inputs, fake_inputs])

            # make a minibatch of labels for fake/real discrimination
            labels = np.zeros(2 * batch_size)
            labels[:batch_size] = 1
            labels = torch.from_numpy(labels.astype(np.float32))
            if use_gpu:
                labels = labels.cuda()
            labels = Variable(labels)

            # Discriminator
            D_optimizer.zero_grad()
            outputs = InfoGAN_Dis(inputs)

            # add supervision
            if opt.semisup:
                discr_codes_for_reg_discrete = torch.cat((true_labels[real_sup_indices], discr_codes))
                output_shift_indices_discrete = torch.cat((real_sup_indices, indices_disc_fake))
            else:
                discr_codes_for_reg_discrete = discr_codes
                output_shift_indices_discrete = torch.range(start=batch_size, end=batch_size * 2 - 1).long()

            # calculate mutual information lower bound L(G, Q)
            #
            # of discrete code
            for j in range(n_discrete):
                shift = (j * num_category)
                start = 1 + n_conti + shift
                end = start + num_category

                Q_cx_discr = outputs[output_shift_indices_discrete, start:end]
                codes = discr_codes_for_reg_discrete[:, shift:(shift + num_category)]

                condi_entro = -torch.mean(torch.sum(Q_cx_discr * codes, 1))

                if j == 0:
                    L_discrete = -condi_entro
                else:
                    L_discrete -= condi_entro
            L_discrete /= n_discrete

            # of continuous code
            Q_cx_conti = outputs[batch_size:, 1:(1 + n_conti)]
            L_conti = torch.mean(-(((Q_cx_conti - mean) / std) ** 2))

            # Update Discriminator

            D_loss = D_criterion(outputs[:, 0], labels)
            if n_discrete > 0:
                D_loss = D_loss - info_reg_discrete * L_discrete

            if n_conti > 0:
                D_loss = D_loss - info_reg_conti * L_conti

            if n_update_dis > 0 and i % n_update_dis == 0:
                D_loss.backward(retain_graph=True)
                D_optimizer.step()

            # Update Generator
            if n_update_gen > 0 and i % n_update_gen == 0:
                G_optimizer.zero_grad()
                G_loss = G_criterion(outputs[batch_size:, 0],
                                     labels[:batch_size])

                if n_discrete > 0:
                    G_loss = G_loss - info_reg_discrete * L_discrete

                if n_conti > 0:
                    G_loss = G_loss - info_reg_conti * L_conti

                G_loss.backward()
                G_optimizer.step()
            end_time2 = time.process_time()
            if epoch == 0 and i == 0:
                print("Learning time:", (end_time2 - start_time2) * 1000, "ms")
            # print statistics
            D_running_loss += D_loss.data.item()  # [0]
            G_running_loss += G_loss.data.item()  # [0]
        if (epoch + 1) % 10 == 0:  # print_every == (print_every - 1) or i == num_batches -1:
            print('[%d] D loss: %.3f ; G loss: %.3f' %
                  (epoch + 1, D_running_loss / num_batches,
                   G_running_loss / num_batches))
            # D_running_loss = 0.0
            # G_running_loss = 0.0
        # checkpointing
        if (epoch + 1) % 100 == 0 or epoch == n_epoch - 1:
            dataloader.save_image(fake_inputs.detach()[:],
                                  '%s/fake_samples_epoch_%03d{}.png' % (OUTF_samples, epoch + 1))
        if (epoch + 1) <= 1000 and (epoch + 1) % 100 == 0:
            dataloader.save_model(InfoGAN_Gen.state_dict(), '%s/netG_epoch_%d{}.pth' % (OUTF_model, epoch + 1))
            dataloader.save_model(InfoGAN_Dis.state_dict(), '%s/netD_epoch_%d{}.pth' % (OUTF_model, epoch + 1))
        if (epoch + 1) % 1000 == 0 or epoch == n_epoch - 1:
            dataloader.save_model(InfoGAN_Gen.state_dict(), '%s/netG_epoch_%d{}.pth' % (OUTF_model, epoch + 1))
            dataloader.save_model(InfoGAN_Dis.state_dict(), '%s/netD_epoch_%d{}.pth' % (OUTF_model, epoch + 1))

    print('Finished Training')


def run_InfoGAN(info_reg_discrete=1., info_reg_conti=0.5, noise_dim=10,
                n_conti=2, n_discrete=1, mean=0.0, std=0.5, num_category=3,
                n_epoch=2, batch_size=50, use_gpu=False, dis_lr=1e-4,
                gen_lr=1e-3, n_update_dis=1, n_update_gen=1):
    # loading data
    dataloader = DataLoader(batch_size=batch_size, sup_ratio=sup_ratio)

    D_featmap_dim = 1024
    # initialize models
    InfoGAN_Dis = Discriminator(n_conti, n_discrete, num_category,
                                D_featmap_dim,
                                config["NDF"], config["NGF"])
    InfoGAN_Dis.apply(weights_init)
    if opt.pretrained and config["NETD"] != '':
        InfoGAN_Dis.load_state_dict(torch.load(config["NETD"]))

    InfoGAN_Gen = Generator(noise_dim, n_conti, n_discrete, num_category,
                            config["NGF"])
    InfoGAN_Gen.apply(weights_init)
    if opt.pretrained and config["NETG"] != '':
        InfoGAN_Gen.load_state_dict(torch.load(config["NETG"]))

    if use_gpu:
        InfoGAN_Dis = InfoGAN_Dis.cuda()
        InfoGAN_Gen = InfoGAN_Gen.cuda()

    train_InfoGAN(InfoGAN_Dis, InfoGAN_Gen, dis_lr, gen_lr, info_reg_discrete, info_reg_conti,
                  n_conti, n_discrete, mean, std, num_category, dataloader,
                  n_epoch, batch_size, noise_dim,
                  n_update_dis, n_update_gen, use_gpu)


if __name__ == '__main__':
    if opt.pretrained:
        config = load_config(os.path.join(OUTF_model, "config.p"))
    else:
        save_config(os.path.join(OUTF_model, "config.p"), config)

    run_InfoGAN(info_reg_discrete=1., info_reg_conti=0.5,
                num_category=config["NC"],
                noise_dim=config["NOISE"], n_conti=config["N_CONTI"], n_discrete=config["N_DISCRETE"],
                mean=config["CONTI_MEAN"], std=config["CONTI_STD"],
                n_update_dis=1, n_update_gen=1,
                use_gpu=True,
                dis_lr=1e-4, gen_lr=1e-4,
                n_epoch=50000, batch_size=256)
