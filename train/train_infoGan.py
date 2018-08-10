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
from models.networks import FrontEnd, D, Q, weights_init
from models.networks import G_with_fc as G
from config.config import config_dict as config
import time
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--pretrained', help='use pretrained model for initialization', action="store_true")
parser.add_argument('--pretrained_epoch', default=0, help='epoch of pretrained model')
parser.set_defaults(pretrained=False)
opt = parser.parse_args()

created_outf = [False, 0, config["OUTF"]]

while not created_outf[0]:
    try:
        os.makedirs(created_outf[2])
        created_outf[0] = True
    except OSError:
        created_outf[1] += 1
        created_outf[2] = config["OUTF"] + "_" + str(created_outf[1])
        pass
config["OUTF"] = created_outf[2]

OUTF_samples = config["OUTF"] + "/samples"
OUTF_model = config["OUTF"] + "/model"

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


class log_gaussian:
    def __call__(self, x, mu, var):
        logli = -0.5 * (var.mul(2 * np.pi) + 1e-6).log() - \
                (x - mu).pow(2).div(var.mul(2.0) + 1e-6)

        return logli.sum(1).mean().mul(-1)


class Trainer:
    def __init__(self, G, FE, D, Q, config):

        self.G = G
        self.FE = FE
        self.D = D
        self.Q = Q

        self.dim_noise = config["NOISE"]
        self.dim_code_conti = config["N_CONTI"]
        self.dim_code_disc = config["NC"] * config["N_DISCRETE"]
        self.num_categories = config["NC"]
        self.dim_signature = config["NDF"]
        self.size_total = self.dim_noise + self.dim_code_conti + self.dim_code_disc

        self.batch_size = 40 * self.num_categories
        self.dataloader = DataLoader(batch_size=self.batch_size)

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

    def _noise_sample(self, dis_c, con_c, noise, bs):

        idx = np.random.randint(self.num_categories, size=bs)
        c = np.zeros((bs, self.num_categories))
        c[range(bs), idx] = 1.0

        dis_c.data.copy_(torch.Tensor(c))
        con_c.data.uniform_(-1.0, 1.0)
        noise.data.uniform_(-1.0, 1.0)

        z = self._set_noise(noise, dis_c, con_c)

        return z, idx

    def train(self):

        real_x = torch.FloatTensor(self.batch_size, self.dim_signature).cuda()  # TODO
        label = torch.FloatTensor(self.batch_size).cuda()
        dis_c = torch.FloatTensor(self.batch_size, self.num_categories).cuda()
        con_c = torch.FloatTensor(self.batch_size, self.dim_code_conti).cuda()
        noise = torch.FloatTensor(self.batch_size, self.dim_noise).cuda()

        real_x = Variable(real_x)
        label = Variable(label, requires_grad=False)
        dis_c = Variable(dis_c)
        con_c = Variable(con_c)
        noise = Variable(noise)

        criterionD = nn.BCELoss().cuda()
        criterionQ_dis = nn.CrossEntropyLoss().cuda()
        criterionQ_con = log_gaussian()

        optimD = optim.Adam([{'params': self.FE.parameters()},
                             {'params': self.D.parameters()}],
                            lr=0.0001,  # 0.0002
                            betas=(0.5, 0.99))
        optimG = optim.Adam([{'params': self.G.parameters()},
                             {'params': self.FE.parameters()},
                             {'params': self.Q.parameters()}],
                            lr=0.0001,  # 0.001
                            betas=(0.5, 0.99))


        # fixed random variables
        batch_size_eval = self.batch_size
        c = np.linspace(-1, 1, batch_size_eval // self.num_categories).reshape(1, -1)
        c = np.repeat(c, self.num_categories, 0).reshape(-1, 1)
        c1 = np.hstack([c])
        if self.dim_code_conti >= 2:
            for _ in range(self.dim_code_conti - 1):
                c1 = np.hstack([c1, np.zeros_like(c)])

        idx = np.arange(self.num_categories).repeat(batch_size_eval // self.num_categories)
        one_hot = np.zeros((batch_size_eval, self.num_categories))
        one_hot[range(batch_size_eval), idx] = 1
        fix_noise = torch.Tensor(batch_size_eval, self.dim_noise).uniform_(-1, 1)

        num_batches = self.dataloader.size_train // self.batch_size

        n_epoch = 5000
        for epoch in range(n_epoch):
            for num_iters in range(num_batches):

                x, _, _ = self.dataloader.fetch_batch(onehot=False, num_classes=3)
                # real part
                optimD.zero_grad()

                bs = x.size(0)
                real_x.data.resize_(x.size())
                label.data.resize_(bs, 1)
                dis_c.data.resize_(bs, self.num_categories)
                con_c.data.resize_(bs, self.dim_code_conti)
                noise.data.resize_(bs, self.dim_noise)
                real_x.data.copy_(x)
                fe_out1 = self.FE(real_x)
                probs_real = self.D(fe_out1)
                label.data.fill_(1)
                loss_real = criterionD(probs_real, label)
                loss_real.backward()

                # fake part
                z, idx = self._noise_sample(dis_c, con_c, noise, bs)
                fake_x = self.G(z)
                fe_out2 = self.FE(fake_x.detach())
                probs_fake = self.D(fe_out2)
                label.data.fill_(0)
                loss_fake = criterionD(probs_fake, label)
                loss_fake.backward()

                D_loss = loss_real + loss_fake

                optimD.step()

                # G and Q part
                optimG.zero_grad()

                fe_out = self.FE(fake_x)
                probs_fake = self.D(fe_out)
                label.data.fill_(1.0)

                reconstruct_loss = criterionD(probs_fake, label)

                q_logits, q_mu, q_var = self.Q(fe_out)

                # supervised training not implemented here, use train_infogan_supervised.üy
                #class_ = torch.LongTensor(idx).cuda()
                #target = Variable(class_)
                #dis_loss = criterionQ_dis(q_logits, target)
                con_loss = criterionQ_con(con_c, q_mu, q_var) * .1  # 0.1

                G_loss = reconstruct_loss + con_loss  # + dis_loss
                G_loss.backward()
                optimG.step()

                if num_iters == num_batches - 1 and (epoch + 1) % 100 == 0:
                    print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
                        epoch + 1, num_iters, D_loss.data.cpu().numpy(),
                        G_loss.data.cpu().numpy())
                    )

                    noise.data.copy_(fix_noise)
                    dis_c.data.copy_(torch.Tensor(one_hot))

                    con_c.data.copy_(torch.from_numpy(c1))
                    z = self._set_noise(noise, dis_c, con_c)
                    x_save = self.G(z)
                    self.dataloader.save_image(x_save.data,
                                          '%s/call_fake_samples_epoch_%03d{}.png' % (OUTF_samples, epoch + 1))

            if (epoch + 1) < 2000 and (epoch + 1) % 250 == 0:
                self.dataloader.save_model(self.G.state_dict(), '%s/netG_epoch_%d{}.pth' % (OUTF_model, epoch + 1))
                self.dataloader.save_model(self.D.state_dict(), '%s/netD_epoch_%d{}.pth' % (OUTF_model, epoch + 1))
                self.dataloader.save_model(self.FE.state_dict(), '%s/netFE_epoch_%d{}.pth' % (OUTF_model, epoch + 1))
                self.dataloader.save_model(self.Q.state_dict(), '%s/netQ_epoch_%d{}.pth' % (OUTF_model, epoch + 1))

            if (epoch + 1) >= 2000 and (epoch + 1) % 1000 == 0 or epoch == n_epoch - 1:
                self.dataloader.save_model(self.G.state_dict(), '%s/netG_epoch_%d{}.pth' % (OUTF_model, epoch + 1))
                self.dataloader.save_model(self.D.state_dict(), '%s/netD_epoch_%d{}.pth' % (OUTF_model, epoch + 1))
                self.dataloader.save_model(self.FE.state_dict(), '%s/netFE_epoch_%d{}.pth' % (OUTF_model, epoch + 1))
                self.dataloader.save_model(self.Q.state_dict(), '%s/netQ_epoch_%d{}.pth' % (OUTF_model, epoch + 1))

if __name__ == '__main__':

    save_config(os.path.join(OUTF_model, "config.p"), config)

    config["NETG"] = config["NETG"].format("", opt.pretrained_epoch)
    config["NETD"] = config["NETD"].format("", opt.pretrained_epoch)

    size_total = config["NOISE"] + config["N_CONTI"] + (config["NC"] * config["N_DISCRETE"])
    fe = FrontEnd()
    d = D()
    q = Q(dim_conti=config["N_CONTI"], dim_disc=config["NC"] * config["N_DISCRETE"])
    g = G(size_total, config["NDF"])

    for i in [fe, d, q, g]:
        i.cuda()
        i.apply(weights_init)

    trainer = Trainer(g, fe, d, q, config)
    trainer.train()