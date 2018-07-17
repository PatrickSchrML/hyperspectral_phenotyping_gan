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
parser.add_argument('--semisup', action="store_true", help='True: semi-supervised | False: unsupervised')
parser.add_argument('--sup_ratio', default=1.0, required=False, help='ratio of semi-supervised labels')
parser.add_argument('--pretrained', help='use pretrained model for initialization', action="store_true")
parser.add_argument('--pretrained_epoch', default=0, help='epoch of pretrained model')
parser.set_defaults(pretrained=False)
parser.set_defaults(semisup=False)
opt = parser.parse_args()

sup_ratio = float(opt.sup_ratio)  # 0.1

config["OUTF"] += "_supervised"
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

OUTF_samples = config["OUTF"] + "/samples" + ("_ratio-{}".format(int(sup_ratio * 100)) if opt.semisup else "")
OUTF_model = config["OUTF"] + "/model" + ("_ratio-{}".format(int(sup_ratio * 100)) if opt.semisup else "")

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
        self.num_categories = config["NC"]
        self.size_total = self.dim_noise + self.dim_code_conti + self.num_categories

        self.batch_size = 40 * self.num_categories
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

    def train(self):

        real_x = torch.FloatTensor(self.batch_size, 160).cuda()
        real_y = torch.LongTensor(self.batch_size).cuda()
        label = torch.FloatTensor(self.batch_size).cuda()
        dis_c = torch.FloatTensor(self.batch_size, self.num_categories).cuda()
        con_c = torch.FloatTensor(self.batch_size, self.dim_code_conti).cuda()
        noise = torch.FloatTensor(self.batch_size, self.dim_noise).cuda()

        real_x = Variable(real_x)
        real_y = Variable(real_y)
        label = Variable(label, requires_grad=False)
        dis_c = Variable(dis_c)
        con_c = Variable(con_c)
        noise = Variable(noise)

        criterionD = nn.BCELoss().cuda()
        criterionQ_supervised = nn.CrossEntropyLoss().cuda()
        criterionQ_dis = nn.CrossEntropyLoss().cuda()
        criterionQ_con = log_gaussian()

        optimD = optim.Adam([{'params': self.FE.parameters()},
                             {'params': self.D.parameters()},
                             {'params': self.Q.parameters()}],
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

        c1 = np.hstack([c, np.zeros_like(c)])
        c2 = np.hstack([np.zeros_like(c), c])
        c3 = np.hstack([np.zeros_like(c), c])
        c_all = np.hstack([c, c])

        if self.dim_code_conti > 3:
            raise ValueError("Continuous code of dim > 3 not implemented")
        if self.dim_code_conti == 3:
            c1 = np.hstack([c1, np.zeros_like(c)])
            c2 = np.hstack([np.zeros_like(c), c2])
            c3 = np.hstack([c3, np.zeros_like(c)])
            c_all = np.hstack([c_all, c])

        idx = np.arange(self.num_categories).repeat(batch_size_eval // self.num_categories)
        one_hot = np.zeros((batch_size_eval, self.num_categories))
        one_hot[range(batch_size_eval), idx] = 1
        fix_noise = torch.Tensor(batch_size_eval, self.dim_noise).uniform_(-1, 1)

        num_batches = self.dataloader.size_train // self.batch_size

        n_epoch = 3000
        for epoch in range(n_epoch):
            for num_iters in range(num_batches):

                x, y, supervised_indices = self.dataloader.fetch_batch(onehot=False, num_classes=config["NC"])
                y = y.squeeze()

                # real part discriminator
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
                loss_d_real = criterionD(probs_real, label)

                # real part - supervised discrete code
                real_y.data.copy_(y)
                q_logits_real, _, _ = self.Q(fe_out1)

                loss_q_real = criterionQ_supervised(q_logits_real[supervised_indices, :],
                                                    real_y[supervised_indices]) * 0.5
                loss_real = loss_d_real + loss_q_real
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
                class_ = torch.LongTensor(idx).cuda()
                target = Variable(class_)
                dis_loss = criterionQ_dis(q_logits, target)  # * 0.5
                con_loss = criterionQ_con(con_c, q_mu, q_var) * 0.5  # 0.1

                G_loss = reconstruct_loss + dis_loss + con_loss
                G_loss.backward()
                optimG.step()

                if num_iters == num_batches - 1 and (epoch + 1) % 10 == 0:
                    print('Epoch/Iter:{0}/{1}, Dloss: {2}, Gloss: {3}'.format(
                        epoch + 1, num_iters, D_loss.data.cpu().numpy(),
                        G_loss.data.cpu().numpy())
                    )

                    noise.data.copy_(fix_noise)
                    dis_c.data.copy_(torch.Tensor(one_hot))

                    con_c.data.copy_(torch.from_numpy(c_all))
                    z = torch.cat([noise, dis_c, con_c], 1).view(-1, self.size_total)
                    x_save = self.G(z)
                    self.dataloader.save_image(x_save.data,
                                          '%s/call_fake_samples_epoch_%03d{}.png' % (OUTF_samples, epoch + 1))

                    #con_c.data.copy_(torch.from_numpy(c2))
                    #z = torch.cat([noise, dis_c, con_c], 1).view(-1, self.size_total)
                    #x_save = self.G(z)
                    #self.dataloader.save_image(x_save.data,
                    #                      '%s/c_2fake_samples_epoch_%03d{}.png' % (OUTF_samples, epoch + 1))

            if (epoch + 1) < 5000 and (epoch + 1) % 100 == 0:
                self.dataloader.save_model(self.G.state_dict(), '%s/netG_epoch_%d{}.pth' % (OUTF_model, epoch + 1))
                self.dataloader.save_model(self.D.state_dict(), '%s/netD_epoch_%d{}.pth' % (OUTF_model, epoch + 1))
                self.dataloader.save_model(self.FE.state_dict(), '%s/netFE_epoch_%d{}.pth' % (OUTF_model, epoch + 1))
                self.dataloader.save_model(self.Q.state_dict(), '%s/netQ_epoch_%d{}.pth' % (OUTF_model, epoch + 1))

            if (epoch + 1) >= 5000 and (epoch + 1) % 1000 == 0 or epoch == n_epoch - 1:
                self.dataloader.save_model(self.G.state_dict(), '%s/netG_epoch_%d{}.pth' % (OUTF_model, epoch + 1))
                self.dataloader.save_model(self.D.state_dict(), '%s/netD_epoch_%d{}.pth' % (OUTF_model, epoch + 1))
                self.dataloader.save_model(self.FE.state_dict(), '%s/netFE_epoch_%d{}.pth' % (OUTF_model, epoch + 1))
                self.dataloader.save_model(self.Q.state_dict(), '%s/netQ_epoch_%d{}.pth' % (OUTF_model, epoch + 1))

if __name__ == '__main__':

    save_config(os.path.join(OUTF_model, "config.p"), config)

    config["NETG"] = config["NETG"].format("_ratio-{}".format(int(sup_ratio * 100)) if opt.semisup else "",
                                           opt.pretrained_epoch)
    config["NETD"] = config["NETD"].format("_ratio-{}".format(int(sup_ratio * 100)) if opt.semisup else "",
                                           opt.pretrained_epoch)

    size_total = config["NOISE"] + config["N_CONTI"] + config["NC"]
    fe = FrontEnd()
    d = D()
    q = Q(dim_conti=config["N_CONTI"], dim_disc=config["NC"])
    g = G(size_total)

    for i in [fe, d, q, g]:
        i.cuda()
        i.apply(weights_init)

    trainer = Trainer(g, fe, d, q, config)
    trainer.train()