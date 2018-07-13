import torch.nn as nn
import torch
import torch.nn.functional as F


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        # print(input.shape)
        # print(input.view(*self.shape).shape)
        return input.view(*self.shape)


class InfoGAN_Discriminator(nn.Module):
    def __init__(self, n_conti=2, n_discrete=1,
                 num_category=10, featmap_dim=1024, NDF=160, NGF=40, classification=False):
        super(InfoGAN_Discriminator, self).__init__()
        self.n_conti = n_conti
        self.n_discrete = n_discrete
        self.num_category = num_category

        self.ndf = NDF
        self.ngf = NGF
        self.featmap_dim = featmap_dim

        self.classification = classification

        # Discriminator
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),  # 32, 160
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),  # 32, 80 - downsampling
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            ##nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),  # 32, 80
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),   # 32, 40 - downsampling
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            ##nn.MaxPool1d((2, 1)),
        )

        # output layer - prob(real) and auxiliary distributions Q(c_j|x)
        n_output = 1 + n_conti + n_discrete * num_category
        self.classifier = nn.Sequential(
            nn.Linear(32 * NDF / 4, self.featmap_dim),
            nn.Linear(self.featmap_dim, n_output),
        )

    def forward(self, x):
        """
        Output the probability of being in real dataset
        plus the conditional distributions of latent codes.
        """
        x = x.view([x.shape[0], 1, x.shape[1]])
        x = self.features(x)
        x = x.view([x.shape[0], x.shape[1] * x.shape[2]])
        x = self.classifier(x)
        # output layer

        x[:, 0] = F.sigmoid(x[:, 0].clone())
        labels = None
        for j in range(self.n_discrete):
            start = 1 + self.n_conti + j * self.num_category
            end = start + self.num_category
            labels = F.softmax(x[:, start:end].clone(), dim=1)
            x[:, start:end] = labels

        res = labels if self.classification else x
        return res
