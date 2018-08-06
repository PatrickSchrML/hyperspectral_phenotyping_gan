import torch.nn as nn


class FrontEnd(nn.Module):
    ''' front end part of discriminator and Q'''

    def __init__(self):
        super(FrontEnd, self).__init__()

        self.main = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(32, 64, kernel_size=5, stride=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(64, 128, kernel_size=5, stride=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv1d(128, 1024, kernel_size=8, bias=False),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )
        """
        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 1024, 7, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
        )
        """

    def forward(self, x):
        x = x.view([x.shape[0], 1, x.shape[1]])
        output = self.main(x)
        return output


class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()

        self.main = nn.Sequential(
            nn.Conv1d(1024, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x).view(-1, 1)
        return output


class Q(nn.Module):
    def __init__(self, dim_disc=3, dim_conti=2):
        super(Q, self).__init__()

        self.dim_disc = dim_disc
        self.dim_conti = dim_conti

        self.conv = nn.Conv1d(1024, 128, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(128)
        self.lReLU = nn.LeakyReLU(0.1, inplace=True)
        self.conv_disc = nn.Conv1d(128, dim_disc, 1)
        self.conv_mu = nn.Conv1d(128, dim_conti, 1)
        self.conv_var = nn.Conv1d(128, dim_conti, 1)

    def forward(self, x):
        y = self.conv(x)

        if self.dim_disc == 0:
            disc_logits = None
        else:
            disc_logits = self.conv_disc(y).squeeze()
        mu = self.conv_mu(y).squeeze()
        var = self.conv_var(y).squeeze().exp()

        return disc_logits, mu, var


def create_generator_1d(n_input, NGF, starting_nbfeatures=128):
    fc = nn.Sequential(
        nn.Linear(n_input, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, NGF * starting_nbfeatures),
        nn.BatchNorm1d(NGF * starting_nbfeatures),
        nn.ReLU(inplace=True),
    )
    conv = nn.Sequential(
        nn.ConvTranspose1d(starting_nbfeatures, starting_nbfeatures,
                           kernel_size=4, stride=2, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(starting_nbfeatures),
        nn.Conv1d(starting_nbfeatures, starting_nbfeatures // 2,
                  kernel_size=5, stride=1, padding=2, bias=False),
        nn.BatchNorm1d(starting_nbfeatures // 2),
        nn.ReLU(inplace=True),
        nn.ConvTranspose1d(starting_nbfeatures // 2, starting_nbfeatures // 2,
                           kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm1d(starting_nbfeatures // 2),
        nn.ReLU(inplace=True),
        nn.Conv1d(starting_nbfeatures // 2, 1, kernel_size=5, stride=1, padding=2, bias=False),
        nn.BatchNorm1d(1),
        nn.Sigmoid(),
    )
    return fc, conv


class G_with_fc(nn.Module):
    def __init__(self, dim_noise, dim_output):
        super(G_with_fc, self).__init__()

        self.starting_nbfeatures = 128
        self.width = dim_output // 4

        self.fc, self.conv = create_generator_1d(n_input=dim_noise, NGF=self.width,
                                                 starting_nbfeatures=self.starting_nbfeatures)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], x.shape[1] / self.width, x.shape[1] / self.starting_nbfeatures)
        x = self.conv(x)
        x = x.squeeze()
        return x


"""
def create_generator_1d_new(n_input, NGF, starting_nbfeatures=128):
    fc = nn.Sequential(
        nn.Linear(n_input, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, NGF * starting_nbfeatures),
        nn.BatchNorm1d(NGF * starting_nbfeatures),
        nn.ReLU(inplace=True),
    )
    conv = nn.Sequential(
        nn.ConvTranspose1d(starting_nbfeatures, starting_nbfeatures,
                           kernel_size=4, stride=2, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(starting_nbfeatures),
        nn.Conv1d(starting_nbfeatures, starting_nbfeatures // 2,
                  kernel_size=5, stride=1, padding=2, bias=False),
        nn.BatchNorm1d(starting_nbfeatures // 2),
        nn.ReLU(inplace=True),
        nn.ConvTranspose1d(starting_nbfeatures // 2, starting_nbfeatures // 2,
                           kernel_size=4, stride=2, padding=1, bias=False),
        nn.BatchNorm1d(starting_nbfeatures // 2),
        nn.ReLU(inplace=True),
        nn.ConvTranspose1d(starting_nbfeatures // 2, 1, kernel_size=3, stride=1, padding=1, bias=False),
        nn.BatchNorm1d(1),
        nn.Sigmoid(),
    )
    return fc, conv

class G_with_fc_new(nn.Module):
    def __init__(self, dim_noise):
        super(G_with_fc_new, self).__init__()

        self.starting_nbfeatures = 128
        self.width = 40

        self.fc, self.conv = create_generator_1d_new(n_input=dim_noise, NGF=self.width,
                                                 starting_nbfeatures=self.starting_nbfeatures)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.shape[0], x.shape[1] / self.width, x.shape[1] / self.starting_nbfeatures)
        x = self.conv(x)
        x = x.squeeze()
        return x
"""


class G_without_fc(nn.Module):
    def __init__(self, dim_noise):
        super(G_without_fc, self).__init__()

        self.starting_nbfeatures = 128
        self.width = 40

        self.main = nn.Sequential(
            nn.ConvTranspose1d(1, self.width, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm1d(self.width),
            nn.ReLU(True),
            nn.ConvTranspose1d(self.width, self.starting_nbfeatures, kernel_size=4, stride=2, bias=False),
            nn.BatchNorm1d(self.starting_nbfeatures),
            nn.ReLU(True),
            nn.ConvTranspose1d(self.starting_nbfeatures, self.starting_nbfeatures, kernel_size=5, stride=2, bias=False),
            nn.BatchNorm1d(self.starting_nbfeatures),
            nn.ReLU(True),
            nn.ConvTranspose1d(self.starting_nbfeatures, self.starting_nbfeatures // 2,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.starting_nbfeatures // 2),
            nn.ConvTranspose1d(self.starting_nbfeatures // 2, self.starting_nbfeatures // 2,
                               kernel_size=5, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(self.starting_nbfeatures // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(self.starting_nbfeatures // 2, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1])
        x = self.main(x)
        x = x.squeeze()
        return x


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)