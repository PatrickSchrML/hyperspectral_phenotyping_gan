import torch.nn as nn

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, input):
        return input.transpose(self.dim1, self.dim2)


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


def create_generator(n_input, NGF):
    main = nn.Sequential(
        # input is Z, going into a convolution
        nn.Linear(n_input, 1024),  # -> 1024
        nn.LeakyReLU(inplace=True),
        nn.Linear(1024, NGF * 128),  # -> 40 x 128
        nn.BatchNorm1d(NGF * 128),
        nn.LeakyReLU(inplace=True),
        View([-1, NGF, 128, 1]),  # -> 40, 128
        nn.ConvTranspose2d(NGF, NGF * 2, 3, 1, 1, bias=False),  # -> 80, 128
        nn.LeakyReLU(inplace=True),
        nn.BatchNorm2d(NGF * 2),
        nn.Conv2d(NGF * 2, NGF * 2, 3, 2, 1, bias=False),  # -> 80, 64
        nn.BatchNorm2d(NGF * 2),
        nn.LeakyReLU(inplace=True),
        nn.ConvTranspose2d(NGF * 2, NGF * 4, 3, 1, 1, bias=False),  # -> 160, 64
        nn.BatchNorm2d(NGF * 4),
        nn.LeakyReLU(inplace=True),
        Transpose(1, 2),
        nn.Conv2d(64, 1, 3, 1, 1, bias=False),  # -> 160, 1
        nn.BatchNorm2d(1),
        Transpose(1, 2),
        nn.Sigmoid(),
    )
    return main


class InfoGAN_Generator(nn.Module):
    def __init__(self, noise_dim=10, n_conti=2, n_discrete=1,
                 num_category=3, NGF=40):
        """
        InfoGAN Generator, have an additional input branch for latent codes.
        Architecture brought from DCGAN.
        """
        super(InfoGAN_Generator, self).__init__()
        self.n_conti = n_conti
        self.n_discrete = n_discrete
        self.num_category = num_category
        self.starting_nbfeatures = 128
        self.width = NGF

        # calculate input dimension
        n_input = noise_dim + n_conti + n_discrete * num_category

        self.fc, self.conv = create_generator_1d(n_input=n_input, NGF=self.width,
                                                 starting_nbfeatures=self.starting_nbfeatures)

    def forward(self, x):
        """
        Input the random noise plus latent codes to generate fake images.
        """
        x = self.fc(x)
        x = x.view(x.shape[0], x.shape[1] / self.width, x.shape[1] / self.starting_nbfeatures)
        x = self.conv(x)
        x = x.squeeze()
        return x