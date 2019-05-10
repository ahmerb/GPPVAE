import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def f_act(x, act="elu"):
    if act == "relu":
        return F.relu(x)
    elif act == "elu":
        return F.elu(x)
    elif act == "linear":
        return x
    else:
        return None


class Conv2dCellDown(nn.Module):
    def __init__(self, ni, no, ks=3, act="elu"):
        super(Conv2dCellDown, self).__init__()
        self.act = act
        self.conv1 = nn.Conv2d(ni, no, kernel_size=ks, stride=2, padding=1)
        # self.conv2 = nn.Conv2d(no, no, kernel_size=ks, stride=2, padding=1)

    def forward(self, x):
        # print(" x0 is nan?=", torch.isnan(x).sum())
        # print("  w is nan?=", torch.isnan(self.conv1.weight).sum())
        x = self.conv1(x)
        # print(" x1 is nan?=", torch.isnan(x).sum())
        x = f_act(x, act=self.act)
        # print(" x2 is nan?=", torch.isnan(x).sum())
        # x = f_act(self.conv2(x), act=self.act)
        return x


class Conv2dCellUp(nn.Module):
    def __init__(self, ni, no, ks=3, act1="elu", act2="elu"):
        super(Conv2dCellUp, self).__init__()
        self.act1 = act1
        # self.act2 = act2
        self.conv1 = nn.Conv2d(ni, no, kernel_size=ks, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(no, no, kernel_size=ks, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        x = f_act(self.conv1(x), act=self.act1)
        # x = f_act(self.conv2(x), act=self.act2)
        return x


class RotatedMnistVAE(nn.Module):
    # nf: in and out channels per conv2d layer (except first conv layer is 'colors' in channels and 'nf' out channels)
    def __init__(
        self, img_size=28, nf=8, zdim=16, steps=3, colors=1, act="elu", vy=1e-3
    ):

        super(RotatedMnistVAE, self).__init__()

        # store useful stuff

        # output size on last conv layer = (nf x red_img_size x red_img_size)
        self.red_img_size = math.ceil(img_size / (2 ** steps))
        self.nf = nf # number of filters per conv layer
        self.size_flat = self.red_img_size ** 2 * nf # size of fully connect layer will be size_flat -> zdim
        self.K = img_size ** 2 * colors # number of pixels in image
        ks = 3 # kernel size will bs (ks x ks)

        # define variance
        self.vy = nn.Parameter(torch.Tensor([vy]), requires_grad=False)

        # conv cells encoder
        self.econv = nn.ModuleList()
        cell = Conv2dCellDown(colors, nf, ks, act)
        self.econv += [cell]
        for i in range(steps - 1):
            cell = Conv2dCellDown(nf, nf, ks, act)
            self.econv += [cell]

        # conv cells decoder
        self.dconv = nn.ModuleList()
        for i in range(steps - 1):
            cell = Conv2dCellUp(nf, nf, ks, act1=act, act2=act)
            self.dconv += [cell]
        cell = Conv2dCellUp(nf, colors, ks, act1=act, act2="linear")
        self.dconv += [cell]

        # dense layers
        self.dense_zm = nn.Linear(self.size_flat, zdim)
        self.dense_zs = nn.Linear(self.size_flat, zdim)
        self.dense_dec = nn.Linear(zdim, self.size_flat)

    def encode(self, x):
        for ic, cell in enumerate(self.econv):
            x = cell(x)
        x = x.view(-1, self.size_flat)
        # print(" x3 is nan?=", torch.isnan(x).sum())
        zm = self.dense_zm(x)
        # print(" zm is nan?=", torch.isnan(zm).sum())
        zs = F.softplus(self.dense_zs(x)) # softplus presumably because you can't have negative covariance values
        # print(" zs is nan?=", torch.isnan(zs).sum())
        return zm, zs

    def sample(self, x, eps):
        zm, zs = self.encode(x)
        z = zm + eps * zs
        return z

    def decode(self, x):
        x = self.dense_dec(x)
        x = x.view(-1, self.nf, self.red_img_size, self.red_img_size)
        for cell in self.dconv:
            x = cell(x)
        # print(" xr is nan? =", torch.isnan(x).sum())
        x = F.relu(x)
        return x

    def nll(self, x, xr):
        # print("vy=", self.vy)
        # print("x  is nan? =", torch.isnan(x).sum())
        # print("((xr - x) ** 2).sum()=", ((xr - x) ** 2).sum())
        mse = ((xr - x) ** 2).view(x.shape[0], self.K).mean(1)[:, None]
        nll = mse / (2 * self.vy)
        nll += 0.5 * torch.log(self.vy)
        return nll, mse

    def forward(self, x, eps):
        zm, zs = self.encode(x)
        z = zm + eps * zs
        # print("eps is nan? =", torch.isnan(eps).sum())
        # print("  z is nan? =", torch.isnan(z).sum())
        xr = self.decode(z)
        nll, mse = self.nll(x, xr)
        kld = (
            -0.5 * (1 + 2 * torch.log(zs) - zm ** 2 - zs ** 2).sum(1)[:, None] / self.K
        )
        elbo = nll + kld
        return elbo, mse, nll, kld


if __name__ == "__main__":
    net = RotatedMnistVAE()
    print(net)
