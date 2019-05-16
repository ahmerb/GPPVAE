import torch
from torch import nn
import torch.nn.functional as F


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
        self.conv1 = nn.Conv2d(ni, no, kernel_size=ks, stride=1, padding=1)
        self.conv2 = nn.Conv2d(no, no, kernel_size=ks, stride=2, padding=1)

    def forward(self, x):
        x = f_act(self.conv1(x), act=self.act)
        x = f_act(self.conv2(x), act=self.act)
        return x


class Conv2dCellUp(nn.Module):
    def __init__(self, ni, no, ks=3, act1="elu", act2="elu"):
        super(Conv2dCellUp, self).__init__()
        self.act1 = act1
        self.act2 = act2
        self.conv1 = nn.Conv2d(ni, no, kernel_size=ks, stride=1, padding=1)
        self.conv2 = nn.Conv2d(no, no, kernel_size=ks, stride=1, padding=1)

    def forward(self, x):
        x = F.upsample(x, scale_factor=2)
        x = f_act(self.conv1(x), act=self.act1)
        x = f_act(self.conv2(x), act=self.act2)
        return x


class FaceVAE(nn.Module):
    # nf: in and out channels per conv2d layer (except first conv layer is 'colors' in channels and 'nf' out channels)
    def __init__(
        self, img_size=128, nf=32, zdim=256, steps=5, colors=3, act="elu", vy=1e-3, cvae=False
    ):

        super(FaceVAE, self).__init__()

        # if cvae=True, stack (sinw, cosw) then make CVAE
        self.cvae = cvae

        # store useful stuff
        self.img_size = img_size
        self.red_img_size = img_size // (2 ** steps)
        self.nf = nf # number of filters per conv layer
        self.size_flat = self.red_img_size ** 2 * nf
        self.K = img_size ** 2 * colors # number of pixels in image?
        ks = 3

        # define variance
        self.vy = nn.Parameter(torch.Tensor([vy]), requires_grad=False)

        n_channels = colors + 2 if cvae else colors # two extra channels for sinw's and cosw's

        # conv cells encoder
        self.econv = nn.ModuleList()
        cell = Conv2dCellDown(n_channels, nf, ks, act)
        self.econv += [cell]
        for i in range(steps - 1):
            cell = Conv2dCellDown(nf, nf, ks, act)
            self.econv += [cell]

        # conv cells decoder
        self.dconv = nn.ModuleList()
        cell = Conv2dCellUp(nf + 2, nf, ks, act1=act, act2=act) if cvae else Conv2dCellUp(nf, nf, ks, act1=act, act2=act)
        self.dconv += [cell]
        for i in range(steps - 2):
            cell = Conv2dCellUp(nf, nf, ks, act1=act, act2=act)
            self.dconv += [cell]
        cell = Conv2dCellUp(nf, colors, ks, act1=act, act2="linear")
        self.dconv += [cell]

        # dense layers
        self.dense_zm = nn.Linear(self.size_flat, zdim)
        self.dense_zs = nn.Linear(self.size_flat, zdim)
        self.dense_dec = nn.Linear(zdim + 2, self.size_flat) if cvae else nn.Linear(zdim, self.size_flat)

    def encode(self, x, w=None):
        if self.cvae:
            # add sin and cos of rotation angle as extra channels to image
            sin_w = torch.sin(w).reshape(-1, 1, 1, 1).expand(-1, 1, self.img_size, self.img_size)
            cos_w = torch.cos(w).reshape(-1, 1, 1, 1).expand(-1, 1, self.img_size, self.img_size)
            x = torch.cat((x, sin_w, cos_w), dim=1)

        for ic, cell in enumerate(self.econv):
            x = cell(x)
        x = x.view(-1, self.size_flat)
        zm = self.dense_zm(x)
        zs = F.softplus(self.dense_zs(x))
        return zm, zs

    def sample(self, x, eps):
        zm, zs = self.encode(x)
        z = zm + eps * zs
        return z

    def decode(self, x, w=None):
        if self.cvae:
            # add [sinw,cosw] to end of latent vector z
            x = torch.cat((x, torch.sin(w), torch.cos(w)), dim=1)

        x = self.dense_dec(x)
        x = x.view(-1, self.nf, self.red_img_size, self.red_img_size)

        if self.cvae:
            # add sin and cos of rotation angle as extra channels to image
            sin_w = torch.sin(w).reshape(-1, 1, 1, 1).expand(-1, 1, self.red_img_size, self.red_img_size)
            cos_w = torch.cos(w).reshape(-1, 1, 1, 1).expand(-1, 1, self.red_img_size, self.red_img_size)
            x = torch.cat((x, sin_w, cos_w), dim=1)

        for cell in self.dconv:
            x = cell(x)
        return x

    def nll(self, x, xr):
        mse = ((xr - x) ** 2).view(x.shape[0], self.K).mean(1)[:, None]
        nll = mse / (2 * self.vy)
        nll += 0.5 * torch.log(self.vy)
        return nll, mse

    def forward(self, x, eps, w=None):
        zm, zs = self.encode(x, w)
        z = zm + eps * zs
        xr = self.decode(z, w)
        nll, mse = self.nll(x, xr)
        kld = (
            -0.5 * (1 + 2 * torch.log(zs) - zm ** 2 - zs ** 2).sum(1)[:, None] / self.K
        )
        elbo = nll + kld
        return elbo, mse, nll, kld


if __name__ == "__main__":
    net = FaceVAE()
    print(net)
