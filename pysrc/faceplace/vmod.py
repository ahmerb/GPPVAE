import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy as sp
import os
import pdb


def normalize_rows(x):
    diagonal = torch.einsum("ij,ij->i", [x, x])[:, None]
    return x / torch.sqrt(diagonal)


class Vmodel(nn.Module):
    # P number of unique objects, denoted x, M dims. 
    # Q number of unique   views, denoted w, R dims.
    # N                  samples, denoted y, K dims.
    # N   latent representations, denoted z, L dims.
    #
    # A latent z_n of object p_n in view q_n is generated
    #  from obj feature vector x_{p_n} and view feature vector w_{q_n}.
    #
    # Called with: (in gppvae after setting up the VAE)
    # P = sp.unique(obj["train"]).shape[0]
    # Q = sp.unique(view["train"]).shape[0]
    # vm = Vmodel(P, Q, opt.xdim, Q).cuda() # xdim is rank of object linear covariance
    # gp = GP(n_rand_effs=1).to(device)
    # gp_params = nn.ParameterList()
    # gp_params.extend(vm.parameters())
    # gp_params.extend(gp.parameters())
    def __init__(self, P, Q, p, q):
        super(Vmodel, self).__init__()
        self.x0 = nn.Parameter(torch.randn(P, p))
        self.v0 = nn.Parameter(torch.randn(Q, q))
        self._init_params()

    def x(self):
        return normalize_rows(self.x0)

    def v(self):
        return normalize_rows(self.v0)

    def forward(self, d, w):
        # embed
        X = F.embedding(d, self.x())
        W = F.embedding(w, self.v())
        # multiply
        V = torch.einsum("ij,ik->ijk", [X, W])
        V = V.reshape([V.shape[0], -1])
        return V

    def _init_params(self):
        self.x0.data[:, 0] = 1.0
        self.x0.data[:, 1:] = 1e-3 * torch.randn(*self.x0[:, 1:].shape)
        self.v0.data[:] = torch.eye(*self.v0.shape) + 1e-3 * torch.randn(*self.v0.shape)


if __name__ == "__main__":

    P = 4
    Q = 4
    p = 2
    q = 2

    pdb.set_trace()

    vm = Vmodel(P, Q, p, q).cuda()

    # _d and _w
    _d = sp.kron(sp.arange(P), sp.ones(2))
    _w = sp.kron(sp.ones(2), sp.arange(Q))

    # d and w
    d = Variable(torch.Tensor(_d).long(), requires_grad=False).cuda()
    w = Variable(torch.Tensor(_w).long(), requires_grad=False).cuda()

    V = vm(d, w)
    pdb.set_trace()
