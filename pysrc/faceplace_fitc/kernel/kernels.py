import torch
from torch import nn


class KernelComposer():
    @staticmethod
    def Product(K1, K2):
        return K1 * K2  # elem-wise

    @staticmethod
    def Addition(K1, K2):
        return K1 + K2


class Kernel(nn.Module):
    def __init__(self):
        super(Kernel, self).__init__()

    def forward(self, X1, X2=None):
        raise NotImplementedError("Kernel is an abstract class")

    def dist(self, X1, X2=None, p=2):
        # if X2 not given, then set X2=X1, so we can make calls like k(X)
        if X2 is None:
            X2 = X1

        # if x in X1 and y in X2 are scalars, then expand X1 and X2 from shape [N] into shape [1, N]
        if X1.dim() == 1 and X2.dim() == 1:
            X1 = X1.unsqueeze(1)
            X2 = X2.unsqueeze(1)

        if X1.size(1) != X2.size(1):
            raise ValueError("Inputs must have same number of features")

        out = torch.zeros([X1.size(0), X2.size(0)])

        for i in range(X1.size(0)):
            for j in range(X2.size(0)):
                out[i][j] = torch.dist(X1[i], X2[j], p)
        return out


class RotationKernel(Kernel):
    def __init__(self):
        super(RotationKernel, self).__init__()
        self.beta = nn.Parameter(torch.randn(1).clamp(min=0.0001)) # inverse noise param to rotation kernel
        self.lengthscale = nn.Parameter(torch.randn(1).clamp(min=0.001)) # lengthscale squared param to rotation kernel

    def forward(self, X1, X2=None, diag=False):
        dist = self.dist(X1, X2)
        sineDistSqd = torch.sin(dist) * torch.sin(dist)
        K = self.beta * torch.exp((-2 * sineDistSqd) / (self.lengthscale * self.lengthscale))

        # TODO dont compute the entire thing if diag=True
        return K.diag() if diag else K


class LinearKernel(Kernel):
    def __init__(self):
        super(LinearKernel, self).__init__()

    def forward(self, X1, X2=None):
        if X2 is None:
            X2 = X1

        if X1.size(1) != X2.size(1):
            raise ValueError("Inputs must have same number of features")

        return torch.mm(X1, X2.t())
