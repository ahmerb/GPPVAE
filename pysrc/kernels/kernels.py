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

    def dist(self, X1, X2=None, p=2, diag=False):
        # if X2 not given, then set X2=X1, so we can make calls like k(X)
        if X2 is None:
            X2 = X1

        # if x in X1 and y in X2 are scalars, then expand X1 and X2 from shape [N] into shape [1, N]
        if X1.dim() == 1 and X2.dim() == 1:
            X1 = X1.unsqueeze(1)
            X2 = X2.unsqueeze(1)

        if X1.size(1) != X2.size(1):
            raise ValueError("Inputs must have same number of features")

        if diag:
            if X1.size(0) != X2.size(0):
                raise ValueError("Inputs must have same shape with diag=True")
            N = X1.size(0)
            out = torch.zeros(N).to(X1.device)
            for i in range(N):
                out[i] = torch.dist(X1[i], X2[i], p)
            return out

        else:
            out = torch.zeros([X1.size(0), X2.size(0)]).to(X1.device)
            for i in range(X1.size(0)):
                for j in range(X2.size(0)):
                    out[i][j] = torch.dist(X1[i], X2[j], p)
            return out


class SEKernel(Kernel):
    def __init__(self, beta=None, lengthscale=None):
        super(SEKernel, self).__init__()
        if beta is None:
            self.beta = nn.Parameter(torch.randn(1).clamp(min=0.0001)) # inverse noise param to rotation kernel
        else:
            if type(beta) == float:
                self.beta = nn.Parameter(torch.tensor(beta))
            elif type(beta) == torch.Tensor and beta.dim() == 0:
                self.beta = nn.Parameter(beta)
            else:
                raise TypeError('Kernel hyperparameter beta should be of class float or torch.Tensor with .dim()=0')
        if lengthscale is None:
            self.lengthscale = nn.Parameter(torch.randn(1).clamp(min=0.0001)) # inverse noise param to rotation kernel
        else:
            if type(lengthscale) == float:
                self.lengthscale = nn.Parameter(torch.tensor(lengthscale))
            elif type(lengthscale) == torch.Tensor and lengthscale.dim() == 0:
                self.lengthscale = nn.Parameter(lengthscale)
            else:
                raise TypeError('Kernel hyperparameter lengthscale should be of class float or torch.Tensor with .dim()=0')

    def forward(self, X1, X2=None, diag=False):
        dist = self.dist(X1, X2, diag=diag)
        distSqd = dist * dist
        K = self.beta * torch.exp((-2 * distSqd) / (self.lengthscale * self.lengthscale))
        return K


class RotationKernel(Kernel):
    def __init__(self):
        super(RotationKernel, self).__init__()
        self.beta = nn.Parameter(torch.randn(1).clamp(min=0.0001)) # inverse noise param to rotation kernel
        self.lengthscale = nn.Parameter(torch.randn(1).clamp(min=0.001)) # lengthscale squared param to rotation kernel

    def forward(self, X1, X2=None, diag=False):
        dist = self.dist(X1, X2, diag=diag)
        sineDistSqd = torch.sin(dist) * torch.sin(dist)
        K = self.beta * torch.exp((-2 * sineDistSqd) / (self.lengthscale * self.lengthscale))
        return K


class LinearKernel(Kernel):
    def __init__(self):
        super(LinearKernel, self).__init__()

    def forward(self, X1, X2=None):
        if X2 is None:
            X2 = X1

        if X1.size(1) != X2.size(1):
            raise ValueError("Inputs must have same number of features")

        return torch.mm(X1, X2.t())
