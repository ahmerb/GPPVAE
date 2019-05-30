import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy as sp


class Vmodel(nn.Module):
    def __init__(self):
        super(Vmodel, self).__init__()

    def forward(self, d, w):
        # multiply

        # V_ijk the product of scalars X_ij and W_ik.
        # That is, V_ijk = X_ij * W_ik
        # That is, for ith datapoint, V_ijk is that datapoint's
        #   jth obj-feature-vector element multiplied by its kth
        #   view-feature-vector element.
        # V is shape (N,p,q)
        V = torch.einsum("ij,ik->ijk", [d, w])

        # Reshape V to be (N, p*q)
        # Each row i corresponds to a datapoint
        V = V.reshape([V.shape[0], -1])

        # We have computed the formula in the footnote of the paper.
        # that is, each column of V is the jth column of X multiplied by the kth column of W.

        # So, we have used linear kernels for both object and view kernels

        # We have achieved computing in linear time (i think)

        # V . V^T + I gives an NxN matrix (the approx covariance matrix??)
        return V

if __name__ == "__main__":

    P = 4
    Q = 4
    p = 2
    q = 2

    #pdb.set_trace()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vm = Vmodel(P, Q, p, q).to(device)

    # _d and _w
    _d = sp.kron(sp.arange(P), sp.ones(2))
    _w = sp.kron(sp.ones(2), sp.arange(Q))

    # d and w
    d = Variable(torch.Tensor(_d).long(), requires_grad=False).to(device)
    w = Variable(torch.Tensor(_w).long(), requires_grad=False).to(device)

    print(d.size())
    print(w.size())
    V = vm(d, w)
    print(V.size())

    #pdb.set_trace()

# # test what the einsum("ij,ik->ijk", [X,W]) does
# N = 5 # num samples
# p = 7 # obj feature vector dim
# q = 4 # view feature vector dim

# X = torch.randn(N, p)
# W = torch.randn(N, p, q)

# V2 = torch.randn(N, p, q)

# for i in range(N):
#     for j in range(p):
#         for k in range(q):
#             V2[i][j][k] = X[i][j] * W[i][k]
