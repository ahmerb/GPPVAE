import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import scipy as sp
import os
import pdb

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

    # V in K = V * V^T is the output of the net
    # trainable params are object and view covariances x0 and v0 ????
    def __init__(self, P, Q, p, q):
        super(Vmodel, self).__init__()

        # each feature vector (obj and view) is actually a word embedding
        # that is, say obj id is 17, this maps to some word embedding (=feature vector) e.g. [1.0, 0.3332432, 5.432, ...]
        # P is number of obj's, p is the size of the word embedding (=feature vector)
        # Q is number of views, q is the size of the word embedding (=feature vector)

        # the  obj i has its feature vector as row i in x0
        # the view i has its feature vector as row i in v0

        # linear object covariance
        # dimension PxM (where M=opt.xdim=64)
        self.x0 = nn.Parameter(torch.randn(P, p)) # trainable parameter

        # Use full rank view covariance
        # Q=q=9 (9 different face poses)
        self.v0 = nn.Parameter(torch.randn(Q, q)) # trainable parameter

        self._init_params()

        self.viewKernel = RotationKernel()
        self.objKernel = LinearKernel()

    def x(self):
        return self._normalize_rows(self.x0)

    def v(self):
        return self._normalize_rows(self.v0)

    # d is vector of object id's (dim Nx1)
    # w is vector of   view id's (dim Nx1)
    def forward(self, d, w):
        # embed

        # retrieve the obj feature vectors for the obj's in d
        # X is dim Nxp
        X = F.embedding(d, self.x())

        # retrieve the view feature vectors for the views in w
        # W is dim Nxq
        W = F.embedding(w, self.v())

        # compute K
        K = self.viewKernel(W) * self.objKernel(X) # elem-wise mult

        # # eigendecomposition
        # # evals (Nx2) is eigenvalues (e[:, 0] are real parts, e[:, 1] are imaginary parts)
        # # evecs (NxN) are eigenvectors
        # evals, evecs = torch.eig(K, eigenvectors=True)

        # # take the first (p*q) eigenvectors
        # # todo: eigenvalues not necessarily returned in order
        # p = X.size(1)
        # q = W.size(1)
        # N = X.size(0)
        # k = min(p*q, N) # can't have higher than full rank....
        # first_k_eigenvalues_sqrt = torch.diag(torch.sqrt(torch.abs(evals[:k, 0])))
        # first_k_eigenvectors = evecs[:k]

        # # print(k)
        # # print(evals)
        # # print(torch.sqrt(evals[:k, 0]))

        # V = first_k_eigenvectors.t().mm(first_k_eigenvalues_sqrt)
        # return V
        print('begin svd')
        U, S, V = torch.svd(K)
        print('end svd')
        p = X.size(1)
        q = W.size(1)
        N = X.size(0)
        k = min(p*q, N) # can't have higher than full rank....
        
        # take first k singular values
        Sprime = torch.diag(torch.sqrt((S[:k])))
        V = U.mm(Sprime)
        return V

    def _init_params(self):
        # make the first column all 1.0's
        self.x0.data[:, 0] = 1.0

        # 1) x0[:, 1:] gives matrix excluding first column,
        # so if x0.shape is (a,b), then x0[:, 1:].shape is (a,b-1).
        # 2) Create matrix of random numbers of size (a,b-1), each elem multiplied by 0.001
        # 3) Set this matrix to the right b-1 columns of x0 (i.e. excluding first column)
        self.x0.data[:, 1:] = 1e-3 * torch.randn(*self.x0[:, 1:].shape)

        # now, x is (a,b) shape matrix where first column is 1.0 and remaining elems
        # are all random numbers multiplied by 0.001

        # 1) create a matrix shaped like v0 of random numbers, multiply elems by 0.001
        # 2) add the identity matrix
        # 3) set this to v0
        self.v0.data[:] = torch.eye(*self.v0.shape) + 1e-3 * torch.randn(*self.v0.shape)

    def _normalize_rows(self, x):
        # i think this does following:
        #  multiplies x*x element-wise (sq each element),
        #  then sums along each row,
        #  then the [:, None] converts something like [sum1, sum2, ...]
        #     into [ [sum1], [sum2], ... ]
        diagonal = torch.einsum("ij,ij->i", [x, x])[:, None]

        # divides each element by the sq root of the sum of its row elem's squared
        # x[i][j] = x[i][j] / sum( x[i][j']^2 for x[i][j'] in x[i] )
        return x / torch.sqrt(diagonal)


if __name__ == "__main__":
    P = 5 # number uniq objs
    Q = 3 # number uniq views
    p = 3 # obj dim
    q = 5 # view dim

    vm = Vmodel(P, Q, p, q)

    # select d and w ids
    d = torch.randn([20]).clamp(min=1, max=P).long() # N=20
    w = torch.randn([20]).clamp(min=1, max=Q).long()

    # manually compute K
    X = F.embedding(d, vm.x())
    W = F.embedding(w, vm.v())
    K = vm.viewKernel(W) * vm.objKernel(X)

    # find output and compare to K
    V = vm(d, w)

    print('***\n')
    print(K - V.mm(V.t()))
    print('***\n')
    print(K)
    print('***\n')
    print(V)






# if __name__ == "__main__":

    # P = 4
    # Q = 4
    # p = 2
    # q = 2

    # #pdb.set_trace()

    # vm = Vmodel(P, Q, p, q)#.cuda()

    # # _d and _w
    # _d = sp.kron(sp.arange(P), sp.ones(2))
    # _w = sp.kron(sp.ones(2), sp.arange(Q))

    # # d and w
    # d = Variable(torch.Tensor(_d).long(), requires_grad=False)#.cuda()
    # w = Variable(torch.Tensor(_w).long(), requires_grad=False)#.cuda()

    # print(d.size())
    # print(w.size())
    # V = vm(d, w)
    # print(V.size())

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