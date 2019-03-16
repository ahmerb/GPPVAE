import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.distributions as dists
import math # to compute the constant log(2*pi)

from kernels import RotationKernel

class SparseGPRegression(nn.Module):
    def __init__(self, X, y, kernel, Xu, mean_function=None, noise=1.):
        super(SparseGPRegression, self).__init__()
        self.X = X
        self.y = y
        self.Xu = Parameter(Xu)
        self.kernel = kernel()
        self.mean_function = mean_function if mean_function is not None else self._zero_mean_function
        self.noise = Parameter(Xu.new_tensor(noise))

    def _zero_mean_function(self, x):
        return x.new_zeros(x.shape) # creates zero tensor with same shape, dtype, etc...

    def forward(self, predictive=False):
        if predictive:
            raise NotImplementedError()
        else:
            return self.forward_train()

    def forward_train(self):
        """Computes logp(y|X,Xu)"""

        # number of true data points
        N = self.X.size(0)

        # number of inducing inputs
        M = self.Xu.size(0)

        Kfu = self.kernel(self.X, self.Xu)
        Kuu = self.kernel(self.Xu, self.Xu)
        
        Luu = Kuu.cholesky(upper=False)
        W = torch.trtrs(Kfu.t(), Luu, upper=False)[0].t()

        # computes diagonal of Qff = Kfu @ KuuInv @ Kuf = W @ W.T
        Qffdiag = W.pow(2).sum(dim=-1) 

        Kffdiag = self.kernel(self.X, self.X, diag=True)

        noise_diag = self.noise.expand(N)

        # The total covariance is
        # Σ = Qff + (Kff - Qff).diag() + noise*I = Qff + D

        D = Kffdiag - Qffdiag + noise_diag

        # mean
        mu = self.mean_function(self.X)

        return self.low_rank_log_likelihood(mu, W.t(), D) # W.t() is mxn

    def low_rank_log_likelihood(self, mu, W, D):
        """find logp(y|0,cov) where cov = W @ W.T + D"""
        M = W.shape[0]
        N = W.shape[1]
        y = self.y - mu # Do this to compute the Mahalanobis distance correctly

        # Now, we need to find the logp(y|X,Xu) given the mean and Σ
        # We calculate it in terms of W and D
        # logp(y|X,Xu)
        #  = log N(y|0, Σ)
        #  = -log((2*pi)^(N/2) * |Σ|^(1/2)) - 0.5 * y.T * Σinv * y
        #  = -0.5log(2*pi) - 0.5log|Σ| - 0.5*MDIST(y,N(y|0, Σ))^2
        # where MDIST is the Mahalanobis distance between point y and distribution N(y|0, Σ),
        #  which is the likelihood we're computing.

        # 1) We need to compute the logdet term, log|Σ|

        # First, so we can apply matrix-det lemma and Woodbury identity,
        # Expand Σ as follows
        #    Σ =  W @ W.T + D
        #      = (W @ Dinv @ W.T + I) @ D  (divide by D)
        #      := K @ D
        W_Dinv = W / D
        K = W_Dinv.mm(W.t()) + torch.eye(M, M)

        # Compute cholesky decomposition K = L @ L.T
        # where L is lower triangular
        L = torch.cholesky(K, upper=False)

        # Note that
        # log|Σ|
        #   = log|K@D|
        #   = log|K||D|
        #   = log|K| + log|D|

        # Now we simplify a) log|L @ L.T|, and b) log|D|
        
        # a) log|L @ L.T|
        #     = log|L||L.T|
        #     = log|L||L|
        #     = log(L_11*...*L_NN * L_11*...*L_NN) (L is lower triangular, so det's are on diagonal)
        #     = log(L_11*L_11 * L_22*L_22 * ... * L_NN*L_NN)
        #     = log(L_11^2) + ... + log(L_NN^2)
        #     = 2log(L_11) + ... + 2log(L_NN)
        #     = 2 * \Sigma{log(L_ii)}
        logdetK = 2 * L.diag().log().sum()

        # b) log|D|
        #     = log(D_1 * D_2 * ... * D_N)
        #     = log(D_1) + ... + log(D_2)
        #     = \Sigma{log(D_i)}
        logdetD = D.log().sum()

        logdetCov = logdetK + logdetD

        # 2) Now we compute the Mahalanobis distance MDIST(y,N(y|0, Σ)) = 0.5 * y.T * inv(Σ) * y
        # First, apply the Woodbury identity to inv(Σ) gives:
        # inv(Σ)
        #   = inv(W.T @ W + D)
        #   = inv(D) - inv(D) @ W.T() @ inv(I + W @ inv(D) @ W.T) @ W @ inv(D)
        #   = inv(D) - inv(D) @ W.T() @ inv(K) @ W @ inv(D)
        # Now sub inv(Σ) into y.T @ inv(Σ) @ y and expand
        #   = ....... (see my notes for derivation)
        #   = y.T @ inv(D) @ y - Linv_W_Dinv_y.T @ Linv_W_Dinv_y
        #
        # Note how inv(D) is trivial as D is diagonal

        # a) compute Linv_W_Dinv_y.T @ Linv_W_Dinv_y
        W_Dinv_y = W_Dinv.mm(y)
        
        # Linv_W_Dinv_y is the matrix M that solves eqn
        # LM = W_Dinv_y
        Linv_W_Dinv_y = torch.trtrs(W_Dinv_y, L, upper=False)[0]

        mdist1 = (Linv_W_Dinv_y * Linv_W_Dinv_y).sum(-1)

        # b) compute y.T @ inv(D) @ y
        mdist2 = ((y * y) / D)

        # total mdist^2
        mdist_squared = mdist1 - mdist2

        # Finally, compute the total logp(y|X,Xu)
        #   = -0.5log(2*pi) - 0.5log|Σ| - 0.5*MDIST(y,N(y|0, Σ))^2
        norm_const = math.log(2 * math.pi)
        logprob = -0.5 * (norm_const + logdetCov + mdist_squared)

        return logprob


if __name__ == "__main__":
    N = 1000
    X = dists.Uniform(0.0, 5.0).sample(sample_shape=(N,))
    y = 0.5 * torch.sin(3*X) + dists.Normal(0.0, 0.2).sample(sample_shape=(N,))
    Xu = torch.arange(20.) / 4.0

    sgpr = SparseGPRegression(X, y, RotationKernel, Xu)

    optimizer = torch.optim.Adam(sgpr.parameters(), lr=0.005)

    n_steps = 100
    losses = []
    for i in range(n_steps):
        optimizer.zero_grad()
        loss = -sgpr()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(losses)





