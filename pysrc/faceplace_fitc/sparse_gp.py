import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.distributions as dists
import math # to compute the constant log(2*pi)
import matplotlib.pyplot as pl

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

        self.jitter = 1e-6 # stablise Cholesky decompositions

    def _zero_mean_function(self, x):
        return x.new_zeros(x.shape) # creates zero tensor with same shape, dtype, etc...

    def _add_jitter(self, matrix):
        matrix.view(-1)[::(matrix.shape[0]) + 1] += self.jitter

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

        Kfu = self.kernel(self.X, self.Xu) # NxM
        Kuu = self.kernel(self.Xu, self.Xu) # MxM
        
        self._add_jitter(Kuu)
        Luu = torch.cholesky(Kuu, upper=False) # MxM
        W = torch.trtrs(Kfu.t(), Luu, upper=False)[0].t() # NxM
        # W.T = LuuInv @ Kuf
        # MxN     MxM    MxN

        # computes diagonal of Qff = Kfu @ KuuInv @ Kuf = W @ W.T
        Qffdiag = W.pow(2).sum(dim=-1) # NxN

        Kffdiag = self.kernel(self.X, self.X, diag=True) # NxN

        noise_diag = self.noise.expand(N) # NxN

        # The total covariance is
        # Σ = Qff + (Kff - Qff).diag() + noise*I = Qff + D

        D = Kffdiag - Qffdiag + noise_diag # NxN (represented by torch.Size([N]))

        # mean
        mu = self.mean_function(self.y) # Nx1 (represented by torch.Size([N]))

        return self.low_rank_log_likelihood(mu, W.t(), D) # W.t() is mxn

    def low_rank_log_likelihood(self, mu, W, D):
        """
        Find logp(y|mu,cov) where cov = W @ W.T + D

        :param  torch.Tensor mu: tensor with size([N]) that represents Nx1 mean vector
        :param  torch.Tensor  W: MxN matrix
        :param  torch.Tensor  D: tensor with size([N]) that represents an NxN diagonal matrix
        :returns: tensor with size([]) that gives logp(y|mu,cov)
        :rtype: torch.Tensor
        """
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
        W_Dinv = W / D # MxN
        K = W_Dinv.mm(W.t()) + torch.eye(M, M) # MxM

        # Compute cholesky decomposition K = L @ L.T
        # where L is lower triangular
        L = torch.cholesky(K, upper=False) # MxM

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
        logdetK = 2 * L.diag().log().sum() # scalar, size([])

        # b) log|D|
        #     = log(D_1 * D_2 * ... * D_N)
        #     = log(D_1) + ... + log(D_2)
        #     = \Sigma{log(D_i)}
        logdetD = D.log().sum() # scalar, size([]) 

        logdetCov = logdetK + logdetD # scalar, size([])

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
        # y.unsqueeze(1) converts is from size([N]) to size([N,1])
        W_Dinv_y = W_Dinv.mm(y.unsqueeze(1)) # Mx1
        
        # Linv_W_Dinv_y is the matrix M that solves eqn
        # LM = W_Dinv_y
        Linv_W_Dinv_y = torch.trtrs(W_Dinv_y, L, upper=False)[0] # Mx1

        mdist1 = (Linv_W_Dinv_y * Linv_W_Dinv_y).sum() # scalar, size([])

        # b) compute y.T @ inv(D) @ y
        mdist2 = ((y * y) / D).sum() # scalar, size([])

        # total mdist^2
        mdist_squared = mdist2 - mdist1 # scalar, size([])

        # Finally, compute the total logp(y|X,Xu)
        #   = -0.5log(2*pi) - 0.5log|Σ| - 0.5*MDIST(y,N(y|0, Σ))^2
        norm_const = math.log(2 * math.pi)
        logprob = -0.5 * (norm_const + logdetCov + mdist_squared) # scalar, size([])

        return logprob

    def predict_and_plot(self, Xnew, nsamples=3):
        N = Xnew.shape[0]
        mu, cov = self.posterior_predictive(Xnew)

        # L = stddev = sqrt(cov) = cholesky(cov)
        L = torch.cholesky(self._add_jitter(cov.contiguous()))

        # draw samples from posterior
        f_posterior = mu + L.mm(torch.normal(mu=torch.zeros(nsamples, N), stdev=torch.ones(N)))
        
        pl.plot(self.X, self.y, 'bs', ms=8)
        pl.plot(Xnew, f_posterior)
        pl.gca().fill_between(Xnew, mu-2*L.diag(), mu+2*L.diag(), color="#dddddd")
        pl.plot(Xnew, mu, 'r--', lw=2)
        pl.title('{} Samples from GP Posterior'.format(nsamples))
        pl.show()

# # Noiseless training data
# Xtrain = np.array([-4, -3, -2, -1, 1]).reshape(5,1)
# ytrain = np.sin(Xtrain)

# # Apply the kernel function to our training points
# K = kernel(Xtrain, Xtrain, param)
# L = np.linalg.cholesky(K + 0.00005*np.eye(len(Xtrain)))

# # Compute the mean at our test points.
# K_s = kernel(Xtrain, Xtest, param)
# Lk = np.linalg.solve(L, K_s)
# mu = np.dot(Lk.T, np.linalg.solve(L, ytrain)).reshape((n,))

# # Compute the standard deviation so we can plot it
# s2 = np.diag(K_ss) - np.sum(Lk**2, axis=0)
# stdv = np.sqrt(s2)
# # Draw samples from the posterior at our test points.
# L = np.linalg.cholesky(K_ss + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
# f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,3)))

# pl.plot(Xtrain, ytrain, 'bs', ms=8)
# pl.plot(Xtest, f_post)
# pl.gca().fill_between(Xtest.flat, mu-2*stdv, mu+2*stdv, color="#dddddd")
# pl.plot(Xtest, mu, 'r--', lw=2)
# pl.axis([-5, 5, -3, 3])
# pl.title('Three samples from the GP posterior')
# pl.show()
    
    # NOTE this is adapted from Pyro v0.21 pyro.contrib.gp.models.SparseGPRegression#forward
    def posterior_predictive(self, Xnew, full_cov=False, noiseless=True):
        r"""
        Computes the mean and covariance matrix (or variance) of Gaussian Process
        posterior on a test input data :math:`X_{new}`:

        .. math:: p(f^* \mid X_{new}, X, y, k, X_u, \epsilon) = \mathcal{N}(loc, cov).

        .. note:: The noise parameter ``noise`` (:math:`\epsilon`), the inducing-point
            parameter ``Xu``, together with kernel's parameters have been learned from
            a training procedure (MCMC or SVI).

        :param torch.Tensor Xnew: A input data for testing. Note that
            ``Xnew.shape[1:]`` must be the same as ``self.X.shape[1:]``.
        :param bool full_cov: A flag to decide if we want to predict full covariance
            matrix or just variance.
        :param bool noiseless: A flag to decide if we want to include noise in the
            prediction output or not.
        :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        self._check_Xnew_shape(Xnew)

        # W = inv(Luu) @ Kuf
        # Ws = inv(Luu) @ Kus
        # D as in self.model()
        # K = I + W @ inv(D) @ W.T = L @ L.T
        # S = inv[Kuu + Kuf @ inv(D) @ Kfu]
        #   = inv(Luu).T @ inv[I + inv(Luu)@ Kuf @ inv(D)@ Kfu @ inv(Luu).T] @ inv(Luu)
        #   = inv(Luu).T @ inv[I + W @ inv(D) @ W.T] @ inv(Luu)
        #   = inv(Luu).T @ inv(K) @ inv(Luu)
        #   = inv(Luu).T @ inv(L).T @ inv(L) @ inv(Luu)
        # loc = Ksu @ S @ Kuf @ inv(D) @ y = Ws.T @ inv(L).T @ inv(L) @ W @ inv(D) @ y
        # cov = Kss - Ksu @ inv(Kuu) @ Kus + Ksu @ S @ Kus
        #     = kss - Ksu @ inv(Kuu) @ Kus + Ws.T @ inv(L).T @ inv(L) @ Ws

        N = self.X.size(0)
        M = self.Xu.size(0)

        # TODO: cache these calculations to get faster inference

        Kuu = self.kernel(self.Xu).contiguous()
        Kuu.view(-1)[::M + 1] += self.jitter  # add jitter to the diagonal
        Luu = Kuu.cholesky()

        Kuf = self.kernel(self.Xu, self.X)

        W = Kuf.trtrs(Luu, upper=False)[0]
        D = self.noise.expand(N)
        Kffdiag = self.kernel(self.X, diag=True)
        Qffdiag = W.pow(2).sum(dim=0)
        D = D + Kffdiag - Qffdiag

        W_Dinv = W / D
        K = W_Dinv.matmul(W.t()).contiguous()
        K.view(-1)[::M + 1] += 1  # add identity matrix to K
        L = K.cholesky()

        # get y_residual and convert it into 2D tensor for packing
        y_residual = self.y - self.mean_function(self.X)
        y_2D = y_residual.reshape(-1, N).t()
        W_Dinv_y = W_Dinv.matmul(y_2D)

        # End caching ----------

        Kus = self.kernel(self.Xu, Xnew)
        Ws = Kus.trtrs(Luu, upper=False)[0]
        pack = torch.cat((W_Dinv_y, Ws), dim=1)
        Linv_pack = pack.trtrs(L, upper=False)[0]
        # unpack
        Linv_W_Dinv_y = Linv_pack[:, :W_Dinv_y.shape[1]]
        Linv_Ws = Linv_pack[:, W_Dinv_y.shape[1]:]

        C = Xnew.size(0)
        loc_shape = self.y.shape[:-1] + (C,)
        loc = Linv_W_Dinv_y.t().matmul(Linv_Ws).reshape(loc_shape)

        if full_cov:
            Kss = self.kernel(Xnew).contiguous()
            if not noiseless:
                Kss.view(-1)[::C + 1] += self.noise  # add noise to the diagonal
            Qss = Ws.t().matmul(Ws)
            cov = Kss - Qss + Linv_Ws.t().matmul(Linv_Ws)
        else:
            Kssdiag = self.kernel(Xnew, diag=True)
            if not noiseless:
                Kssdiag = Kssdiag + self.noise
            Qssdiag = Ws.pow(2).sum(dim=0)
            cov = Kssdiag - Qssdiag + Linv_Ws.pow(2).sum(dim=0)

        cov_shape = self.y.shape[:-1] + (C, C)
        cov = cov.expand(cov_shape)

        return loc + self.mean_function(Xnew), cov

    def _check_Xnew_shape(self, Xnew):
        """
        Checks the correction of the shape of new data.
        :param torch.Tensor Xnew: A input data for testing. Note that
            ``Xnew.shape[1:]`` must be the same as ``self.X.shape[1:]``.
        """
        if Xnew.dim() != self.X.dim():
            raise ValueError("Train data and test data should have the same "
                             "number of dimensions, but got {} and {}."
                             .format(self.X.dim(), Xnew.dim()))
        if self.X.shape[1:] != Xnew.shape[1:]:
            raise ValueError("Train data and test data should have the same "
                             "shape of features, but got {} and {}."
                             .format(self.X.shape[1:], Xnew.shape[1:]))

def testStuff():
    N = 50
    X = dists.Uniform(0.0, 5.0).sample(sample_shape=(N,))
    y = 0.5 * torch.sin(3*X) + dists.Normal(0.0, 0.2).sample(sample_shape=(N,))
    Xu = torch.arange(20.) / 4.0

    sgpr = SparseGPRegression(X, y, RotationKernel, Xu)

    optimizer = torch.optim.Adam(sgpr.parameters(), lr=0.005)

    n_steps = 100
    losses = []
    print("begin training")
    for i in range(n_steps):
        optimizer.zero_grad()
        loss = -sgpr()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print("epoch {}: loss={}".format(i, loss.item()))

    Xtest = torch.linspace(0.0, 5.0, 10)
    sgpr.predict_and_plot(Xtest)


if __name__ == "__main__":
    testStuff()





