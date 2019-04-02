import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.distributions as dists
import math  # to compute the constant log(2*pi)
import matplotlib
import matplotlib.pyplot as pl
import numpy as np

from kernels import RotationKernel

matplotlib.use('Qt5Agg')


class SparseGPRegression(nn.Module):
    def __init__(self, X, y, kernel, Xu, mean_function=None, noise=0.5):
        super(SparseGPRegression, self).__init__()
        self.X = X
        self.train_points = (X,)
        self.y = y
        self.Xu = Parameter(Xu)
        self.kernel = kernel()
        self.mean_function = mean_function if mean_function is not None else self._zero_mean_function
        self.noise = Parameter(Xu.new_tensor(noise))
        self.jitter = 1e-3  # stablise Cholesky decompositions

    def Kfu(self):
        return self.kernel(self.X, self.Xu)  # NxM

    def Kuu(self):
        return self.kernel(self.Xu)  # MxM

    def Kffdiag(self):
        return self.kernel(self.X, self.X, diag=True)  # NxN

    def Kus(self, test_points):
        Xnew = test_points
        return self.kernel(self.Xu, Xnew)

    def Kss(self, test_points):
        Xnew = test_points
        return self.kernel(Xnew)

    def Kssdiag(self, test_points):
        Xnew = test_points
        return self.kernel(Xnew, diag=True)

    def forward(self):
        """Computes logp(y|X,Xu)"""

        # number of true data points
        N = self.X.size(0)

        # number of inducing inputs
        M = self.Xu.size(0)  # noqa: F841

        Kfu = self.Kfu()
        Kuu = self.Kuu()
        Kffdiag = self.Kffdiag()

        Luu = self._cholesky(Kuu)  # MxM
        W = torch.trtrs(Kfu.t(), Luu, upper=False)[0].t()  # NxM
        # W.T = LuuInv @ Kuf
        # MxN     MxM    MxN

        # computes diagonal of Qff = Kfu @ KuuInv @ Kuf = W @ W.T
        Qffdiag = W.pow(2).sum(dim=-1)  # NxN

        noise_diag = self.noise.expand(N)  # NxN

        # The total covariance is
        # Σ = Qff + (Kff - Qff).diag() + noise*I = Qff + D

        D = Kffdiag - Qffdiag + noise_diag  # NxN (represented by torch.Size([N]))

        # mean
        mu = self.mean_function(*self.train_points)  # NxL (L is Y dim)

        return self.low_rank_log_likelihood(mu, W.t(), D)  # W.t() is MxN

    def low_rank_log_likelihood(self, mu, W, D):
        """
        Find logp(y|mu,cov) where cov = W @ W.T + D

        :param  torch.Tensor mu: NxL mean vector
        :param  torch.Tensor  W: MxN matrix
        :param  torch.Tensor  D: tensor with size([N]) that represents an NxN diagonal matrix
        :returns: tensor with size([]) that gives the value logp(y|mu,cov)
        :rtype: torch.Tensor
        """
        M = W.shape[0]
        N = W.shape[1]  # noqa: F841
        y = self.y - mu  # Do this to compute the Mahalanobis distance correctly

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
        W_Dinv = W / D  # MxN
        K = W_Dinv.mm(W.t()) + torch.eye(M, M)  # MxM

        # Compute cholesky decomposition K = L @ L.T
        # where L is lower triangular
        L = self._cholesky(K, upper=False)  # MxM

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
        logdetK = 2 * L.diag().log().sum()  # scalar, size([])

        # b) log|D|
        #     = log(D_1 * D_2 * ... * D_N)
        #     = log(D_1) + ... + log(D_2)
        #     = \Sigma{log(D_i)}
        logdetD = D.log().sum()  # scalar, size([])

        logdetCov = logdetK + logdetD  # scalar, size([])

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

        # XXX the Mahalanobis distance in now LxL term?????
        # (computations may now be wrong for y.dim()=2 instead of y.dim()=1)

        # a) compute Linv_W_Dinv_y.T @ Linv_W_Dinv_y

        if y.dim() == 1:
            # y.unsqueeze(1) converts is from size([N]) to size([N,1])
            y = y.unsqueeze(1)  # y is now NxL, with L=1

        W_Dinv_y = W_Dinv.mm(y)  # MxL

        # Linv_W_Dinv_y is the matrix M that solves eqn
        # LM = W_Dinv_y
        Linv_W_Dinv_y = torch.trtrs(W_Dinv_y, L, upper=False)[0]  # MxL
        if y.dim() == 2:
            Linv_W_Dinv_y = Linv_W_Dinv_y.t()

        mdist1 = (Linv_W_Dinv_y * Linv_W_Dinv_y).sum(-1)  # scalar, size([])

        # b) compute y.T @ inv(D) @ y
        if y.dim() == 2:
            y = y.t()
        mdist2 = ((y * y) / D).sum(-1)  # scalar, size([])

        # total mdist^2
        mdist_squared = mdist2 - mdist1  # scalar, size([])

        # Finally, compute the total logp(y|X,Xu)
        #   = -0.5log(2*pi) - 0.5log|Σ| - 0.5*MDIST(y,N(y|0, Σ))^2
        norm_const = math.log(2 * math.pi)
        logprob = -0.5 * (norm_const + logdetCov + mdist_squared)  # scalar, size([])

        return logprob.sum()  # if L logprobs, then sum them

    def low_rank_log_likelihood_pyro_implementation(self, mu, W, D):
        y = self.y - mu
        y = y.t()

        W_Dinv = W / D  # divides each row by D
        M = W.shape[0]
        Id = torch.eye(M, M, out=W.new_empty(M, M))
        K = Id + W_Dinv.matmul(W.t())
        L = self._cholesky(K)
        if y.dim() == 1:
            W_Dinv_y = W_Dinv.matmul(y)
        elif y.dim() == 2:
            W_Dinv_y = W_Dinv.matmul(y.t())  # in pyro implementation, y is LxN, not NxL
        else:
            raise NotImplementedError("SparseMultivariateNormal distribution does not support "
                                      "computing log_prob for a tensor with more than 2 dimensionals.")

        Linv_W_Dinv_y = torch.trtrs(W_Dinv_y, L, upper=False)[0]
        if y.dim() == 2:
            Linv_W_Dinv_y = Linv_W_Dinv_y.t()

        logdet = 2 * L.diag().log().sum() + D.log().sum()

        # sum(-1) sums along innermost dimension, i.e. sum's each arr[i][*], i.e. along each row
        mahalanobis1 = (y * y / D).sum(-1)
        mahalanobis2 = (Linv_W_Dinv_y * Linv_W_Dinv_y).sum(-1)
        mahalanobis_squared = mahalanobis1 - mahalanobis2  # + trace_term

        norm_const = math.log(2 * math.pi)
        logprob = -0.5 * (norm_const + logdet + mahalanobis_squared)
        return logprob.sum()

    def predict_and_plot(self, Xnew, nsamples=3):
        N = Xnew.shape[0]
        mu, cov = self.posterior_predictive(Xnew, full_cov=True)

        # L = stddev = sqrt(cov) = cholesky(cov)
        L = self._cholesky(cov)

        # draw samples from posterior
        f_posterior = (mu + L.mm(torch.normal(mean=torch.zeros(nsamples, N), std=torch.ones(nsamples, N)).t()).t()).t()

        # convert to numpy
        numpy = {
            'X': self.X.detach().numpy(),
            'Xnew': Xnew.detach().numpy(),
            'mu': mu.detach().numpy(),
            'y': self.y.detach().numpy(),
            'f_posterior': f_posterior.detach().numpy(),
            'Ldiag': L.detach().diag().numpy(),
            'Xu': self.Xu.detach().numpy()
        }

        pl.plot(numpy['X'], numpy['y'], 'bs', ms=8)
        pl.plot(numpy['Xnew'], numpy['f_posterior'])
        pl.gca().fill_between(numpy['Xnew'],
                              numpy['mu'] - 2 * numpy['Ldiag'],
                              numpy['mu'] + 2 * numpy['Ldiag'],
                              color="#dddddd")
        pl.plot(numpy['Xnew'], numpy['mu'], 'r-', lw=2)
        pl.plot(numpy['Xu'], np.zeros(numpy['Xu'].shape[-1]) + pl.ylim()[0], 'r^')
        pl.title('{} Samples from GP Posterior'.format(nsamples))
        pl.show()

    def _get_num_test_points(self, test_points):
        return test_points.size(0)

    # in dual input case, test_points is tuple (Xnew, Wnew)
    # NOTE this is from Pyro v0.21 pyro.contrib.gp.models.SparseGPRegression#forward
    def posterior_predictive(self, test_points, full_cov=False, noiseless=True):
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
        self._check_test_points_shape(test_points)

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

        # TODO: cache these calculations to get faster inference (W_Dinv_y is not computed from test points)

        Kuu = self.Kuu().contiguous()
        Luu = self._cholesky(Kuu)

        Kuf = self.Kfu().t()

        W = Kuf.trtrs(Luu, upper=False)[0]

        # compute D = diag[Kff - Qff + noise*I]
        D = self.noise.expand(N)
        Kffdiag = self.Kffdiag()
        Qffdiag = W.pow(2).sum(dim=0)  # Kuf@inv(Kuu)@Kfu = W@W.T
        D = D + Kffdiag - Qffdiag

        # compute K = I + W @ inv(D) @ W.T = L @ L.T
        W_Dinv = W / D
        K = W_Dinv.matmul(W.t()).contiguous()
        K.view(-1)[::M + 1] += 1  # add identity matrix to K
        L = self._cholesky(K)  # MxM

        # get y_residual and convert it into 2D tensor for packing
        y_residual = self.y - self.mean_function(*self.train_points)

        # if 1dim output, then make y into size Nx1
        # when using Ldim output (CI GPs), we have GP on each dim, so each dim is a column
        y_2D = y_residual.reshape(-1, N).t()  # dim is NxL
        W_Dinv_y = W_Dinv.matmul(y_2D)  # 7x5 (MxL)

        # End caching ----------

        Kus = self.Kus(test_points)
        Ws = Kus.trtrs(Luu, upper=False)[0]
        # pack so we can do elimination simultaneously
        pack = torch.cat((W_Dinv_y, Ws), dim=1)  # pack = [ W_Dinv_y Ws ]
        Linv_pack = pack.trtrs(L, upper=False)[0]  # Linv_pack = [ Linv_W_Dinv_y Linv_Ws ]
        # unpack
        Linv_W_Dinv_y = Linv_pack[:, :W_Dinv_y.shape[1]]
        Linv_Ws = Linv_pack[:, W_Dinv_y.shape[1]:]

        # compute posterior predictive loc
        loc = Linv_Ws.t().mm(Linv_W_Dinv_y)
        if self.y.dim() == 1:
            loc = loc.squeeze()  # convert shape from (Ntext,1) to (Ntest,) i.e. reduce dim to 1

        # compute posterior predictive cov
        C = self._get_num_test_points(test_points)
        if full_cov:
            Kss = self.Kss(test_points).contiguous()
            if not noiseless:
                Kss.view(-1)[::C + 1] += self.noise  # add noise to the diagonal
            Qss = Ws.t().matmul(Ws)
            cov = Kss - Qss + Linv_Ws.t().matmul(Linv_Ws)
        else:
            Kssdiag = self.Kssdiag(test_points)
            if not noiseless:
                Kssdiag = Kssdiag + self.noise
            Qssdiag = Ws.pow(2).sum(dim=0)
            cov = Kssdiag - Qssdiag + Linv_Ws.pow(2).sum(dim=0)

        cov_shape = self.y.shape[:-1] + (C, C)
        cov = cov.expand(cov_shape)

        mu = self.mean_function(*test_points) if type(test_points) == tuple else self.mean_function(test_points)
        return loc + mu, cov

    def _zero_mean_function(self, x):
        return x.new_zeros(x.shape)  # creates zero tensor with same shape, dtype, etc...

    def _cholesky(self, M, upper=False):
        while True:
            try:
                # keep adding jitter until it works
                self._add_jitter(M)
                T = torch.cholesky(M, upper=upper)
                return T
            except RuntimeError as error:
                print("Cholesky failed, retrying.")
                print(error)

    def _add_jitter(self, matrix):
        matrix.view(-1)[::(matrix.shape[0]) + 1] += self.jitter

    def _check_test_points_shape(self, test_points):
        """
        Checks the correction of the shape of new data.
        :param torch.Tensor test_points: A input data for testing. Note that
            ``test_points.shape[1:]`` must be the same as ``self.X.shape[1:]``.
        """
        if test_points.dim() != self.X.dim():
            raise ValueError("Train data and test data should have the same "
                             "number of dimensions, but got {} and {}."
                             .format(self.X.dim(), test_points.dim()))
        if self.X.shape[1:] != test_points.shape[1:]:
            raise ValueError("Train data and test data should have the same "
                             "shape of features, but got {} and {}."
                             .format(self.X.shape[1:], test_points.shape[1:]))


def testStuff():
    N = 100
    M = 7
    X = dists.Uniform(0.0, 5.0).sample(sample_shape=(N,))
    y = 0.5 * torch.sin(3 * X) + dists.Normal(0.0, 0.1).sample(sample_shape=(N,))
    Xu = torch.linspace(0.1, 4.9, M)

    sgpr = SparseGPRegression(X, y, RotationKernel, Xu)

    optimizer = torch.optim.Adam(sgpr.parameters(), lr=0.005)

    n_steps = 10
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
