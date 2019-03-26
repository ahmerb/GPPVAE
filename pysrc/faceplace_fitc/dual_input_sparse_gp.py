import torch
from torch.nn import Parameter
import torch.distributions as dists
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pl
import numpy as np
from sparse_gp import SparseGPRegression
from kernels import RotationKernel, LinearKernel

class KernelComposer():
    @staticmethod
    def Product(K1, K2):
        return K1 * K2 # elem-wise

    @staticmethod
    def Addition(K1, K2):
        return K1 + K2

class DualInputSparseGPRegression(SparseGPRegression):
    def __init__(self, X, W, y, x_kernel, w_kernel, kernel_composer, Xu, Wu, mean_function=None, noise=0.5):
        super(DualInputSparseGPRegression, self).__init__(X, y, x_kernel, Xu, mean_function=mean_function, noise=noise)
        self.x_kernel = self.kernel
        del self.kernel
        self.w_kernel = w_kernel()
        self.W = W
        self.train_points = (X, W)
        self.Wu = Parameter(Wu)
        self.kernel_composer = kernel_composer

    def Kfu(self):
        K1fu = self.x_kernel(self.X, self.Xu)
        K2fu = self.w_kernel(self.W, self.Wu)
        return self.kernel_composer(K1fu, K2fu)

    def Kuu(self):
        K1uu = self.x_kernel(self.Xu, self.Xu)
        K2uu = self.w_kernel(self.Wu, self.Wu)
        return self.kernel_composer(K1uu, K2uu)

    def Kffdiag(self):
        K1ffdiag = self.x_kernel(self.X, self.X, diag=True)
        K2ffdiag = self.w_kernel(self.W, self.W, diag=True)
        return self.kernel_composer(K1ffdiag, K2ffdiag)

    def Kus(self, test_points):
        Xnew, Wnew = test_points
        K1us = self.x_kernel(self.Xu, Xnew)
        K2us = self.w_kernel(self.Wu, Wnew)
        return self.kernel_composer(K1us, K2us)

    def Kss(self, test_points):
        Xnew, Wnew = test_points
        K1ss = self.x_kernel(Xnew)
        K2ss = self.w_kernel(Wnew)
        return self.kernel_composer(K1ss, K2ss)

    def Kssdiag(self, test_points):
        Xnew, Wnew = test_points
        K1ssdiag = self.x_kernel(Xnew, diag=True)
        K2ssdiag = self.w_kernel(Wnew, diag=True)
        return self.kernel_composer(K1ssdiag, K2ssdiag)

    def _get_num_test_points(self, test_points):
        Xnew, _ = test_points
        return Xnew.size(0)

    def _check_test_points_shape(self, test_points):
        Xnew, Wnew = test_points
        if Xnew.size(0) != Wnew.size(0):
            raise ValueError("X and W test data should have the same "
                             "number of samples, but got {} and {}."
                             .format(Xnew.size(0), Wnew.size(0)))
        super(DualInputSparseGPRegression, self)._check_test_points_shape(Xnew)
        if Wnew.dim() != self.W.dim():
            raise ValueError("Train data and test data should have the same "
                             "number of dimensions, but got {} and {}."
                             .format(self.W.dim(), Wnew.dim()))
        if self.W.shape[1:] != Wnew.shape[1:]:
            raise ValueError("Train data and test data should have the same "
                             "shape of features, but got {} and {}."
                             .format(self.W.shape[1:], Wnew.shape[1:]))

    def _zero_mean_function(self, x, w):
        N = x.shape[0] # num inputs, also equals w.shape[0]
        L = self.y.shape[1] # output dimension
        return x.new_zeros(N, L)

    def posterior_predictive(self, Xnew, Wnew, full_cov=False, noiseless=True):
        test_points = (Xnew, Wnew)
        return super(DualInputSparseGPRegression, self).posterior_predictive(test_points, full_cov=full_cov, noiseless=noiseless)

    def predict_and_plot(self, Xnew, Wnew, nsamples=3):
        N = Xnew.shape[0]
        mu, cov = self.posterior_predictive(Xnew, Wnew, full_cov=True)

        # L = stddev = sqrt(cov) = cholesky(cov)
        L = self._cholesky(cov)

        # draw samples from posterior
        f_posterior = (mu + L.mm(torch.normal(mean=torch.zeros(nsamples, N), std=torch.ones(nsamples, N)).t()).t())
        
        # convert to numpy
        numpy = {
            'X': self.X.detach().numpy(),
            'W': self.W.detach().numpy(),
            'Xnew': Xnew.detach().numpy(),
            'Wnew': Wnew.detach().numpy(),
            'mu': mu.detach().numpy(),
            'y': self.y.detach().numpy(),
            'f_posterior': f_posterior.detach().numpy(),
            'Ldiag': L.detach().diag().numpy(),
            'Xu': self.Xu.detach().numpy(),
            'Wu': self.Wu.detach().numpy()
        }

        fig = pl.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(numpy['X'], numpy['W'], numpy['y'], c='b')#, 'bs', ms=8)
        for f_post in list(numpy['f_posterior']):
            ax.plot(numpy['Xnew'], numpy['Wnew'], f_post)
        ax.plot(numpy['Xnew'], numpy['Wnew'], numpy['mu']-2*numpy['Ldiag'], color='#dddddd')
        ax.plot(numpy['Xnew'], numpy['Wnew'], numpy['mu']+2*numpy['Ldiag'], color='#dddddd')
        ax.plot(numpy['Xnew'], numpy['Wnew'], numpy['mu'], color='r')
        ax.scatter(numpy['Xu'], np.zeros(numpy['Xu'].shape[-1]) - ax.get_ylim()[0], np.zeros(numpy['Xu'].shape[-1]) - ax.get_zlim()[0], c='r', marker='^')
        ax.scatter(np.zeros(numpy['Xu'].shape[-1]) - ax.get_xlim()[0], numpy['Wu'], np.zeros(numpy['Xu'].shape[-1]) - ax.get_zlim()[0], c='g', marker='^')
        ax.set_title('{} Samples from GP Posterior'.format(nsamples))
        pl.show()

def testStuff():
    N = 100 # num samples
    M = 7   # num inducing points

    X = dists.Uniform(0.0, 5.0).sample(sample_shape=(N,))
    W = dists.Uniform(-5.0, 11).sample(sample_shape=(N,))
    y = 0.5 * torch.sin(3*X) + torch.cos(2*W) + dists.Normal(0.0, 0.2).sample(sample_shape=(N,))
    Xu = torch.linspace(0.1, 4.9, M)
    Wu = torch.linspace(-4.9, 10.9, M)

    sgpr = DualInputSparseGPRegression(X, W, y, RotationKernel, RotationKernel, KernelComposer.Product, Xu, Wu)

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
    Wtest = torch.linspace(-5.0, 11, 10)
    sgpr.predict_and_plot(Xtest, Wtest)

def testStuff2():
    # multiple-dimension input and output (redefine D as Xdim and define O as num inducing points)
    N = 100 # num samples
    O = 7   # num inducing points

    D = 10  # X dim
    R = 4   # W dim
    L = 5   # Y dim

    # sample training data
    X = dists.Uniform(-10.0, 10.0).sample(sample_shape=(N, D))
    W = dists.Uniform(-20.0,  7.0).sample(sample_shape=(N, R))
    # just do some random matmuls with ones to make the matrix shapes work
    Y = torch.sin(3*X).t().mm(torch.cos(2*W)).mm(torch.ones(R, N)).t().mm(torch.ones(D, L)) + dists.Normal(0.0, 0.3).sample(sample_shape=(N, L))

    # init inducing points
    Xu = torch.linspace(-9.99, 9.99, O).expand(D, O).t() # OxD
    Wu = torch.linspace(-19.9, 6.99, O).expand(R, O).t() # OxR

    sgpr = DualInputSparseGPRegression(X, W, Y, RotationKernel, RotationKernel, KernelComposer.Product, Xu, Wu)
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
    print("training complete")

    # eval, posterior predictive (5 test points)
    Ntest = 5
    Xtest = dists.Uniform(-10.0, 10.0).sample(sample_shape=(Ntest, D))
    Wtest = dists.Uniform(-20.0,  7.0).sample(sample_shape=(Ntest, R))
    Ftest = torch.sin(3*Xtest).t().mm(torch.cos(2*Wtest)).mm(torch.ones(R, Ntest)).t().mm(torch.ones(D, L))

    Fpred = sgpr.posterior_predictive(Xtest, Wtest)
    print("MSE = ", torch.dist(Ftest, Fpred))


if __name__ == "__main__":
    testStuff2()
