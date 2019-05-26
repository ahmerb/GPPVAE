import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import torch
import torch.distributions as dists
import math  # to compute the constant log(2*pi)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from kernels.kernels import SEKernel
from models.gp.sparse_gp import SparseGPRegression
from train.faceplace.utils import smartAppendDict
from train.faceplace.callbacks import save_history


matplotlib.use('Qt5Agg')


def main():
    N = 1000
    M = 10

    # construct X manually where different parts of domain [-5, 20] have more/less datapoints
    X1 = dists.Uniform(0.1, 0.5).sample(sample_shape=(100,))
    X2 = dists.Uniform(0.1, 2.0).sample(sample_shape=(300,)) # 400
    X3 = dists.Uniform(3.0, 3.5).sample(sample_shape=(50,))
    X4 = dists.Uniform(5.5, 7.0).sample(sample_shape=(50,)) # 500
    X5 = dists.Uniform(7.0, 7.5).sample(sample_shape=(50,))
    X6 = dists.Uniform(8.5, 9.0).sample(sample_shape=(50,)) # 600
    X7 = dists.Uniform(10.0, 13.0).sample(sample_shape=(300,))
    X8 = dists.Uniform(13.0, 14.0).sample(sample_shape=(100,)) # 1000
    X = torch.cat((
        X1, X2, X3, X4, X5, X6, X7, X8
    ))
    y = 0.5 * torch.sin(3 * X) + dists.Normal(0.0, 0.2).sample(sample_shape=(N,))
    Xu = torch.linspace(0.1, 17.0, M)
    Xtest = torch.linspace(-4.0, 19.0, 500)

    sgpr = SparseGPRegression(X, y, SEKernel, Xu, lengthscale=2.)

    optimizer = torch.optim.Adam(sgpr.parameters(), lr=0.1)

    n_steps = 1000
    epoch_cb = 10
    print("begin training")
    history = {}
    for i in range(n_steps):
        optimizer.zero_grad()
        loss = -sgpr()
        loss.backward()
        optimizer.step()
        print("epoch {}: loss={}".format(i, loss.item()))
        if i % epoch_cb == 0:
            print("exec callback")
            hv = callback(Xtest, sgpr, loss.item(), i)
            smartAppendDict(history, hv)
            save_history(history, "./", pickle=False)


def callback(Xtest, sgpr, loss, epoch, nsamples=3):
    with torch.no_grad():
        Ns = Xtest.shape[0]
        mu, cov = sgpr.posterior_predictive(Xtest, full_cov=True)

        # L = stddev = sqrt(cov) = cholesky(cov)
        L = sgpr._cholesky(cov)

        # draw samples from posterior
        f_posterior = (mu + L.mm(torch.normal(mean=torch.zeros(nsamples, Ns),
                                              std=torch.ones(nsamples, Ns)).t()).t()).t()

        # convert to numpy
        numpy = {
            'X': sgpr.X.detach().numpy(),
            'Xtest': Xtest.detach().numpy(),
            'mu': mu.detach().numpy(),
            'y': sgpr.y.detach().numpy(),
            'f_posterior': f_posterior.detach().numpy(),
            'Ldiag': L.detach().diag().numpy(),
            'Xu': sgpr.Xu.detach().numpy()
        }

        fig, ax = plt.subplots()
        ax.plot(numpy['X'], numpy['y'], 'bs', ms=1)
        ax.plot(numpy['Xtest'], numpy['f_posterior'])
        ax.fill_between(numpy['Xtest'],
                        numpy['mu'] - 2 * numpy['Ldiag'],
                        numpy['mu'] + 2 * numpy['Ldiag'],
                        color="#969696")
        ax.plot(numpy['Xtest'], numpy['mu'], 'r-', lw=2)
        ax.plot(numpy['Xu'], np.zeros(numpy['Xu'].shape[-1]) + plt.ylim()[0], 'r^')
        ax.set_title('%d Samples from GP Posterior' % nsamples)
        filename = "plot.%.5d.eps" % epoch
        plt.savefig(filename, format='eps', dpi=1000)

        # save hyperparams and -gp_mll
        return {
            "-gp_mll": loss,
            "lengthscale": float(sgpr.kernel.lengthscale.item()),
            "beta": float(sgpr.kernel.beta.item())
        }


if __name__ == "__main__":
    main()
