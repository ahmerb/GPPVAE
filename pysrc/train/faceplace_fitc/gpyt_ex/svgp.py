# flake8: noqa

import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import matplotlib
import urllib.request
import os.path
from scipy.io import loadmat
from math import floor
from torch.utils.data import TensorDataset, DataLoader
import time


matplotlib.use('Qt5Agg')

# Download and preprocess dataset

if not os.path.isfile('3droad.mat'):
    print('Downloading \'3droad\' UCI dataset...')
    urllib.request.urlretrieve('https://www.dropbox.com/s/f6ow1i59oqx05pl/3droad.mat?dl=1', '3droad.mat')

data = torch.Tensor(loadmat('3droad.mat')['data'])
X = data[:, :-1]
X = X - X.min(0)[0]
X = 2 * (X / X.max(0)[0]) - 1
y = data[:, -1]

# Use the first 80% of the data for training, and the last 20% for testing.
train_n = int(floor(0.8*len(X)))

train_x = X[:train_n, :].contiguous()#.cuda()
train_y = y[:train_n].contiguous()#.cuda()

test_x = X[train_n:, :].contiguous()#.cuda()
test_y = y[train_n:].contiguous()#.cuda()

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)


# Create DataLoader

train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)


# Create DKL Feature Extractor

data_dim = train_x.size(-1)

data_dim = train_x.size(-1)

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 1000))
        self.add_module('bn1', torch.nn.BatchNorm1d(1000))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(1000, 1000))
        self.add_module('bn2', torch.nn.BatchNorm1d(1000))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(1000, 500))
        self.add_module('bn3', torch.nn.BatchNorm1d(500))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(500, 50))
        self.add_module('bn4', torch.nn.BatchNorm1d(50))
        self.add_module('relu4', torch.nn.ReLU())
        self.add_module('linear5', torch.nn.Linear(50, 2))

feature_extractor = LargeFeatureExtractor()#.cuda()
# num_features is the number of final features extracted by the neural network, in this case 2.
num_features = 2


# Define GP Regression Layer

from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import WhitenedVariationalStrategy

class GPRegressionLayer(AbstractVariationalGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = WhitenedVariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPRegressionLayer, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# Define DKL Model

class DKLModel(gpytorch.Module):
    def __init__(self, inducing_points, feature_extractor, num_features):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GPRegressionLayer(inducing_points)
        self.num_features = num_features

    def forward(self, x):
        features = self.feature_extractor(x)
        res = self.gp_layer(features)
        return res
inducing_points = feature_extractor(train_x[:500, :])
model = DKLModel(inducing_points=inducing_points, feature_extractor=feature_extractor, num_features=num_features)#.cuda()
likelihood = gpytorch.likelihoods.GaussianLikelihood()#.cuda()


# Training the Model

model.train()
likelihood.train()

num_epochs = 3

optimizer = torch.optim.Adam([
    {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-3},
    {'params': model.gp_layer.parameters()},
    {'params': likelihood.parameters()}
], lr=0.01)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 5], gamma=0.1)

# Our loss object. We're using the VariationalELBO, which essentially just computes the ELBO
mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=train_y.size(0), combine_terms=False)

# We use more CG iterations here because the preconditioner introduced in the NeurIPS paper seems to be less
# effective for VI.
with gpytorch.settings.max_cg_iterations(45):
    for i in range(num_epochs):
        scheduler.step()
        for minibatch_i, (x_batch, y_batch) in enumerate(train_loader):
            start_time = time.time()
            optimizer.zero_grad()
            output = model(x_batch)
            log_lik, kl_div, log_prior = mll(output, y_batch)
            loss = -(log_lik - kl_div + log_prior)
            print('Epoch %d [%d/%d] - Loss: %.3f [%.3f, %.3f, %.3f]' % (i + 1, minibatch_i, len(train_loader), loss.item(), log_lik.item(), kl_div.item(), log_prior.item()))
            loss.backward()
            optimizer.step()


# Making predictions

# can also make preds in minibatches, using LOVE, etc, (see docs)

model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.max_cg_iterations(50), gpytorch.settings.use_toeplitz(False):
    preds = model(test_x)

print('Test MAE: {}'.format(torch.mean(torch.abs(preds.mean - test_y))))
