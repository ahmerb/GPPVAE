# flake8: noqa

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import scipy as sp
import matplotlib

import sys
import os
import pdb
import logging
from optparse import OptionParser
import pickle

from models.vae import FaceVAE
from models.gp.dual_input_sparse_gp import DualInputSparseGPRegression
from models.unobserved_feature_vectors import UnobservedFeatureVectors
from kernels.kernels import RotationKernel, KernelComposer

from utils import smartSum, smartAppendDict, smartAppend, export_scripts
from callbacks import callback_gppvae
from data_parser import read_face_data, FaceDataset

import gpytorch
from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import WhitenedVariationalStrategy

# TODO upgrade from optparse to argparse
parser = OptionParser()
parser.add_option(
    "--data",
    dest="data",
    type=str,
    default="../../data/data_faces.h5",
    help="dataset path",
)
parser.add_option(
    "--outdir", dest="outdir", type=str, default="./out/gppvae", help="output dir"
)
parser.add_option("--vae_cfg", dest="vae_cfg", type=str, default=None)
parser.add_option("--vae_weights", dest="vae_weights", type=str, default=None)
parser.add_option("--seed", dest="seed", type=int, default=0, help="seed")
parser.add_option(
    "--vae_lr",
    dest="vae_lr",
    type=float,
    default=2e-4,
    help="learning rate of vae params",
)
parser.add_option(
    "--gp_lr", dest="gp_lr", type=float, default=1e-3, help="learning rate of gp params"
)
parser.add_option(
    "--xdim", dest="xdim", type=int, default=64, help="rank of object linear covariance"
)
parser.add_option("--bs", dest="bs", type=int, default=64, help="batch size")
parser.add_option(
    "--epoch_cb",
    dest="epoch_cb",
    type=int,
    default=10,
    help="number of epoch by which a callback (plot + dump weights) is executed",
)
parser.add_option(
    "--epochs", dest="epochs", type=int, default=100, help="total number of epochs"
)
parser.add_option("--debug", action="store_true", dest="debug", default=False)
parser.add_option(
    '--enable-cuda', action='store_true', dest="enable_cuda", help='Enable CUDA', default=False
)
(opt, args) = parser.parse_args()
opt_dict = vars(opt)

# parse args

# VAE config and weights
if opt.vae_cfg is None:
    opt.vae_cfg = "../faceplace/out/vae/vae.cfg.p"
vae_cfg = pickle.load(open(opt.vae_cfg, "rb"))

if opt.vae_weights is None:
    opt.vae_weights = "../faceplace/out/vae/weights/weights.00900.pt"

device = torch.device('cpu')
if opt.enable_cuda and torch.cuda.is_available():
    device = torch.device('cuda:0')

if not opt.enable_cuda:
    matplotlib.use("Qt5Agg")

# output dir
if not os.path.exists(opt.outdir):
    os.makedirs(opt.outdir)

wdir = os.path.join(opt.outdir, "weights")
fdir = os.path.join(opt.outdir, "plots")

if not os.path.exists(wdir):
    os.makedirs(wdir)
if not os.path.exists(fdir):
    os.makedirs(fdir)

# copy code to output folder
export_scripts(os.path.join(opt.outdir, "scripts"))

# create logfile
log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)
fh = logging.FileHandler(os.path.join(opt.outdir, "log.txt"))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info("opt = %s", opt)

class SVGPRegression(AbstractVariationalGP):
    """
    Creates x and w tensors of shape num_outputs x n x d, and passes these to GPyTorch.
    z_dim is dimension of GP output.
    """
    def __init__(self, inducing_points, x_dim, w_dim, z_dim):
        """inducing points is tensor size: M x (x_dim+w_dim)"""
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0), batch_size=z_dim)
        variational_strategy = WhitenedVariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(SVGPRegression, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean(batch_size=z_dim)
        self.obj_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_size=z_dim),
            batch_size=z_dim
        )
        self.view_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel(batch_size=z_dim),
            batch_size=z_dim
        )
        self.x_dim = x_dim
        self.w_dim = w_dim
        self.z_dim = z_dim

    def forward(self, inp):
        """inp is tensor size: n x (x_dim+w_dim)"""
        x = inp[:, :self.x_dim] # n x x_dim
        x = x.expand(self.z_dim, *x.shape) # z_dim x n x x_dim
        w = inp[:, self.x_dim:] # n x w_dim
        w = w.expand(self.z_dim, *w.shape) # z_dim x n x w_dim
        mean = self.mean_module(inp)
        # print("z_dim x bs x x/w_dim")
        # print("x.shape=", x.shape)
        # print("w.shape=", w.shape)
        # print("bs=n=", x.shape[0])
        # print("x_dim=", x_dim)
        # print("w_dim=", w_dim)
        # print("z_dim=", z_dim)
        covar = self.obj_covar_module(x) * self.view_covar_module(w) # z_dim x n x n
        dist = gpytorch.distributions.MultivariateNormal(mean, covar)
        return dist


def main():
    torch.manual_seed(opt.seed)

    if opt.debug:
        pdb.set_trace()

    # load data
    img, obj, view = read_face_data(opt.data)  # image, object, and view
    train_data = FaceDataset(img["train"], obj["train"], view["train"])
    val_data = FaceDataset(img["val"], obj["val"], view["val"])

    # iterating over a queue gives data minibatches that look like:
    # data = [ imgs, objs, views, indices of corresponding data ]
    train_queue = DataLoader(train_data, batch_size=opt.bs, shuffle=True)
    val_queue = DataLoader(val_data, batch_size=opt.bs, shuffle=False)

    # convert images and auxillary data (obj and view id's) to long tensors
    Dt = obj["train"][:, 0].long()
    Wt = view["train"][:, 0].long()
    Dv = obj["val"][:, 0].long()
    Wv = view["val"][:, 0].long()

    # number of unique object and views
    P = sp.unique(obj["train"]).shape[0]
    Q = sp.unique(view["train"]).shape[0]

    # we have sparse gp's, so we now can use arbitrarily large object feature vec size (before it was limited)
    x_dim = opt.xdim
    w_dim = Q
    x_features = UnobservedFeatureVectors(Dt, P, x_dim)
    w_features = UnobservedFeatureVectors(Wt, Q, w_dim)

    # define VAE and optimizer
    vae = FaceVAE(**vae_cfg)
    RV = torch.load(opt.vae_weights, map_location=device) # remove map_location when using gpu
    vae.load_state_dict(RV)
    vae.to(device)

    # define gp

    # init inducing inputs
    M = 100 # num inducing points
    with torch.no_grad():
        Dt_min = torch.min(Dt)
        Dt_max = torch.max(Dt)
        Wt_min = torch.min(Wt)
        Wt_max = torch.min(Wt)
    Xu = torch.linspace(Dt_min, Dt_max, M).expand(x_dim, M).t().to(device) # M x x_dim
    Wu = torch.linspace(Wt_min, Wt_max, M).expand(w_dim, M).t().to(device) # M x w_dim

    inducing_points = torch.cat((Xu, Wu), 1) # M x (x_dim+w_dim)

    z_dim = vae_cfg["zdim"]
    gp_model = SVGPRegression(inducing_points, x_dim, w_dim, z_dim).to(device)
    gp_likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    gp_model.train()
    gp_likelihood.train()

    vae_optim = optim.Adam(vae.parameters(), lr=opt.vae_lr)
    gp_params = nn.ParameterList()
    gp_params.extend(x_features.parameters())
    gp_params.extend(w_features.parameters())
    gp_params.extend(gp_model.parameters())
    gp_optim = optim.Adam(gp_params, lr=opt.gp_lr)

    gp_mll = gpytorch.mlls.VariationalELBO(gp_likelihood, gp_model, num_data=Dt.size(0), combine_terms=True)

    for epoch in range(opt.epochs):
        print('epoch start')
        for batch_i, data in enumerate(train_queue):
            y, x, w, idxs = data
            y = y.to(device) # bs x 3 x 128 x 128
            x = x.to(device) # bs x x_dim
            w = w.to(device) # bs x w_dim
            #idxs = idxs.to(device)

            # vae
            print('vae')
            zm, zs = vae.encode(y)
            eps = torch.normal(mean=torch.zeros(y.size(0), z_dim), std=torch.ones(y.size(0), z_dim))
            z = zm + zs * eps
            yr = vae.decode(z)
            recon_term, _ = vae.nll(y, yr)

            # gp
            print('gp')
            x_ = x_features(x[:, 0].long()) # bs x x_dim
            w_ = w_features(w[:, 0].long()) # bs x w_dim
            gp_inp = torch.cat((x_, w_), 1) # bs x (x_dim+w_dim)
            gp_likelihood_dist = gp_model(gp_inp)
            gp_mll_term = -gp_mll(gp_likelihood_dist, z.t()) #/ vae.K # z is bs x z_dim

            # penalization
            pen_term = -0.5 * zs.sum() / vae.K

            # optim step
            loss = recon_term.sum() + gp_mll_term.sum() + pen_term.sum().to(device)
            print('Epoch %d [%d/%d] - Loss: %.3f [%.3f, %.3f, %.3f]' % (epoch + 1, batch_i, len(train_queue), loss.item(), recon_term.sum().item(), gp_mll_term.sum().item(), pen_term.item()))
            loss.backward()
            vae_optim.step()
            gp_optim.step()


def evaluate_gppvae(vae, gp, Zm, x_features, w_features, val_queue, epoch, device):
    rv = {}

    vae.eval()
    gp.eval()
    x_features.eval()
    w_features.eval()

    with torch.no_grad():
        Zm.to(device)
        gp.y = Zm

        nll = torch.zeros(opt.bs, 1, device=device)
        mse = torch.zeros(opt.bs, 1, device=device)

        for batch_i, data in enumerate(val_queue):
            print("eval minibatch")
            y_test, x_test, w_test, idxs = data

            # retrieve feature vectors
            x_test_features = x_features(x_test[:, 0].long())
            w_test_features = w_features(w_test[:, 0].long())

            # gp posterior
            z_test_mu, z_test_cov = gp.posterior_predictive(x_test_features, w_test_features)

            # decode to get reconstructions
            y_test_recon = vae.decode(z_test_mu)

            # compute error
            nll_batch, mse_batch = vae.nll(y_test, y_test_recon)
            nll += nll_batch
            mse += mse_batch

        print("VALID: epoch {}: nll={}, mse={}".format(epoch, nll.item(), mse.item()))


if __name__ == "__main__":
    main()
