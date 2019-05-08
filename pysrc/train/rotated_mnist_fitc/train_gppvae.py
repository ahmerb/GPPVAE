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
from models.gp.dual_input_sparse_gp import SparseGPRegression
from models.unobserved_feature_vectors import UnobservedFeatureVectors
from kernels.kernels import RotationKernel, SEKernel, KernelComposer

from .utils import smartSum, smartAppendDict, smartAppend, export_scripts
from .callbacks import callback_gppvae
from .data_parser import read_face_data, FaceDataset

import torchvision
import torchvision.datasets
from .rotated_mnist import RotatedMnistDataset, ToTensor, getMnistPilThrees


# TODO upgrade from optparse to argparse
parser = OptionParser()
parser.add_option(
    "--data",
    dest="data",
    type=str,
    default='../../../mnist_data',
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

z_dim = vae_cfg["zdim"]

if opt.vae_weights is None:
    opt.vae_weights = "../faceplace/out/vae/weights/weights.00900.pt"

device = torch.device('cpu')
if opt.enable_cuda and torch.cuda.is_available():
    device = torch.device('cuda')

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


def main():
    torch.manual_seed(opt.seed)

    if opt.debug:
        pdb.set_trace()

    train_pil_ims = getMnistPilThrees(root_dir=opt.data, start_ix=0, end_ix=400)
    test_pil_ims  = getMnistPilThrees(root_dir=opt.data, start_ix=400, end_ix=500)
    valid_pil_ims = getMnistPilThrees(root_dir=opt.data, start_ix=500, end_ix=600)
    train_data = RotatedMnistDataset(train_pil_ims, transform=ToTensor())
    test_data  = RotatedMnistDataset(test_pil_ims, transform=ToTensor())
    valid_data = RotatedMnistDataset(valid_pil_ims, transform=ToTensor())
    train_queue = DataLoader(train_data, batch_size=opt.bs, shuffle=True)
    test_queue  = DataLoader(test_data, batch_size=opt.bs, shuffle=False)
    valid_queue = DataLoader(valid_data, batch_size=opt.bs, shuffle=False)

    N = len(train_data)

    num_rotations = 16 # number of unique rotation angles

    # define VAE
    vae = FaceVAE(**vae_cfg)
    vae_state = torch.load(opt.vae_weights, map_location=device)
    vae.load_state_dict(vae_state)
    vae.to(device)

    # define GP

    # init inducing points
    M = 50
    min_rot_angle = torch.tensor([0.0])
    max_rot_angle = torch.tensor([337.5])
    Xu = torch.linspace(min_rot_angle, max_rot_angle, M, device=device)

    # extract auxiliary data
    X = list(map(lambda datapoint: datapoint[1], train_data.data))

    # init gp
    gp = SparseGPRegression(X, None, RotationKernel, Xu).to(device)

    # optimizers
    vae_optimizer = optim.Adam(vae.parameters(), lr=opt.vae_lr)
    gp_optimizer = optim.Adam(gp.parameters(), lr=opt.gp_lr) # TODO: will this include kernel fn params??

    # train loop
    for epoch in range(opt.epochs):
        vae_optimizer.zero_grad()
        gp_optimizer.zero_grad()
        vae.train()
        gp.train()

        Zm = torch.zeros(N, z_dim, device="cpu")
        Zs = torch.zeros(N, z_dim, device="cpu")
        Z = torch.zeros(N, z_dim, device="cpu")
        Eps = torch.normal(mean=torch.zeros(N, z_dim, device='cpu'), std=torch.ones(N, z_dim, device='cpu'))

        recon_term = []
        recon_term_append = recon_term.append
        mse = []
        mse_append = mse.append

        # for each minibatch
        for batch_i, data in enumerate(train_queue):
            print("encode minibatch")

            # get data from minibatch
            idxs = data['index']
            y = data['image'].to(device) # size(bs, 3, 128, 128)
            eps = Eps[idxs].to(device=device, copy=True)

            print("y", y.device)
            print("eps", eps.device)

            # forward encoder on minibatch
            zm, zs = vae.encode(y) # size(bs, zdim)

            # sample z's
            z = zm + zs * eps # size(bs, zdim)

            # forward decoder on minibatch
            yr = vae.decode(z) # size(bs, 3, 128, 128)

            # store z's of this minibatch
            Zm[idxs] = zm.detach().to('cpu')
            Zs[idxs] = zs.detach().to('cpu')
            Z[idxs]  = z.detach().to('cpu')
            print("Zm", Zm.device)

            # compute and update mse and nll
            recon_term_batch, mse_term_batch = vae.nll(y, yr) # size(bs, 1)
            recon_term_append(recon_term_batch.sum())
            mse_batch, mse = vae.nll(y, yr) # size(bs, 1)
            mse_append(mse_batch.sum())
            # XXX .item()
            # wait, recon_term still needs to be a tensor, as we use it later to compute `loss`, and we
            # need to backprop through it....

            print("memory usage: ", torch.cuda.max_memory_allocated())

        print(recon_term)

        # forward gp (using FITC)
        Z.to(device)
        gp.y = Z
        gp_nll = -gp()
        gp_nll = gp_nll / z_dim

        # penalization (compute the regularization term)
        pen_term = -0.5 * Zs.sum() / z_dim

        # loss and backprop
        loss = recon_term.sum() + gp_nll + pen_term.sum().to(device)
        loss.backward()

        print("TRAIN: epoch {}: loss={}, mse={}".format(epoch, loss.item(), mse.item()))

        vae_optimizer.step()
        gp_optimizer.step()

        # eval on validation set (using GPPVAE posterior predictive)
        # rv_eval = evaluate_gppvae(vae, gp, Zm, x_features, w_features, val_queue, epoch, device)
        # NOTE should eval be on test or valid set???????


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
