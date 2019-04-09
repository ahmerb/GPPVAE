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

# select device (CPU/GPU)
# device_nn = None
# device_gp = None
# if opt.enable_cuda and torch.cuda.is_available() and torch.cuda.device_count() >= 2:
#     device_nn = torch.device('cuda:0')
#     device_gp = torch.device('cuda:1')
# elif opt.enable_cuda and torch.cuda.is_available():
#     device_nn = torch.device('cuda')
#     device_gp = torch.device('cuda')
# else:
#     device_nn = torch.device('cpu')
#     device_gp = torch.device('cpu')
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
    x_features = UnobservedFeatureVectors(Dt, P, opt.xdim)
    w_features = UnobservedFeatureVectors(Wt, Q, Q)

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
    Xu = torch.linspace(Dt_min, Dt_max, M).expand(opt.xdim, M).t().to(device)
    Wu = torch.linspace(Wt_min, Wt_max, M).expand(Q, M).t().to(device)
    gp = DualInputSparseGPRegression(x_features(Dt), w_features(Wt), None, RotationKernel, RotationKernel, KernelComposer.Product, Xu, Wu) \
            .to(device)
    # TODO change obj kernel to gaussian kernel

    # put feature vec and gp params in one param-list
    gp_params = nn.ParameterList()
    gp_params.extend(x_features.parameters())
    gp_params.extend(w_features.parameters())
    gp_params.extend(gp.parameters())

    # define optimizers
    vae_optimizer = optim.Adam(vae.parameters(), lr=opt.vae_lr)
    gp_optimizer = optim.Adam(gp_params, lr=opt.gp_lr)

    # begin training
    for epoch in range(opt.epochs):
        print('epoch start')

        vae_optimizer.zero_grad()
        gp_optimizer.zero_grad()
        vae.train()
        gp.train()
        x_features.train()
        w_features.train()

        N = train_queue.dataset.Y.shape[0]

        Zm = torch.zeros(N, vae_cfg["zdim"], device='cpu')
        Zs = torch.zeros(N, vae_cfg["zdim"], device='cpu')
        Z = torch.zeros(N, vae_cfg["zdim"], device='cpu')
        Eps = torch.normal(mean=torch.zeros(N, vae_cfg["zdim"], device='cpu'), std=torch.ones(N, vae_cfg["zdim"], device='cpu'))

        recon_term = []
        recon_term_append = recon_term.append

        # for each minibatch
        for batch_i, data in enumerate(train_queue):
            print("encode minibatch")

            # get data from minibatch
            idxs = data[-1]
            y = data[0].to(device) # size(bs, 3, 128, 128)
            eps = Eps[idxs].to(device=device, copy=True)

            print("y", y.device)
            print("eps", eps.device)

            # forward encoder on minibatch
            zm, zs = vae.encode(y) # size(bs, zdim)

            # sample z's
            z = zm + zs * eps # need to replace this with matmul?? size(bs, zdim)

            # forward decoder on minibatch
            yr = vae.decode(z) # size(bs, 3, 128, 128)

            # store z's of this minibatch
            Zm[idxs] = zm.detach().to('cpu')
            Zs[idxs] = zs.detach().to('cpu')
            Z[idxs]  = z.detach().to('cpu')
            print("Zm", Zm.device)

            # compute and update mse and nll
            recon_term_batch, _ = vae.nll(y, yr) # size(bs, 1)
            recon_term_append(recon_term_batch.sum())
            # XXX .item()
            # wait, recon_term still needs to be a tensor, as we use it later to compute `loss`, and we
            # need to backprop through it....

            print("memory usage: ", torch.cuda.max_memory_allocated())

        print(recon_term)


        # forward gp (using FITC)
        Z.to(device)
        gp.y = Z
        gp_nll = -gp()
        gp_nll = gp_nll / vae.K
        # TODO do we still need to divide by vae.K???

        # penalization (compute the regularization term)
        pen_term = -0.5 * Zs.sum() / vae.K

        # loss and backprop
        loss = recon_term.sum() + gp_nll + pen_term.sum().to(device)
        loss.backward()

        # XXX can't free each vae tensor used in forward pass as need forward tensors to do backwards pass
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
