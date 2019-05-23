import matplotlib
import sys
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from vae import FaceVAE
import h5py
import scipy as sp
import os
import pdb
import logging
import pylab as pl
from utils import smartSum, smartAppendDict, smartAppend, export_scripts
from callbacks import callback, save_history
from data_parser import read_face_data, FaceDataset
from optparse import OptionParser
import logging
import pickle
import time


parser = OptionParser()
parser.add_option(
    "--data",
    dest="data",
    type=str,
    default="../../../data/data_faces.h5",
    help="dataset path",
)
parser.add_option(
    "--outdir", dest="outdir", type=str, default="./out/cvae", help="output dir"
)
parser.add_option("--seed", dest="seed", type=int, default=0, help="seed")
parser.add_option(
    "--filts", dest="filts", type=int, default=32, help="number of convol filters"
)
parser.add_option("--zdim", dest="zdim", type=int, default=256, help="zdim")
parser.add_option(
    "--vy", dest="vy", type=float, default=2e-3, help="conditional norm lik variance"
)
parser.add_option("--lr", dest="lr", type=float, default=2e-4, help="learning rate")
parser.add_option("--bs", dest="bs", type=int, default=64, help="batch size")
parser.add_option(
    "--epoch_cb",
    dest="epoch_cb",
    type=int,
    default=100,
    help="number of epoch by which a callback (plot + dump weights) is executed",
)
parser.add_option(
    "--epochs", dest="epochs", type=int, default=10000, help="total number of epochs"
)
parser.add_option("--debug", action="store_true", dest="debug", default=False)
(opt, args) = parser.parse_args()
opt_dict = vars(opt)


if not os.path.exists(opt.outdir):
    os.makedirs(opt.outdir)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# output dir
wdir = os.path.join(opt.outdir, "weights")
fdir = os.path.join(opt.outdir, "plots")
hdir = os.path.join(opt.outdir, "history")
if not os.path.exists(wdir):
    os.makedirs(wdir)
if not os.path.exists(fdir):
    os.makedirs(fdir)
if not os.path.exists(hdir):
    os.makedirs(hdir)

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

# extract VAE settings and export
vae_cfg = {"nf": opt.filts, "zdim": opt.zdim, "vy": opt.vy, "cvae": True}
pickle.dump(vae_cfg, open(os.path.join(opt.outdir, "vae.cfg.p"), "wb"))


def main():

    torch.manual_seed(opt.seed)

    if opt.debug:
        pdb.set_trace()

    # define VAE and optimizer
    vae = FaceVAE(**vae_cfg).to(device)

    # optimizer
    optimizer = optim.Adam(vae.parameters(), lr=opt.lr)

    # load data
    img, obj, view = read_face_data(opt.data)  # image, object, and view
    train_data = FaceDataset(img["train"], obj["train"], view["train"])
    val_data = FaceDataset(img["val"], obj["val"], view["val"])
    train_queue = DataLoader(train_data, batch_size=opt.bs, shuffle=True)
    val_queue = DataLoader(val_data, batch_size=opt.bs, shuffle=False)

    history = {}
    for epoch in range(opt.epochs):

        # train and eval
        ht = train_ep(vae, train_queue, optimizer)
        hv = eval_ep(vae, val_queue)
        smartAppendDict(history, ht)
        smartAppendDict(history, hv)
        logging.info(
            "epoch %d - train_mse: %f - test_mse %f" % (epoch, ht["mse"], hv["mse_val"])
        )

        # callbacks
        if epoch % opt.epoch_cb == 0:
            logging.info("epoch %d - executing callback" % epoch)
            wfile = os.path.join(wdir, "weights.%.5d.pt" % epoch)
            ffile = os.path.join(fdir, "plot.%.5d.png" % epoch)
            torch.save(vae.state_dict(), wfile)
            callback(epoch, val_queue, vae, history, ffile, device)
            save_history(history, hdir, pickle=True)


def train_ep(vae, train_queue, optimizer):

    rv = {}
    vae.train()

    for batch_i, data in enumerate(train_queue):

        # forward
        y, _, w, _ = data
        eps = Variable(torch.randn(y.shape[0], 256), requires_grad=False)
        y, w, eps = y.to(device), w.to(device), eps.to(device)
        elbo, mse, nll, kld = vae.forward(y, eps, w)
        loss = elbo.sum()

        # back propagate
        optimizer.zero_grad()
        loss.backward()

        # if gradient explosion, should get progressively bigger until NaNs
        print("loss=", loss)
        print("decoder (max_weight, max_grad)")
        for layer in reversed(vae.dconv):
            print(torch.max(layer.conv1.weight), torch.max(layer.conv1.weight.grad))
        print(torch.max(vae.dense_dec.weight), torch.max(vae.dense_dec.weight.grad))
        print("encoder (max_weight, max_grad)")
        print(torch.max(vae.dense_zm.weight), torch.max(vae.dense_zm.weight.grad))
        print(torch.max(vae.dense_zs.weight), torch.max(vae.dense_zs.weight.grad))
        for layer in reversed(vae.econv):
            print(torch.max(layer.conv1.weight), torch.max(layer.conv1.weight.grad))

        print("total econv[0].conv1.weight nans =", torch.isnan(vae.econv[0].conv1.weight).sum())

        optimizer.step()

        # sum metrics
        _n = train_queue.dataset.Y.shape[0]
        smartSum(rv, "mse", float(mse.data.sum().cpu()) / float(_n))
        smartSum(rv, "nll", float(nll.data.sum().cpu()) / float(_n))
        smartSum(rv, "kld", float(kld.data.sum().cpu()) / float(_n))
        smartSum(rv, "loss", float(elbo.data.sum().cpu()) / float(_n))

    return rv


def eval_ep(vae, val_queue):
    rv = {}
    vae.eval()

    with torch.no_grad():

        for batch_i, data in enumerate(val_queue):

            # forward
            y, _, w, _ = data
            eps = Variable(torch.randn(y.shape[0], 256), requires_grad=False)
            y, w, eps = y.to(device), w.to(device), eps.to(device)
            elbo, mse, nll, kld = vae.forward(y, eps, w)

            # sum metrics
            _n = val_queue.dataset.Y.shape[0]
            smartSum(rv, "mse_val", float(mse.data.sum().cpu()) / float(_n))
            smartSum(rv, "nll_val", float(nll.data.sum().cpu()) / float(_n))
            smartSum(rv, "kld_val", float(kld.data.sum().cpu()) / float(_n))
            smartSum(rv, "loss_val", float(elbo.data.sum().cpu()) / float(_n))

    return rv


if __name__ == "__main__":
    main()
