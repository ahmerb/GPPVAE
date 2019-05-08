import matplotlib
matplotlib.use("Qt5Agg") # TODO

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.vae import RotatedMnistVAE
import pdb
import logging
from train.rotated_mnist_fitc.utils import smartSum, smartAppendDict, export_scripts
from train.rotated_mnist_fitc.callbacks import callback
from optparse import OptionParser
import pickle
from train.rotated_mnist_fitc.rotated_mnist import RotatedMnistDataset, ToTensor, getMnistPilThrees


parser = OptionParser()
parser.add_option(
    "--data",
    dest="data",
    type=str,
    default="../../../mnist_data",
    help="dataset path",
)
parser.add_option(
    "--outdir", dest="outdir", type=str, default="./out/vae", help="output dir"
)
parser.add_option("--seed", dest="seed", type=int, default=0, help="seed")
parser.add_option(
    "--filts", dest="filts", type=int, default=8, help="number of convol filters"
)
parser.add_option("--zdim", dest="zdim", type=int, default=16, help="zdim")
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# output dir
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

# extract VAE settings and export
vae_cfg = {"nf": opt.filts, "zdim": opt.zdim, "vy": opt.vy}
pickle.dump(vae_cfg, open(os.path.join(opt.outdir, "vae.cfg.p"), "wb"))


def main():

    torch.manual_seed(opt.seed)

    if opt.debug:
        pdb.set_trace()

    train_pil_ims = getMnistPilThrees(root_dir=opt.data, start_ix=0, end_ix=400)
    # test_pil_ims = getMnistPilThrees(root_dir=opt.data, start_ix=400, end_ix=500)
    valid_pil_ims = getMnistPilThrees(root_dir=opt.data, start_ix=500, end_ix=600)
    train_data = RotatedMnistDataset(train_pil_ims, transform=ToTensor())
    # test_data = RotatedMnistDataset(test_pil_ims, transform=ToTensor())
    valid_data = RotatedMnistDataset(valid_pil_ims, transform=ToTensor())
    train_queue = DataLoader(train_data, batch_size=opt.bs, shuffle=True)
    # test_queue = DataLoader(test_data, batch_size=opt.bs, shuffle=False)
    valid_queue = DataLoader(valid_data, batch_size=opt.bs, shuffle=False)

    # define VAE and optimizer
    vae = RotatedMnistVAE(**vae_cfg).to(device)

    # optimizer
    optimizer = optim.Adam(vae.parameters(), lr=opt.lr)

    # load data

    history = {}
    for epoch in range(opt.epochs):

        # train and eval
        ht = train_ep(vae, train_queue, optimizer)
        hv = eval_ep(vae, valid_queue)
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
            callback(epoch, valid_queue, vae, history, ffile, device)


def train_ep(vae, train_queue, optimizer):

    rv = {}
    vae.train()

    for batch_i, data in enumerate(train_queue):
        print(batch_i)

        # forward
        y = data[0].unsqueeze(dim=1) # make bsx28x28 into bsx1x28x28 for compatibility with VAE (colour channel = 1)
        eps = Variable(torch.randn(y.shape[0], opt.zdim), requires_grad=False)
        y, eps = y.to(device), eps.to(device)
        elbo, mse, nll, kld = vae.forward(y, eps)
        loss = elbo.sum()

        # back propagate
        optimizer.zero_grad()
        loss.backward()
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
            y = data['image']
            eps = Variable(torch.randn(y.shape[0], opt.zdim), requires_grad=False)
            y, eps = y.to(device), eps.to(device)
            elbo, mse, nll, kld = vae.forward(y, eps)

            # sum metrics
            _n = val_queue.dataset.Y.shape[0]
            smartSum(rv, "mse_val", float(mse.data.sum().cpu()) / float(_n))
            smartSum(rv, "nll_val", float(nll.data.sum().cpu()) / float(_n))
            smartSum(rv, "kld_val", float(kld.data.sum().cpu()) / float(_n))
            smartSum(rv, "loss_val", float(elbo.data.sum().cpu()) / float(_n))

    return rv


if __name__ == "__main__":
    main()
