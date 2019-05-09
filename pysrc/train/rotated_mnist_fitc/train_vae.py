import matplotlib
matplotlib.use("Qt5Agg") # TODO

import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from models.rotated_mnist_vae import RotatedMnistVAE
import pdb
import logging
from train.rotated_mnist_fitc.utils import smartSum, smartAppendDict, export_scripts
from train.rotated_mnist_fitc.callbacks import callback
from optparse import OptionParser
import pickle
from train.rotated_mnist_fitc.rotated_mnist import RotatedMnistDataset, ToTensor, Resize, getMnistPilThrees


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
    default=50,
    help="number of epoch by which a callback (plot + dump weights) is executed",
)
parser.add_option(
    "--epochs", dest="epochs", type=int, default=1000, help="total number of epochs"
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
vae_cfg = {"img_size": 32, "nf": opt.filts, "zdim": opt.zdim, "steps": 3, "colors": 1, "vy": opt.vy}
pickle.dump(vae_cfg, open(os.path.join(opt.outdir, "vae.cfg.p"), "wb"))


def normalize_tsfm(sample):
    sample['image'] = sample['image'] / 255.0
    return sample


def main():

    #torch.manual_seed(opt.seed)

    if opt.debug:
        pdb.set_trace()

    # make images 32x32 so they are a power of 2 so that vae conv sizes work
    # then convert PIL image to tensor
    # then normalize values from [0,255] to [0,1]
    transform = torchvision.transforms.Compose([
        Resize((32, 32)),
        ToTensor(),
        torchvision.transforms.Lambda(normalize_tsfm)
    ])

    train_pil_ims = getMnistPilThrees(root_dir=opt.data, start_ix=0, end_ix=400)
    # test_pil_ims = getMnistPilThrees(root_dir=opt.data, start_ix=400, end_ix=500)
    valid_pil_ims = getMnistPilThrees(root_dir=opt.data, start_ix=500, end_ix=600)
    train_data = RotatedMnistDataset(train_pil_ims, transform=transform)
    # test_data = RotatedMnistDataset(test_pil_ims, transform=transform)
    valid_data = RotatedMnistDataset(valid_pil_ims, transform=transform)
    train_queue = DataLoader(train_data, batch_size=opt.bs, shuffle=True)
    # test_queue = DataLoader(test_data, batch_size=opt.bs, shuffle=False)
    valid_queue = DataLoader(valid_data, batch_size=opt.bs, shuffle=False)
    Ntrain = len(train_data)
    Nvalid = len(valid_data)

    # define VAE and optimizer
    vae = RotatedMnistVAE(**vae_cfg).to(device)

    # optimizer
    optimizer = optim.Adam(vae.parameters(), lr=opt.lr, eps=1e-3)

    # load data

    history = {}
    for epoch in range(opt.epochs):

        # train and eval
        ht = train_ep(vae, train_queue, optimizer, Ntrain)
        hv = eval_ep(vae, valid_queue, Nvalid)
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


def train_ep(vae, train_queue, optimizer, Ntrain):

    rv = {}
    vae.train()

    for batch_i, data in enumerate(train_queue):
        # print("batch")
        y = data['image']

        # forward
        y = y.unsqueeze(dim=1) # make bsx28x28 into bsx1x28x28 for compatibility with VAE (colour channel=1)
        eps = Variable(torch.randn(y.shape[0], opt.zdim), requires_grad=False)
        y, eps = y.to(device), eps.to(device)
        elbo, mse, nll, kld = vae.forward(y, eps)
        loss = elbo.sum()

        # back propagate
        optimizer.zero_grad()
        loss.backward()

        # print("w_back1 is nan?=", torch.isnan(vae.econv[0].conv1.weight).sum()) # NaN via backprop
        # print("max weight=", torch.max(vae.econv[0].conv1.weight)) # are the weights exploding?
        # print("max   grad=", torch.max(vae.econv[0].conv1.weight.grad)) # are the weights exploding?
        # print max weight and max grad for each layer from final to first
        # if gradient explosion, should get progressively bigger until NaNs
        # print("loss=", loss)
        # print("decoder (max_weight, max_grad)")
        # for layer in reversed(vae.dconv):
        #     print(torch.max(layer.conv1.weight), torch.max(layer.conv1.weight.grad))
        # print(torch.max(vae.dense_dec.weight), torch.max(vae.dense_dec.weight.grad))
        # print("encoder (max_weight, max_grad)")
        # print(torch.max(vae.dense_zm.weight), torch.max(vae.dense_zm.weight.grad))
        # print(torch.max(vae.dense_zs.weight), torch.max(vae.dense_zs.weight.grad))
        # for layer in reversed(vae.econv):
        #     print(torch.max(layer.conv1.weight), torch.max(layer.conv1.weight.grad))

        # print("total econv[0].conv1.weight nans =", torch.isnan(vae.econv[0].conv1.weight))
        #torch.nn.utils.clip_grad_norm_(vae.parameters(), 100.0)
        optimizer.step()
        # print("w_back2 is nan?=", torch.isnan(vae.econv[0].conv1.weight).sum()) # NaN via weight update

        # sum metrics
        # print("mse=", mse.data.sum().cpu())
        smartSum(rv, "mse", float(mse.data.sum().cpu()) / float(Ntrain))
        smartSum(rv, "nll", float(nll.data.sum().cpu()) / float(Ntrain))
        smartSum(rv, "kld", float(kld.data.sum().cpu()) / float(Ntrain))
        smartSum(rv, "loss", float(elbo.data.sum().cpu()) / float(Ntrain))

    return rv


def eval_ep(vae, val_queue, Nvalid):
    rv = {}
    vae.eval()

    with torch.no_grad():

        for batch_i, data in enumerate(val_queue):

            # forward
            y = data['image'].unsqueeze(dim=1)
            eps = Variable(torch.randn(y.shape[0], opt.zdim), requires_grad=False)
            y, eps = y.to(device), eps.to(device)
            elbo, mse, nll, kld = vae.forward(y, eps)

            # sum metrics
            smartSum(rv, "mse_val", float(mse.data.sum().cpu()) / float(Nvalid))
            smartSum(rv, "nll_val", float(nll.data.sum().cpu()) / float(Nvalid))
            smartSum(rv, "kld_val", float(kld.data.sum().cpu()) / float(Nvalid))
            smartSum(rv, "loss_val", float(elbo.data.sum().cpu()) / float(Nvalid))

    return rv


if __name__ == "__main__":
    main()
