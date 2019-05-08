from optparse import OptionParser

import os
import os.path
import logging
import sys

from callbacks import callback_gppvae
from utils import smartSum, smartAppendDict, smartAppend, export_scripts

import torch
from torch.utils.data import DataLoader

import matplotlib


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

# minibatch
# cuda
# use unobserved feature vectors
# use vae weights

# DATASETS
# FacePlace v3
# Rotated MNIST

# specify which model and dataset to use
# calls to generic interface and runs training loop


# choose device
device = torch.device('cpu')
if opt.enable_cuda and torch.cuda.is_available():
    device = torch.device('cuda:0')

# if on local (no cuda) need to use Qt5Agg because my Tk doesn't work
if not opt.enable_cuda:
    matplotlib.use("Qt5Agg")

# output dir
if not os.path.exists(opt.outdir):
    os.makedirs(opt.outdir)

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


def getModelTrainer(model):
    if model == "vae_only":
        pass
    elif model == "gppvae_dis_taylor":
        pass
    elif model == "gppvae_joint_taylor":
        pass
    elif model == "gppvae_both_taylor":
        pass
    elif model == "gppvae_dis_fitc":
        pass
    elif model == "gppvae_joint_fitc":
        pass
    elif model == "gppvae_both_fitc":
        pass
    elif model == "gppvae_dis_vfe":
        pass
    elif model == "gppvae_joint_vfe":
        pass
    elif model == "gppvae_both_vfe":
        pass
    elif model == "gppvae_dis_svgp":
        pass
    elif model == "gppvae_joint_svgp":
        pass
    elif model == "gppvae_both_svgp":
        pass
    else:
        raise TypeError("Not a valid model")
    return {}


def getDataset(dataset):
    if dataset == "faceplace3":
        pass
    elif dataset == "rotated_mnist":
        pass
    return {}


def main():
    torch.manual_seed(opt.seed)

    trainer = getModelTrainer(opt.model)
    dataset = getDataset(opt.dataset)
    dataset.init()
    trainer.init(dataset=dataset)

    train_data = dataset.getTrainData()
    val_data = dataset.getValidationData()

    train_queue = DataLoader(train_data, batch_size=opt.bs, shuffle=True)
    val_queue = DataLoader(val_data, batch_size=opt.bs, shuffle=False)

    aux_features = trainer.createAuxiliaryFeatures()
