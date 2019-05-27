import matplotlib

import torch
from torch import optim
from torch.utils.data import DataLoader
import torchvision

import sys
import os

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import pdb
import logging
from optparse import OptionParser
import pickle

from models.rotated_mnist_vae import RotatedMnistVAE
from models.gp.dual_input_sparse_gp import DualInputSparseGPRegression
from kernels.kernels import RotationKernel, SEKernel, KernelComposer

from train.rotated_mnist_fitc.utils import smartSum, smartAppendDict, export_scripts
from train.rotated_mnist_fitc.callbacks import callback_gppvae, save_history

from train.rotated_mnist_fitc.rotated_mnist import RotatedMnistDataset, ToTensor, Resize, getMnistPilThrees


parser = OptionParser()
parser.add_option(
    "--data",
    dest="data",
    type=str,
    default='../../../mnist_data',
    help="dataset path",
)
parser.add_option(
    "--outdir", dest="outdir", type=str, default="./out/fitc_gppvae", help="output dir"
)
parser.add_option("--vae_cfg", dest="vae_cfg", type=str, default="./out/vae/vae.cfg.p")
parser.add_option("--vae_weights", dest="vae_weights", type=str, default="./out/vae/weights/weights.00950.pt")
parser.add_option("--seed", dest="seed", type=int, default=0, help="seed")
parser.add_option(
    "--vae_lr",
    dest="vae_lr",
    type=float,
    default=0.0003,
    help="learning rate of vae params",
)
parser.add_option(
    "--gp_lr", dest="gp_lr", type=float, default=3e-3, help="learning rate of gp params"
)
parser.add_option(
    "--xdim", dest="xdim", type=int, default=1, help="rank of object linear covariance"
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
    "--epochs", dest="epochs", type=int, default=500, help="total number of epochs"
)
parser.add_option("--debug", action="store_true", dest="debug", default=False)
parser.add_option("--train_unison", action="store_true", dest="train_unison", default=False)
# only use below options if train_unison is True
parser.add_option(
    "--filts", dest="filts", type=int, default=8, help="number of convol filters"
)
parser.add_option("--zdim", dest="zdim", type=int, default=16, help="zdim")
parser.add_option(
    "--vy", dest="vy", type=float, default=2e-3, help="conditional norm lik variance"
)
(opt, args) = parser.parse_args()
opt_dict = vars(opt)

# parse args

# VAE config
vae_cfg = None
if opt.train_unison:
    vae_cfg = {"img_size": 32, "nf": opt.filts, "zdim": opt.zdim, "steps": 3, "colors": 1, "vy": opt.vy}
    # pickle.dump(vae_cfg, open(os.path.join(opt.outdir, "vae_unison.cfg.p"), "wb"))
else:
    vae_cfg = pickle.load(open(opt.vae_cfg, "rb"))
z_dim = vae_cfg["zdim"]

# device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    matplotlib.use("Qt5Agg")

# output dir
if not os.path.exists(opt.outdir):
    os.makedirs(opt.outdir)
wdir = os.path.join(opt.outdir, "weights")
gp_wdir = os.path.join(opt.outdir, "gp_weights")
fdir = os.path.join(opt.outdir, "plots")
hdir = os.path.join(opt.outdir, "history")
if not os.path.exists(wdir):
    os.makedirs(wdir)
if not os.path.exists(gp_wdir):
    os.makedirs(gp_wdir)
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


def normalize_tsfm(sample):
    sample['image'] = sample['image'] / 255.0
    return sample


def main():
    torch.manual_seed(opt.seed)

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
    # test_pil_ims  = getMnistPilThrees(root_dir=opt.data, start_ix=400, end_ix=500)
    valid_pil_ims = getMnistPilThrees(root_dir=opt.data, start_ix=500, end_ix=600)
    train_data = RotatedMnistDataset(train_pil_ims, transform=transform)
    # test_data  = RotatedMnistDataset(test_pil_ims, transform=transform)
    valid_data = RotatedMnistDataset(valid_pil_ims, transform=transform)
    train_queue = DataLoader(train_data, batch_size=opt.bs, shuffle=True)
    # test_queue  = DataLoader(test_data, batch_size=opt.bs, shuffle=False)
    valid_queue = DataLoader(valid_data, batch_size=opt.bs, shuffle=False)
    Ntrain = len(train_data)
    Nvalid = len(valid_data)

    num_rotations = 16 # number of unique rotation angles

    # define VAE
    vae = RotatedMnistVAE(**vae_cfg).to(device)
    if not opt.train_unison:
        vae_state = torch.load(opt.vae_weights, map_location=device)
        vae.load_state_dict(vae_state)
        vae.to(device)

    # define GP

    # init view inducing points (view is full rank)
    M = num_rotations
    min_rot_angle = 0.0
    max_rot_angle = 337.5
    Wu = torch.linspace(min_rot_angle, max_rot_angle, M).to(device)

    # extract auxiliary data
    Wtrain = torch.tensor(
        list(map(lambda datapoint: float(datapoint[1]), train_data.data))
    ).to(device)
    Wvalid = torch.tensor(
        list(map(lambda datapoint: float(datapoint[1]), valid_data.data))
    ).to(device)

    # init object inducing points
    min_ix = 0.0
    max_ix = float(len(train_data.data))
    Xu = torch.linspace(min_ix, max_ix, M).to(device)

    # extract auxiliary data
    Xtrain = torch.tensor(
        list(map(lambda tmp: float(tmp[0]), enumerate(train_data.data)))
    ).to(device)
    Xvalid = torch.tensor(
        list(map(lambda tmp: float(tmp[0]), enumerate(valid_data.data)))
    ).to(device)

    # init gp
    x_kernel = SEKernel
    w_kernel = RotationKernel
    kernel_composer = KernelComposer.Product
    # gp_cfg = [Xtrain, Wtrain, None, x_kernel, w_kernel, kernel_composer, Xu, Wu]
    # pickle.dump(gp_cfg, open(os.path.join("./plot", "gp.cfg.p"), "wb"))
    # sys.exit(1)
    gp = DualInputSparseGPRegression(Xtrain, Wtrain, None, x_kernel, w_kernel, kernel_composer,
                                     Xu, Wu, wu_trainable=False).to(device)

    # optimizers
    vae_optimizer = optim.Adam(vae.parameters(), lr=opt.vae_lr)
    gp_optimizer = optim.Adam(gp.parameters(), lr=opt.gp_lr)

    history = {}
    for epoch in range(opt.epochs):
        vae_optimizer.zero_grad()
        gp_optimizer.zero_grad()
        vae.train()
        gp.train()

        ht = {}

        Zm = torch.zeros(Ntrain, z_dim, device="cpu")
        Zs = torch.zeros(Ntrain, z_dim, device="cpu")
        Z = torch.zeros(Ntrain, z_dim, device="cpu")
        Eps = torch.normal(mean=torch.zeros(Ntrain, z_dim, device='cpu'), std=torch.ones(Ntrain, z_dim, device='cpu'))

        recon_term = torch.zeros(opt.bs, 1, device=device)
        mse = torch.zeros(opt.bs, 1, device=device)

        # for each minibatch
        # print('vae train')
        for batch_i, data in enumerate(train_queue):
            # get data from minibatch
            idxs = data['index']
            y = data['image'].unsqueeze(dim=1).to(device) # size(bs, 1 32, 32)
            eps = Eps[idxs].to(device=device, copy=True)

            # forward encoder on minibatch
            zm, zs = vae.encode(y) # size(bs, zdim)

            # sample z's
            z = zm + zs * eps # size(bs, zdim)

            # forward decoder on minibatch
            yr = vae.decode(z) # size(bs, 1, 32, 32)

            # store z's of this minibatch
            Zm[idxs] = zm.detach().to('cpu')
            Zs[idxs] = zs.detach().to('cpu')
            Z[idxs] = z.detach().to('cpu')

            # compute and update mse and nll
            recon_term_batch, mse_batch = vae.nll(y, yr) # size(bs, 1)
            recon_term += recon_term_batch
            mse += mse_batch

            # if torch.cuda.is_available():
            #     print("memory usage: ", torch.cuda.max_memory_allocated())

        # XXX
        # tmp = Zm.numpy()
        # pickle.dump(tmp, open(os.path.join("./plot", "Zm.dump"), "wb"))
        # sys.exit(1)

        # forward gp (using FITC)
        # print('gp train')
        Z = Z.to(device)
        Z = Z - Z.mean()
        gp.y = Z
        gp_mll = gp() / z_dim # vae.K use zdim and see if this works better

        # penalization (compute the regularization term)
        pen_term = (0.5 * Zs.sum() / z_dim).to(device) # vae.K use zdim and see if this works better

        # loss and backprop
        loss = recon_term.sum() - gp_mll + pen_term.sum()
        loss.backward()

        vae_optimizer.step()
        gp_optimizer.step()

        smartSum(ht, "mse", float(mse.data.sum().cpu()) / float(Ntrain))
        smartSum(ht, "gp_nll", -float(gp_mll.data.cpu()) / float(Ntrain))
        smartSum(ht, "recon_term", float(recon_term.data.sum().cpu()) / float(Ntrain))
        smartSum(ht, "pen_term", float(pen_term.data.sum().cpu()) / float(Ntrain))
        smartSum(ht, "loss", float(loss.data.cpu()) / float(Ntrain))
        smartAppendDict(history, ht)

        # eval on validation set (using GPPVAE posterior predictive)
        if epoch % opt.epoch_cb == 0:
            hv, imgs = evaluate_gppvae(vae, gp, Zm, Xvalid, Wvalid, valid_queue, Nvalid, epoch, device)
            smartAppendDict(history, hv)

            logging.info(
                "epoch %d - train_mse: %f - test_mse %f" % (epoch, ht["mse"], hv["mse_val"])
            )

            # callbacks
            logging.info("epoch %d - executing callback" % epoch)
            wfile = os.path.join(wdir, "weights.%.5d.pt" % epoch)
            gp_wfile = os.path.join(gp_wdir, "gp_weights.%.5d.pt" % epoch)
            ffile = os.path.join(fdir, "plot.%.5d.png" % epoch)
            torch.save(vae.state_dict(), wfile)
            torch.save(gp.state_dict(), gp_wfile)
            covs = compute_covs(Wu, gp.w_kernel, gp.x_kernel, Xtrain, Wu, device)
            callback_gppvae(epoch, history, covs, imgs, ffile)
        else:
            logging.info(
                "epoch %d - train_mse: %f [%5f,%5f,%5f]" % (epoch, ht["mse"], ht["recon_term"], ht["gp_nll"], ht["pen_term"])
            )

    save_history(history, hdir, pickle=True)


def compute_covs(Wu, w_kernel, x_kernel, Xtrain, Xu, device):
    # print('compute covs')
    # NOTE this is all random crap rn
    with torch.no_grad():
        # covariance between all rotation angles
        Kview = w_kernel(Wu).data.cpu().numpy()

        # covariance between all Xu
        #Kobj = x_kernel(Xtrain).data.cpu().numpy()
        #K_uu = x_kernel(Xu).data.cpu().numpy()
        # print('done covs')
        return {"K": Kview, "Kuu": Kview}


def evaluate_gppvae(vae, gp, Zm, Xvalid, Wvalid, valid_queue, Nvalid, epoch, device):
    hv = {}
    imgs = {}

    vae.eval()
    gp.eval()

    with torch.no_grad():
        # gp posterior predictive
        # print('gp eval')
        Zm = Zm.to(device)
        gp.y = Zm - Zm.mean()
        z_test_mu = gp.posterior_predictive(Xvalid, Wvalid, compute_cov=False)

        # print('vae eval')
        for batch_i, data in enumerate(valid_queue):
            y_test = data['image'].unsqueeze(dim=1)
            idxs = data['index']
            y_test, idxs = y_test.to(device), idxs.to(device)

            # decode to get reconstructions
            y_test_recon = vae.decode(z_test_mu[idxs])

            # compute error
            recon_term, mse = vae.nll(y_test, y_test_recon)
            smartSum(hv, "mse_val", float(mse.data.sum().cpu()) / float(Nvalid))
            smartSum(hv, "recon_term_val", float(recon_term.data.sum().cpu()) / float(Nvalid))

            # store a few examples
            if batch_i == 0:
                imgs = {}
                imgs["Y"] = y_test[:24].data.cpu().numpy().transpose(0, 2, 3, 1)
                imgs["Yr"] = y_test_recon[:24].data.cpu().numpy().transpose(0, 2, 3, 1)

    return hv, imgs


if __name__ == "__main__":
    main()
