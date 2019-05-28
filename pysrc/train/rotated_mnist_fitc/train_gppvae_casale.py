import sys, os

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import matplotlib
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models.rotated_mnist_vae import RotatedMnistVAE
from train.rotated_mnist_fitc.vmod import Vmodel
from train.faceplace.gp import GP
import scipy as sp
import pdb
import logging
from train.rotated_mnist_fitc.utils import smartSum, smartAppendDict, smartAppend, export_scripts
from train.rotated_mnist_fitc.callbacks import callback_casale_gppvae, save_history
from optparse import OptionParser
import pickle
from train.rotated_mnist_fitc.rotated_mnist import RotatedMnistDataset, ToTensor, Resize, getMnistPilThrees
import torchvision

parser = OptionParser()
parser.add_option(
    "--data",
    dest="data",
    type=str,
    default='../../../mnist_data',
    help="dataset path",
)
parser.add_option(
    "--outdir", dest="outdir", type=str, default="./out/gppvae_unison", help="output dir"
)
parser.add_option("--vae_cfg", dest="vae_cfg", type=str, default="./out/vae/vae.cfg.p")
parser.add_option("--vae_weights", dest="vae_weights", type=str, default="./out/vae/weights/weights.02500.pt")
parser.add_option("--seed", dest="seed", type=int, default=0, help="seed")
parser.add_option(
    "--vae_lr",
    dest="vae_lr",
    type=float,
    #default=3e-4,
    default=2e-4,
    help="learning rate of vae params",
)
parser.add_option(
    "--gp_lr", dest="gp_lr", type=float, default=1e-2, help="learning rate of gp params"
)
parser.add_option(
    "--xdim", dest="xdim", type=int, default=64, help="rank of object linear covariance"
)
parser.add_option("--bs", dest="bs", type=int, default=64, help="batch size")
parser.add_option(
    "--epoch_cb",
    dest="epoch_cb",
    type=int,
    default=50,
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


# vae config
vae_cfg = None
if opt.train_unison:
    vae_cfg = {"img_size": 32, "nf": opt.filts, "zdim": opt.zdim, "steps": 3, "colors": 1, "vy": opt.vy}
    pickle.dump(vae_cfg, open(os.path.join(opt.outdir, "vae_unison.cfg.p"), "wb"))
else:
    vae_cfg = pickle.load(open(opt.vae_cfg, "rb"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    matplotlib.use("Qt5Agg")

if not os.path.exists(opt.outdir):
    os.makedirs(opt.outdir)

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

    # extract auxiliary data
    Wt = torch.tensor(
        list(map(lambda datapoint: float(datapoint[1]), train_data.data))
    ).unsqueeze(dim=1).to(device)
    Wv = torch.tensor(
        list(map(lambda datapoint: float(datapoint[1]), valid_data.data))
    ).unsqueeze(dim=1).to(device)

    # extract auxiliary data
    Dt = torch.tensor(
        list(map(lambda tmp: float(tmp[0]), enumerate(train_data.data)))
    ).unsqueeze(dim=1).to(device)
    Dv = torch.tensor(
        list(map(lambda tmp: float(tmp[0]), enumerate(valid_data.data)))
    ).unsqueeze(dim=1).to(device)
    print(Dt.shape)

    # define VAE and optimizer
    vae = RotatedMnistVAE(**vae_cfg).to(device)
    if not opt.train_unison:
        RV = torch.load(opt.vae_weights, map_location=device) # remove map_location when using gpu
        vae.load_state_dict(RV)
        vae.to(device)

    # define gp
    P = Dt.shape[0] # number unique obj's
    Q = sp.unique(num_rotations).shape[0] # number unique views
    vm = Vmodel().to(device) # low-rank approx
    gp = GP(n_rand_effs=1).to(device)
    gp_params = nn.ParameterList()
    # gp_params.extend(vm.parameters())
    gp_params.extend(gp.parameters())

    # define optimizers
    vae_optimizer = optim.Adam(vae.parameters(), lr=opt.vae_lr)
    gp_optimizer = optim.Adam(gp_params, lr=opt.gp_lr)

    if opt.debug:
        pdb.set_trace()

    history = {}
    for epoch in range(opt.epochs):

        # 1. encode Y in mini-batches
        Zm, Zs = encode_Y(vae, train_queue, Ntrain) # gets encodings (distrib params q(z|y=y)) for entire dataset

        # 2. sample Z
        # sample a z for each encoding (zm,zs) above
        Eps = Variable(torch.randn(*Zs.shape), requires_grad=False).to(device)
        Z = Zm + Eps * Zs

        # 3. evaluation step (not needed for training)
        # run Vmodel on object and view training ids (entire training data?) to give us low-rank approx V for kernel K
        Vt = vm(Dt, Wt).detach() # Dt is training obj ids, Wt is training view ids. Vt is V in K=V*V^t+alpha*I (eqn 20 in paper), i.e. low rank aproximation for kernel

        Vv = vm(Dv, Wv).detach() # Dv is validation obj ids, Wv is validation view ids.
        rv_eval, imgs, covs = eval_step(vae, gp, vm, valid_queue, Zm, Vt, Vv, Dt, Wt)

        # 4. compute first-order Taylor expansion coefficient
        # evaluate a, B, c across all samples (which we used for Vt)?
        Zb, Vbs, vbs, gp_nll = gp.taylor_coeff(Z, [Vt])
        rv_eval["gp_nll"] = float(gp_nll.data.mean().cpu()) / vae.K

        # 5. accumulate gradients over mini-batches and update params
        # use taylor series approx for the gp loss
        rv_back = backprop_and_update(
            vae,
            gp,
            vm,
            train_queue,
            Dt,
            Wt,
            Eps,
            Zb,
            Vbs,
            vbs,
            vae_optimizer,
            gp_optimizer,
            Ntrain
        )
        rv_back["loss"] = (
            rv_back["recon_term"] + rv_eval["gp_nll"] + rv_back["pen_term"]
        )

        smartAppendDict(history, rv_eval)
        smartAppendDict(history, rv_back)
        smartAppend(history, "vs", gp.get_vs().data.cpu().numpy())

        logging.info(
            "epoch %d - tra_mse_val: %f - train_mse_out: %f"
            % (epoch, rv_eval["mse_val"], rv_eval["mse_out"])
        )

        # callback?
        if epoch % opt.epoch_cb == 0:
            logging.info("epoch %d - executing callback" % epoch)
            ffile = os.path.join(fdir, "plot.%.5d.png" % epoch)
            callback_casale_gppvae(epoch, history, covs, imgs, ffile)

    save_history(history, hdir, pickle=True)


def encode_Y(vae, train_queue, Ntrain):
    # finds encoding of every single datapoint
    # does forward encoding passes in minibatches

    vae.eval()

    with torch.no_grad():

        n = Ntrain
        Zm = Variable(torch.zeros(n, vae_cfg["zdim"]), requires_grad=False).to(device)
        Zs = Variable(torch.zeros(n, vae_cfg["zdim"]), requires_grad=False).to(device)

        for batch_i, data in enumerate(train_queue):
            # data = [ imgs, objs, views, indices of corresponding data ]
            y = data['image'].unsqueeze(dim=1).to(device)
            idxs = data['index'].to(device)
            zm, zs = vae.encode(y)
            Zm[idxs], Zs[idxs] = zm.detach(), zs.detach()

    return Zm, Zs


def eval_step(vae, gp, vm, val_queue, Zm, Vt, Vv, Dt, Wt):

    rv = {}

    with torch.no_grad():

        _X = Dt.data.cpu().numpy()
        _W = Wt.data.cpu().numpy()
        covs = {"XX": sp.dot(_X, _X.T), "WW": sp.dot(_W, _W.T)}
        rv["vars"] = gp.get_vs().data.cpu().numpy()
        # out of sample
        vs = gp.get_vs()
        U, UBi, _ = gp.U_UBi_Shb([Vt], vs)
        Kiz = gp.solve(Zm, U, UBi, vs)
        Zo = vs[0] * Vv.mm(Vt.transpose(0, 1).mm(Kiz))
        mse_out = Variable(torch.zeros(Vv.shape[0], 1), requires_grad=False).to(device)
        mse_val = Variable(torch.zeros(Vv.shape[0], 1), requires_grad=False).to(device)
        for batch_i, data in enumerate(val_queue):
            idxs = data['index'].to(device)
            Yv = data['image'].unsqueeze(dim=1).to(device)
            Zv = vae.encode(Yv)[0].detach() # gets the mean (ignored z_sigma output)
            Yr = vae.decode(Zv)
            Yo = vae.decode(Zo[idxs])
            mse_out[idxs] = (
                ((Yv - Yo) ** 2).view(Yv.shape[0], -1).mean(1)[:, None].detach()
            )
            mse_val[idxs] = (
                ((Yv - Yr) ** 2).view(Yv.shape[0], -1).mean(1)[:, None].detach()
            )
            # store a few examples
            if batch_i == 0:
                imgs = {}
                imgs["Yv"] = Yv[:24].data.cpu().numpy().transpose(0, 2, 3, 1) # ORIGINAL IMAGE
                imgs["Yr"] = Yr[:24].data.cpu().numpy().transpose(0, 2, 3, 1) # VAE RECON
                imgs["Yo"] = Yo[:24].data.cpu().numpy().transpose(0, 2, 3, 1) # GPPVAE RECON
        rv["mse_out"] = float(mse_out.data.mean().cpu())
        rv["mse_val"] = float(mse_val.data.mean().cpu())

    return rv, imgs, covs


def backprop_and_update(
    vae, gp, vm, train_queue, Dt, Wt, Eps, Zb, Vbs, vbs, vae_optimizer, gp_optimizer, Ntrain
):

    rv = {}

    vae_optimizer.zero_grad()
    gp_optimizer.zero_grad()
    vae.train()
    gp.train()
    vm.train()

    # for each minibatch
    for batch_i, data in enumerate(train_queue):

        # subset data: (data[-1] gives indices of datapoints in this minibatch)
        y = data['image'].unsqueeze(dim=1).to(device)
        eps = Eps[data['index']]
        _d = Dt[data['index']]
        _w = Wt[data['index']]
        _Zb = Zb[data['index']]
        _Vbs = [Vbs[0][data['index']]]

        # forward vae
        # computes reconstruction loss (distance between true image and output of decoder)
        zm, zs = vae.encode(y)
        z = zm + zs * eps
        yr = vae.decode(z)
        recon_term, mse = vae.nll(y, yr)

        # forward gp
        # use approx V and compute Taylor expansion to find approx for gp_nll
        _Vs = [vm(_d, _w)]
        gp_nll_fo = gp.taylor_expansion(z, _Vs, _Zb, _Vbs, vbs) / vae.K

        # penalization
        # compute the regularization term
        pen_term = -0.5 * zs.sum(1)[:, None] / vae.K

        # loss and backward
        loss = (recon_term + gp_nll_fo + pen_term).sum()
        loss.backward()

        # store stuff
        _n = Ntrain
        smartSum(rv, "mse", float(mse.data.sum().cpu()) / _n)
        smartSum(rv, "recon_term", float(recon_term.data.sum().cpu()) / _n)
        smartSum(rv, "pen_term", float(pen_term.data.sum().cpu()) / _n)

    vae_optimizer.step()
    gp_optimizer.step()

    return rv


if __name__ == "__main__":
    main()
