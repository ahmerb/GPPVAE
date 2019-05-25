import sys, os

sys.path.insert(1, os.path.join(sys.path[0], '../..'))

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import scipy as sp
import matplotlib

import pdb
import logging
from optparse import OptionParser
import pickle

from train.faceplace.utils import smartSum, smartAppendDict, smartAppend, export_scripts
from train.faceplace.callbacks import callback_svi_gppvae, save_history

from models.face_vae import FaceVAE
from models.unobserved_feature_vectors import UnobservedFeatureVectors
from kernels.kernels import RotationKernel, KernelComposer

from train.faceplace.data_parser import read_face_data, FaceDataset

import gpytorch
from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import WhitenedVariationalStrategy

parser = OptionParser()
parser.add_option(
    "--data",
    dest="data",
    type=str,
    default="../../../data/data_faces.h5",
    help="dataset path",
)
parser.add_option(
    "--outdir", dest="outdir", type=str, default="./out/svi_gppvae", help="output dir"
)
parser.add_option("--vae_cfg", dest="vae_cfg", type=str, default=None)
parser.add_option("--vae_weights", dest="vae_weights", type=str, default=None)
parser.add_option("--seed", dest="seed", type=int, default=0, help="seed")
parser.add_option(
    "--vae_lr",
    dest="vae_lr",
    type=float,
    default=1e-3,
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

parser.add_option("--train_unison", action="store_true", dest="train_unison", default=False)
# only use below options if train_unison is True
parser.add_option(
    "--filts", dest="filts", type=int, default=32, help="number of convol filters"
)
parser.add_option("--zdim", dest="zdim", type=int, default=256, help="zdim")
parser.add_option(
    "--vy", dest="vy", type=float, default=2e-3, help="conditional norm lik variance"
)
(opt, args) = parser.parse_args()
opt_dict = vars(opt)

# parse args

vae_cfg = None
if opt.train_unison:
    vae_cfg = {"nf": opt.filts, "zdim": opt.zdim, "vy": opt.vy}
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


class SVGPRegression(AbstractVariationalGP):
    """
    Creates x and w tensors of shape num_outputs x n x d, and passes these to GPyTorch.
    z_dim is dimension of GP output.
    """
    def __init__(self, inducing_points, x_dim, w_dim, z_dim):
        """inducing points is tensor size: M x (x_dim+w_dim)"""
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0), batch_size=z_dim)
        variational_strategy = WhitenedVariationalStrategy(self, inducing_points, variational_distribution,
                                                           learn_inducing_locations=True)
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
        # keep external link to obj and view covar's
        self.Kxx = None
        self.Kww = None

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
        self.Kxx = self.obj_covar_module(x)
        self.Kww = self.view_covar_module(w)
        covar = self.Kxx * self.Kww # z_dim x n x n
        dist = gpytorch.distributions.MultivariateNormal(mean, covar)
        return dist


def main():
    torch.manual_seed(opt.seed)

    if opt.debug:
        pdb.set_trace()

    # load data
    img, obj, view = read_face_data(opt.data)  # image, object, and view
    train_data = FaceDataset(img["train"], obj["train"], view["train"])
    valid_data = FaceDataset(img["val"], obj["val"], view["val"])

    # iterating over a queue gives data minibatches that look like:
    # data = [ imgs, objs, views, indices of corresponding data ]
    train_queue = DataLoader(train_data, batch_size=opt.bs, shuffle=True)
    valid_queue = DataLoader(valid_data, batch_size=opt.bs, shuffle=False)

    # convert images and auxillary data (obj and view id's) to long tensors
    Dt = obj["train"][:, 0].long()
    Wt = view["train"][:, 0].long()
    Dv = obj["val"][:, 0].long()
    Wv = view["val"][:, 0].long()
    Ntrain = len(Dt)
    Nvalid = len(Dv)

    # number of unique object and views
    Xuniq = sp.unique(obj["train"])
    Wuniq = sp.unique(view["train"])
    P = Xuniq.shape[0]
    Q = Wuniq.shape[0]

    # we have sparse gp's, so we now can use arbitrarily large object feature vec size (before it was limited)
    x_dim = opt.xdim
    w_dim = Q
    x_features = UnobservedFeatureVectors(Dt, P, x_dim).to(device)
    w_features = UnobservedFeatureVectors(Wt, Q, w_dim).to(device)

    # define VAE and optimizer
    vae = FaceVAE(**vae_cfg).to(device)
    if not opt.train_unison:
        RV = torch.load(opt.vae_weights, map_location=device) # remove map_location when using gpu
        vae.load_state_dict(RV)
        vae.to(device)

    # define gp

    # init inducing inputs
    M = 50 # num inducing points
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

    gp_mll = gpytorch.mlls.VariationalELBO(gp_likelihood, gp_model, num_data=Dt.size(0), combine_terms=True).to(device)

    history = {}
    for epoch in range(opt.epochs):
        print('epoch start')
        ht = {}
        for batch_i, data in enumerate(train_queue):
            y, x, w, idxs = data
            y = y.to(device) # bs x 3 x 128 x 128
            x = x.to(device) # bs x x_dim
            w = w.to(device) # bs x w_dim
            #idxs = idxs.to(device)

            # vae
            zm, zs = vae.encode(y)
            eps = torch.normal(mean=torch.zeros(y.size(0), z_dim), std=torch.ones(y.size(0), z_dim)).to(device)
            z = zm + zs * eps
            yr = vae.decode(z)
            recon_term, mse = vae.nll(y, yr)

            # gp
            x_ = x_features(x[:, 0].long()) # bs x x_dim
            w_ = w_features(w[:, 0].long()) # bs x w_dim
            gp_inp = torch.cat((x_, w_), 1) # bs x (x_dim+w_dim)
            gp_likelihood_dist = gp_model(gp_inp)
            gp_mll_term = -gp_mll(gp_likelihood_dist, z.t()) / vae.K # z.t() is z_dim x bs

            # penalization
            pen_term = -0.5 * zs.sum() / vae.K

            # optim step
            vae_optim.zero_grad()
            gp_optim.zero_grad()
            loss = recon_term.sum() + gp_mll_term.sum() + pen_term.sum().to(device)
            logging.info('Epoch %d [%d/%d] - Loss: %.3f [%.3f, %.3f, %.3f]' % (epoch + 1, batch_i, len(train_queue), loss.item(), recon_term.sum().item(), gp_mll_term.sum().item(), pen_term.item()))
            loss.backward(retain_graph=True)
            vae_optim.step()
            gp_optim.step()

            # logging etc
            _n = train_queue.dataset.Y.shape[0]
            smartSum(ht, "mse", float(mse.data.sum().cpu()) / _n)
            smartSum(ht, "recon_term", float(recon_term.data.sum().cpu()) / _n)
            smartSum(ht, "pen_term", float(pen_term.data.sum().cpu() / _n))
            smartSum(ht, "gp_nll", float(gp_mll_term.data.sum().cpu()))
            smartSum(ht, "loss", loss.data.cpu().item())
        smartAppendDict(history, ht)

        logging.info("Epoch %d - complete" % (epoch + 1))
        if epoch % opt.epoch_cb == 0:
            hv, imgs = evaluate_gppvae(vae, gp_model, gp_likelihood, x_features, w_features, valid_queue, Nvalid, epoch,
                                       device)
            smartAppendDict(history, hv)
            logging.info("Epoch %d - executing callback" % (epoch + 1))
            w_kernel = gp_model.view_covar_module
            x_kernel = gp_model.obj_covar_module
            covs = compute_covs(Xuniq, Wuniq, Wu, Xu, w_kernel, x_kernel, device)
            vae_wfile = os.path.join(wdir, "weights.%.5d.pt" % epoch)
            gp_wfile = os.path.join(gp_wdir, "gp_weights.%5d.pt" % epoch)
            x_wfile = os.path.join(gp_wdir, "x_weights.%5d.pt" % epoch)
            w_wfile = os.path.join(gp_wdir, "w_weights.%5d.pt" % epoch)
            ffile = os.path.join(fdir, "plot.%.5d.png" % epoch)
            torch.save(vae.state_dict(), vae_wfile)
            torch.save(gp_model.state_dict(), gp_wfile)
            torch.save(x_features.state_dict(), x_wfile)
            torch.save(w_features.state_dict(), w_wfile)
            callback_svi_gppvae(epoch, history, covs, imgs, ffile)
        save_history(history, hdir, pickle=True)


def compute_covs(Xuniq, Wuniq, Wu, Xu, w_kernel, x_kernel, device):
    with torch.no_grad():
        # we have zdim=256 different covar matrices (one gp per output, so different hyperparams used)
        # just use the first
        # Kobj = x_kernel(torch.tensor(Xuniq, device=device)).evaluate_kernel().evaluate().data.cpu().numpy()[0]
        Kview_uu = w_kernel(Wu).evaluate_kernel().evaluate().data.cpu().numpy()[0]
        Kobj_uu = x_kernel(Xu).evaluate_kernel().evaluate().data.cpu().numpy()[0]
        return {"K": Kview_uu, "Kuu": Kobj_uu}


def evaluate_gppvae(vae, gp_model, gp_likelihood, x_features, w_features, valid_queue, Nvalid, epoch, device):
    hv = {}
    imgs = {}

    vae.eval()
    gp_model.eval()
    gp_likelihood.eval()
    x_features.eval()
    w_features.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples():
        for batch_i, data in enumerate(valid_queue):
            # print("eval minibatch")
            y_test, x_test, w_test, idxs = data
            y_test, x_test, w_test = y_test.to(device), x_test.to(device), w_test.to(device)

            # retrieve feature vectors
            x_ = x_features(x_test[:, 0].long())
            w_ = w_features(w_test[:, 0].long())
            gp_inp = torch.cat((x_, w_), 1) # bs x (x_dim+w_dim)

            # gp posterior
            z_test_dist = gp_model(gp_inp)
            z_test_mu = z_test_dist.mean.t()

            # decode to get reconstructions
            y_test_recon = vae.decode(z_test_mu)

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
