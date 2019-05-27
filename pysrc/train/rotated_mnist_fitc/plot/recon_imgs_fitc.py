import sys, os

sys.path.insert(1, os.path.join(sys.path[0], '../../..'))


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
from models.rotated_mnist_vae import RotatedMnistVAE
from models.gp.dual_input_sparse_gp import DualInputSparseGPRegression
from kernels.kernels import RotationKernel, SEKernel, KernelComposer
from train.rotated_mnist_fitc.callbacks import _compose_multi
from train.rotated_mnist_fitc.rotated_mnist import RotatedMnistDataset, ToTensor, Resize, getMnistPilThrees
import torchvision
from torch.utils.data import DataLoader


matplotlib.use("Qt5Agg")
matplotlib.rcParams['savefig.pad_inches'] = 0


def normalize_tsfm(sample):
    sample['image'] = sample['image'] / 255.0
    return sample


def get_valid_queue(bs=16):
    transform = torchvision.transforms.Compose([
        Resize((32, 32)),
        ToTensor(),
        torchvision.transforms.Lambda(normalize_tsfm)
    ])
    data_dir = "../../../../mnist_data"
    valid_pil_ims = getMnistPilThrees(root_dir=data_dir, start_ix=500, end_ix=600)
    valid_data = RotatedMnistDataset(valid_pil_ims, transform=transform)
    valid_queue = DataLoader(valid_data, batch_size=bs, shuffle=False)
    Nvalid = len(valid_data)
    Xvalid = torch.tensor(
        list(map(lambda tmp: float(tmp[0]), enumerate(valid_data.data)))
    )
    Wvalid = torch.tensor(
        list(map(lambda datapoint: float(datapoint[1]), valid_data.data))
    )
    return valid_data, valid_queue, Nvalid, Xvalid, Wvalid


def plot_images(imgs, filename):
    Y, Yr, Yr_vae = imgs["Y"], imgs["Yr"], imgs["Yr_vae"]
    _img = _compose_multi([Y, Yr_vae, Yr])
    fig, ax = plt.subplots(figsize=plt.figaspect(_img))
    fig.subplots_adjust(0, 0, 1, 1) # remove whitespace around ax
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # ax.grid(which="major", axis="both")
    plt.autoscale(tight=True)
    plt.imshow(_img.squeeze())
    plt.savefig(filename)
    plt.close()


def main():
    # load fitc-gppvae-unison VAE weights
    vae_weights_path = "../out/fitc_gppvae_unison_old_fail/weights/weights.02950.pt"
    vae_cfg_path = "../out/fitc_gppvae_unison_old_fail/vae_unison.cfg.p"
    vae_cfg = pickle.load(open(vae_cfg_path, "rb"))
    device = "cpu"
    vae = RotatedMnistVAE(**vae_cfg).to(device)
    RV = torch.load(vae_weights_path, map_location=device) # remove map_location when using gpu
    vae.load_state_dict(RV)
    vae.to(device)

    # load fitc-gppvae-unison GP weights
    gp_weights_path = "./gp_tmp_weights.pt"
    gp_cfg_path = "./gp.cfg.p"
    gp_cfg = pickle.load(open(gp_cfg_path, "rb"))
    gp = DualInputSparseGPRegression(*gp_cfg, wu_trainable=False).to(device)
    RV = torch.load(gp_weights_path, map_location=device)
    gp.load_state_dict(RV)
    gp.to(device)

    #######
    # GPPVAE PRED POST
    #######

    # get valid data
    valid_data, valid_queue, Nvalid, Xvalid, Wvalid = get_valid_queue()

    # load Zm (Z means from encoder on all train data)
    Zm = torch.tensor(pickle.load(open("./Zm.dump", "rb"))).to(device)

    # get Zm and put in GP
    gp.y = Zm - Zm.mean()

    # run GPPVAE pred post on first valid batch
    result = {}
    z_test_mu = gp.posterior_predictive(Xvalid, Wvalid, compute_cov=False)
    data = next(valid_queue.__iter__())
    y_test = data['image'].unsqueeze(dim=1)
    idxs = data['index']
    y_test, idxs = y_test.to(device), idxs.to(device)
    y_test_recon = vae.decode(z_test_mu[idxs])

    # compute error
    recon_term, mse = vae.nll(y_test, y_test_recon)
    result["recon_term"] = recon_term
    result["mse"] = mse

    # store examples
    result["Y"] = y_test.data.cpu().numpy().transpose(0, 2, 3, 1)
    result["Yr"] = y_test_recon.data.cpu().numpy().transpose(0, 2, 3, 1)

    #######
    # VAE ONLY RECON
    #######
    zm, zs = vae.encode(y_test)
    eps = torch.randn(zm.size())
    z = zm + eps * zs
    y_test_recon_vae = vae.decode(z)

    # store examples
    result["Yr_vae"] = y_test_recon_vae.data.cpu().numpy().transpose(0, 2, 3, 1)

    # plot images
    plot_images(result, "./fitc_gppvae_unison_recon_imgs.eps")


if __name__ == "__main__":
    with torch.no_grad():
        main()
