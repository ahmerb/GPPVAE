#!/bin/bash

#SBATCH -t 0-02:00 # Runtime in D-HH:MM
#SBATCH -p gpu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=4G
#SBATCH --account=comsm0018       # use the course account
#SBATCH -J gppvae
#SBATCH -o gppvae_%N_%j.out # File to which STDOUT will be written
#SBATCH -e gppvae_%N_%j.err # File to which STDERR will be written

module add languages/anaconda3/3.5-4.2.0-tflow-1.7
#module add apps/torch/28.01.2019

echo "install pytorch"

pip install torch torchvision --user --upgrade

echo "train vae"

# python train_vae.py --outdir ./out/vae --epochs 1000 --epoch_cb 100

echo "train gppvae"

python train_gppvae.py --outdir ./out/gppvae --vae_cfg ./out/vae/vae.cfg.p --vae_weights ./out/vae/weights/weights.00900.pt --epoch_cb 100 --epochs 5000
