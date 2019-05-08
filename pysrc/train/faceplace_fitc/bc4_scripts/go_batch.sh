#!/bin/bash

#SBATCH -t 0-00:30 # Runtime in D-HH:MM
#SBATCH -p gpu # Partition to submit to
#SBATCH --gres=gpu:2
#SBATCH --mem=4G
#SBATCH -J gppvae
#SBATCH -o gppvae_%N_%j.out # File to which STDOUT will be written
#SBATCH -e gppvae_%N_%j.err # File to which STDERR will be written

#####(disabled) --account=comsm0018       # use the course account

# song said project code is COSC019002 but doesn't work with --reservation nor --account

module add languages/anaconda3/3.5-4.2.0-tflow-1.7
module load CUDA
#module add apps/torch/28.01.2019

echo "install pytorch"

pip install torch torchvision --user --upgrade

echo "train gppvae"

python train_gppvae.py \
  --outdir ./out/gppvae
  --data ../../data/data_faces.h5 \
  --vae_cfg ./out/vae/vae.cfg.p \
  --vae_weights ../faceplace/out/vae/weights/weights.00900.pt \
  --epoch_cb 2 \
  --epochs 10 \
  --enable-cuda
