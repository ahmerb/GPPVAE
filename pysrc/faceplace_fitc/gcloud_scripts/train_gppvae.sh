nohup \
  "python train_gppvae.py \
    --outdir ./out/gppvae \
    --data ../../data/data_faces.h5 \
    --vae_cfg ../faceplace/out/vae/vae.cfg.p \
    --vae_weights ../faceplace/out/vae/weights/weights.00900.pt \
    --epoch_cb 2 \
    --epochs 10 \
    --enable-cuda" \
  &> train_gppvae.out \
  &
