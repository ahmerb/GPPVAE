To connect to the instance, use:

```sh
gcloud compute --project "deep-learning-237010" ssh --zone "us-west1-b" "pytorch-1-vm"
```

One liner to run in interactive mode:

```sh
python train_svgppvae.py --outdir ./out/gppvae --data ../../data/data_faces.h5 --vae_cfg ../faceplace/out/vae/vae.cfg.p --vae_weights ../faceplace/out/vae/weights/weights.00900.pt --epoch_cb 1 --epochs 10 --enable-cuda
```
