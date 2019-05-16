train vae

```bash
nohup python train_vae.py --outdir ./out/vae --data ../../../data/data_faces.h5 --epochs 5000 --lr 0.001 &> train_vae.out &

python train_vae.py --outdir ./out/vae --data ../../../data/data_faces.h5 --epochs 10 --epoch_cb 1 --lr 0.001
```

train cvae

```bash
nohup python train_cvae.py --outdir ./out/cvae --data ../../../data/data_faces.h5 --epochs 5000 --lr 0.001 &> train_cvae.out &
```

training in unison

```bash
nohup python train_gppvae.py --epochs 5500 --train_unison &> train_gppvae_unison.out &

python train_gppvae.py --epochs 10 --epoch_cb 1 --train_unison
```

train gppvae

```bash
nohup python train_gppvae.py --epochs 1000 --vae_cfg ./out/vae/vae.cfg.p --vae_weights ./out/vae/weights/weights.04900.pt --outdir ./out/gppvae &> train_gppvae.out &
```

