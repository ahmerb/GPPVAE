run

```bash
nohup python train_vae.py --outdir ./out/vae --epochs 3000 --epoch_cb 50 --lr 0.0003 &> train_vae.out &
```

or for testing stuff

```bash
python train_vae.py --outdir ./out/vae2 --epochs 100 --epoch_cb 10 --lr 0.0003
```

---

gppvae_fitc unison

```bash
nohup python train_gppvae_fitc.py  --outdir ./out/fitc_gppvae_unison --epochs 3000 --epoch_cb 150 --vae_lr 0.0003 --gp_lr 0.003 --train_unison &> train_gppvae_unison.out &
```

or for testing stuff

```bash
python train_gppvae_fitc.py --outdir ./out/fitc_gppvae_unison --epochs 10 --epoch_cb 1 --vae_lr 0.0003 --gp_lr 0.003 --train_unison
```

---

cvae

```bash
nohup python train_cvae.py --outdir ./out/cvae --epochs 3000 --lr 0.001 &> train_cvae.out &
```

```bash
python train_cvae.py --outdir ./out/cvae --epochs 10 --epoch_cb 1 --lr 0.001
````
