run

```bash
nohup python train_vae.py --outdir ./out/vae --epochs 1000 --epoch_cb 50 --lr 0.0003 &> train_vae.out &
```

or for testing stuff

```bash
python train_vae.py --outdir ./out/vae2 --epochs 100 --epoch_cb 10 --lr 0.0003
```

---

```bash
nohup python train_gppvae.py &> train_vae.out &
```

or for testing stuff

```bash
python train_gppvae.py --epochs 10 --epoch_cb 1
```
