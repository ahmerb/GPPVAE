run

```bash
nohup python train_vae.py --outdir ./out/vae --data ../../data/data_faces.h5 --epochs 5000 --lr 0.001 &> train_vae.out &
```

or for testing stuff

```bash
python train_vae.py --outdir ./out/vae --data ../../data/data_faces.h5 --epochs 10 --epoch_cb 1 --lr 0.001
```
