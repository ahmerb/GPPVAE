#!/bin/bash
module add languages/anaconda3/3.5-4.2.0-tflow-1.7
#module add apps/torch/28.01.2019
srun -p gpu --gres=gpu:1 -A comsm0018 -t 0-02:00 --mem=4G --pty bash

