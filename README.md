#  Knowledge Distillation with Refined Logits

## Environment

Python 3.8, torch 1.7.0

A selection of packages that may require additional installation: torchvision, tensorboardX, yacs, wandb, tqdm, scipy

## Pre-trained Teachers

Pre-trained teachers can be downloaded from [Decoupled Knowledge Distillation (CVPR 2022)](https://github.com/megvii-research/mdistiller/releases/tag/checkpoints). Download the `cifar_teachers.tar` and untar it to `./download_ckpts` via `tar xvf cifar_teachers.tar`.

## Training on CIFAR-100

```sh
# Train method X with the following code
CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/X/vgg13_vgg8.yaml
# You can refer to the following code to additionally specify hyper-parameters
# Train DKD
CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/dkd/vgg13_vgg8.yaml DKD.ALPHA 1. DKD.BETA 8. DKD.T 4.
# Train RLD
CUDA_VISIBLE_DEVICES=0 python tools/train.py --cfg configs/cifar100/rld/vgg13_vgg8.yaml --same-t RLD.ALPHA 1. RLD.BETA 8. RLD.T 4.
```

## Acknowledgment

This codebase is heavily borrowed from [Logit Standardization in Knowledge Distillation (CVPR 2024)](https://github.com/sunshangquan/logit-standardization-KD). Sincere gratitude to the authors for their distinguished efforts. 