#!/usr/bin/env bash
# Single-GPU:
#   bash train_text/train_router.sh
# Multi-GPU (e.g. 4 GPUs):
#   NPROC=4 bash train_text/train_router.sh

NPROC=${NPROC:-1}

torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_text/train_router.py \
    --model_dir  ckpts/Z-Image-Turbo \
    --triplet_dir data/train_triplets \
    --output_dir  train_text/checkpoints/router_version3 \
    --loss_type   supcon \
    --temperature 0.07 \
    --batch_size  128 \
    --epochs      20 \
    --lambda_reg  0.1 \
    --seed        42 \
    --lambda_entropy 0.005 \
    --use_wandb \
    --wandb_project z-image-router \
    --wandb_run     router_v3_multiscale_entropy
