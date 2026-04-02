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
    --output_dir  train_text/checkpoints/router \
    --loss_type   supcon \
    --temperature 0.07 \
    --batch_size  96 \
    --epochs      20 \
    --lambda_reg  0.1 \
    --seed        42 \
    --use_wandb \
    --wandb_project z-image-router
