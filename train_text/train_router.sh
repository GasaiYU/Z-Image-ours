#!/usr/bin/env bash
# Single-GPU:  bash train_text/train_router.sh
# Multi-GPU:   NPROC=8 bash train_text/train_router.sh

NPROC=${NPROC:-1}

torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_text/train_router.py \
    --model_dir  ckpts/Z-Image-Turbo \
    --triplet_dir data/train_triplets \
    --output_dir  train_text/checkpoints/router_v5_entropy \
    --loss_type   supcon \
    --temperature 0.07 \
    --temperature_init 0.2 \
    --temperature_warmup_steps 200 \
    --mid_dim     1024 \
    --batch_size  128 \
    --epochs      20 \
    --lambda_disc    0.0 \
    --disc_temperature 1.0 \
    --lambda_entropy 0.02 \
    --seed        42 \
    --use_wandb \
    --wandb_project z-image-router \
    --wandb_run     router_v5_entropy
