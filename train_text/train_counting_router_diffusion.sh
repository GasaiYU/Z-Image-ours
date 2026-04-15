#!/usr/bin/env bash
set -euo pipefail

# Router-based counting contrastive (+ optional diffusion) training
# Single GPU:  bash train_text/train_counting_router.sh
# Multi GPU:   NUM_GPUS=4 bash train_text/train_counting_router.sh

NUM_GPUS=${NUM_GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29500}

MODEL_DIR=${MODEL_DIR:-ckpts/Z-Image-Turbo}
TRIPLETS_JSONL=${TRIPLETS_JSONL:-data/train_triplets/counting_triplets_filtered.jsonl}
GENERATED_ROOT=${GENERATED_ROOT:-data/generated_images}
OUTPUT_DIR=${OUTPUT_DIR:-train_text/checkpoints/counting_router_dcl}

# Training
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-1}             # image batch for diffusion loss (only used if DIFFUSION_WEIGHT > 0)
CTR_BATCH=${CTR_BATCH:-32}             # text-only batch for contrastive loss
LR=${LR:-3e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
NUM_WORKERS=${NUM_WORKERS:-4}
SEED=${SEED:-42}
MAX_LENGTH=${MAX_LENGTH:-128}

# Router architecture
MID_DIM=${MID_DIM:-1024}
ROUTE_START=${ROUTE_START:-10}
ROUTE_END=${ROUTE_END:-21}

# Loss
LOSS_TYPE=${LOSS_TYPE:-dcl}            # dcl | infonce | triplet
TEMPERATURE=${TEMPERATURE:-0.07}
TRIPLET_MARGIN=${TRIPLET_MARGIN:-0.2}
NUM_NEGATIVES=${NUM_NEGATIVES:-12}
CONTRASTIVE_WEIGHT=${CONTRASTIVE_WEIGHT:-1.0}
DIFFUSION_WEIGHT=${DIFFUSION_WEIGHT:-5.0}  # set >0 to enable diffusion loss

# Logging
SAVE_EVERY=${SAVE_EVERY:-200}
VIS_EVERY=${VIS_EVERY:-100}
WANDB_PROJECT=${WANDB_PROJECT:-z-image-router-counting}
WANDB_RUN=${WANDB_RUN:-router_counting_dcl}

accelerate launch \
  --num_processes "$NUM_GPUS" \
  --main_process_port "$MASTER_PORT" \
  --mixed_precision bf16 \
  train_text/train_counting_router_diffusion.py \
    --model_dir              "$MODEL_DIR" \
    --triplets_jsonl         "$TRIPLETS_JSONL" \
    --generated_root         "$GENERATED_ROOT" \
    --output_dir             "$OUTPUT_DIR" \
    --epochs                 "$EPOCHS" \
    --batch_size             "$BATCH_SIZE" \
    --contrastive_batch_size "$CTR_BATCH" \
    --lr                     "$LR" \
    --weight_decay           "$WEIGHT_DECAY" \
    --num_workers            "$NUM_WORKERS" \
    --seed                   "$SEED" \
    --max_length             "$MAX_LENGTH" \
    --mid_dim                "$MID_DIM" \
    --route_start            "$ROUTE_START" \
    --route_end              "$ROUTE_END" \
    --loss_type              "$LOSS_TYPE" \
    --temperature            "$TEMPERATURE" \
    --triplet_margin         "$TRIPLET_MARGIN" \
    --num_negatives          "$NUM_NEGATIVES" \
    --contrastive_weight     "$CONTRASTIVE_WEIGHT" \
    --diffusion_weight       "$DIFFUSION_WEIGHT" \
    --save_every             "$SAVE_EVERY" \
    --vis_every              "$VIS_EVERY" \
    --mixed_precision        bf16 \
    --use_wandb \
    --wandb_project          "$WANDB_PROJECT" \
    --wandb_run              "$WANDB_RUN"
