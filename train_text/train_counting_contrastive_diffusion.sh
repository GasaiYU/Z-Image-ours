#!/usr/bin/env bash
set -euo pipefail

# ── GPU selection ─────────────────────────────────────────────────────────────
# Set NUM_GPUS to the number of GPUs you want to use.
# e.g.  NUM_GPUS=4 bash train_text/train_counting_contrastive_diffusion.sh
NUM_GPUS=${NUM_GPUS:-8}

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR=${MODEL_DIR:-ckpts/Z-Image-Turbo}
TRIPLETS_JSONL=${TRIPLETS_JSONL:-data/train_triplets/counting_triplets_filtered.jsonl}
GENERATED_ROOT=${GENERATED_ROOT:-data/generated_images}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/counting_text_refiner_infonce}

# ── Data / model ──────────────────────────────────────────────────────────────
VERDICT_THRESHOLD=${VERDICT_THRESHOLD:-0.8}
RESOLUTION=${RESOLUTION:-1024}
MAX_LENGTH=${MAX_LENGTH:-128}

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-4}          # per-GPU batch size
NUM_WORKERS=${NUM_WORKERS:-2}
LR=${LR:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
SEED=${SEED:-42}

# ── Loss ──────────────────────────────────────────────────────────────────────
NUM_NEGATIVES=${NUM_NEGATIVES:-12}
TEMPERATURE=${TEMPERATURE:-0.07}
CONTRASTIVE_WEIGHT=${CONTRASTIVE_WEIGHT:-1.0}
DIFFUSION_WEIGHT=${DIFFUSION_WEIGHT:-5.0}

# ── Logging / checkpoints ─────────────────────────────────────────────────────
SAVE_EVERY=${SAVE_EVERY:-500}
VIS_EVERY=${VIS_EVERY:-500}
WANDB_PROJECT=${WANDB_PROJECT:-z-image-text-refiner-training}
WANDB_RUN=${WANDB_RUN:-text_refiner_counting}

# ── Launch ────────────────────────────────────────────────────────────────────
accelerate launch \
  --num_processes "$NUM_GPUS" \
  --mixed_precision "$MIXED_PRECISION" \
  train_text/train_counting_contrastive_diffusion.py \
    --model_dir "$MODEL_DIR" \
    --triplets_jsonl "$TRIPLETS_JSONL" \
    --generated_root "$GENERATED_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --verdict_threshold "$VERDICT_THRESHOLD" \
    --resolution "$RESOLUTION" \
    --max_length "$MAX_LENGTH" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --lr "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --mixed_precision "$MIXED_PRECISION" \
    --seed "$SEED" \
    --num_negatives "$NUM_NEGATIVES" \
    --temperature "$TEMPERATURE" \
    --contrastive_weight "$CONTRASTIVE_WEIGHT" \
    --diffusion_weight "$DIFFUSION_WEIGHT" \
    --save_every "$SAVE_EVERY" \
    --vis_every "$VIS_EVERY" \
    --use_chat_template \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --wandb_run "$WANDB_RUN"
