#!/usr/bin/env bash
set -euo pipefail

# ── GPU selection ─────────────────────────────────────────────────────────────
NUM_GPUS=${NUM_GPUS:-8}

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR=${MODEL_DIR:-ckpts/Z-Image-Turbo}
TRIPLETS_JSONL=${TRIPLETS_JSONL:-data/train_triplets/counting_triplets_filtered.jsonl}
GENERATED_ROOT=${GENERATED_ROOT:-data/generated_images}
OUTPUT_DIR=${OUTPUT_DIR:-train_text/checkpoints/counting_text_refiner_linear_encoder}

# ── Data / model ──────────────────────────────────────────────────────────────
VERDICT_THRESHOLD=${VERDICT_THRESHOLD:-0.8}
RESOLUTION=${RESOLUTION:-1024}
MAX_LENGTH=${MAX_LENGTH:-128}

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS=${EPOCHS:-10}                  # short: prevent catastrophic forgetting on narrow counting data
BATCH_SIZE=${BATCH_SIZE:-1}
CONTRASTIVE_BATCH_SIZE=${CONTRASTIVE_BATCH_SIZE:-32}
TEXT_CHUNK_SIZE=${TEXT_CHUNK_SIZE:-16}
NUM_WORKERS=${NUM_WORKERS:-2}
LR=${LR:-1e-3}
REFINER_LR=${REFINER_LR:-1e-4}       # 5× lower than before; narrow data + many steps → drift
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
PROJ_HIDDEN_DIM=${PROJ_HIDDEN_DIM:-512}   # unused (single-layer proj, no hidden dim)
PROJ_OUT_DIM=${PROJ_OUT_DIM:-256}         # linear proj output dim: refiner_dim → proj_out_dim
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
SEED=${SEED:-42}

# ── Loss ──────────────────────────────────────────────────────────────────────
NUM_NEGATIVES=${NUM_NEGATIVES:-12}
TEMPERATURE=${TEMPERATURE:-0.07}
CONTRASTIVE_WEIGHT=${CONTRASTIVE_WEIGHT:-0.2}
DIFFUSION_WEIGHT=${DIFFUSION_WEIGHT:-1.0}   # diffusion on narrow counting data causes rapid collapse of frozen joint-attn

# ── Logging / checkpoints ─────────────────────────────────────────────────────
SAVE_EVERY=${SAVE_EVERY:-200}          # frequent checkpoints to detect collapse early
VIS_EVERY=${VIS_EVERY:-50}           # check generation quality every 50 steps
WANDB_PROJECT=${WANDB_PROJECT:-z-image-text-refiner-training}
WANDB_RUN=${WANDB_RUN:-counting_text_refiner_diffusion_baseline}

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
    --contrastive_batch_size "$CONTRASTIVE_BATCH_SIZE" \
    --text_chunk_size "$TEXT_CHUNK_SIZE" \
    --num_workers "$NUM_WORKERS" \
    --lr "$LR" \
    --refiner_lr "$REFINER_LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --proj_hidden_dim "$PROJ_HIDDEN_DIM" \
    --proj_out_dim "$PROJ_OUT_DIM" \
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
