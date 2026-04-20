#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# train_counting_grpo.sh
# Flow-GRPO-Fast for Z-Image counting, Qwen-VL reward
#
# Usage:
#   bash train_text/train_counting_grpo.sh
#
# Key env overrides:
#   NUM_GPUS=4 REWARD_GPU=7 LR=1e-4 bash train_text/train_counting_grpo.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Hardware ──────────────────────────────────────────────────────────────────
NUM_GPUS=${NUM_GPUS:-1}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR=${MODEL_DIR:-ckpts/Z-Image-Turbo}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}
# Reward model runs on a separate GPU to avoid OOM with the training model.
# Set to -1 to co-locate with training GPU (only safe on high-VRAM cards).
REWARD_GPU=${REWARD_GPU:-7}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/counting_grpo}
# Noun list aligned with data/train_triplets/counting_nouns.txt (325 nouns)
NOUNS_FILE=${NOUNS_FILE:-data/train_triplets/counting_nouns.txt}

# ── Model / generation ────────────────────────────────────────────────────────
RESOLUTION=${RESOLUTION:-512}
MAX_LENGTH=${MAX_LENGTH:-128}
# Aligned with flow_grpo fast configs (num_steps=10)
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-10}

# ── GRPO-Fast SDE window ──────────────────────────────────────────────────────
# Aligned with flow_grpo geneval_sd3_fast_nocfg / geneval_flux_fast:
#   sde_type        = cps  (Coefficients-Preserving Sampling)
#   sde_window_size = 3
#   sde_window_range = (0, num_steps//2) = (0, 5) for num_steps=10
SDE_TYPE=${SDE_TYPE:-cps}
SDE_WINDOW_SIZE=${SDE_WINDOW_SIZE:-3}
SDE_WINDOW_RANGE_START=${SDE_WINDOW_RANGE_START:-0}
SDE_WINDOW_RANGE_END=${SDE_WINDOW_RANGE_END:-5}
NOISE_LEVEL=${NOISE_LEVEL:-0.8}

# ── Training ──────────────────────────────────────────────────────────────────
NUM_EPOCHS=${NUM_EPOCHS:-200}
NUM_BATCHES_PER_EPOCH=${NUM_BATCHES_PER_EPOCH:-4}
# unique prompts per batch (per GPU)
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
# images generated per prompt (larger = more stable advantages, more GPU memory)
GROUP_SIZE=${GROUP_SIZE:-4}
MAX_COUNT=${MAX_COUNT:-5}

LR=${LR:-2e-4}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-true}

# ── Prompt contrastive loss ───────────────────────────────────────────────────
CONTRASTIVE_WEIGHT=${CONTRASTIVE_WEIGHT:-1.0}   # 0 to disable
TEMPERATURE=${TEMPERATURE:-0.07}
NUM_CTR_NEGATIVES=${NUM_CTR_NEGATIVES:-4}
CONTRASTIVE_BATCH_SIZE=${CONTRASTIVE_BATCH_SIZE:-16}

# ── GRPO loss ─────────────────────────────────────────────────────────────────
# flow_grpo fast configs use clip_range=1e-5
CLIP_RANGE=${CLIP_RANGE:-1e-5}
ADV_CLIP_MAX=${ADV_CLIP_MAX:-5.0}

# ── Logging ───────────────────────────────────────────────────────────────────
SAVE_EVERY=${SAVE_EVERY:-100}
VIS_EVERY=${VIS_EVERY:-50}
SEED=${SEED:-42}
USE_WANDB=${USE_WANDB:-true}
WANDB_PROJECT=${WANDB_PROJECT:-z-image-counting-grpo}
WANDB_RUN=${WANDB_RUN:-grpo_fast_qwenvl}

# ─────────────────────────────────────────────────────────────────────────────
if [ "${USE_WANDB}" = "true" ]; then
    WANDB_FLAG="--use_wandb"
else
    WANDB_FLAG=""
fi

if [ "${GRADIENT_CHECKPOINTING}" = "true" ]; then
    GRAD_CKPT_FLAG="--gradient_checkpointing"
else
    GRAD_CKPT_FLAG=""
fi

# ─────────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  Flow-GRPO-Fast  |  Z-Image counting  |  Qwen-VL reward"
echo "============================================================"
echo "  GPUs             : ${NUM_GPUS}"
echo "  Model dir        : ${MODEL_DIR}"
echo "  Nouns file       : ${NOUNS_FILE}"
echo "  Reward model     : ${REWARD_MODEL_PATH} (GPU ${REWARD_GPU})"
echo "  Output dir       : ${OUTPUT_DIR}"
echo "  Resolution       : ${RESOLUTION}"
echo "  Inference steps  : ${NUM_INFERENCE_STEPS}"
echo "  SDE type         : ${SDE_TYPE}"
echo "  SDE window size  : ${SDE_WINDOW_SIZE}"
echo "  SDE window range : [${SDE_WINDOW_RANGE_START}, ${SDE_WINDOW_RANGE_END})"
echo "  Noise level      : ${NOISE_LEVEL}"
echo "  Batch size       : ${TRAIN_BATCH_SIZE} prompts × ${GROUP_SIZE} images"
echo "  LR               : ${LR}"
echo "  Clip range       : ${CLIP_RANGE}"
echo "============================================================"

accelerate launch \
    --num_processes "${NUM_GPUS}" \
    --mixed_precision "${MIXED_PRECISION}" \
    train_text/train_counting_grpo.py \
        --model_dir              "${MODEL_DIR}" \
        --reward_model_path      "${REWARD_MODEL_PATH}" \
        --reward_gpu             "${REWARD_GPU}" \
        --output_dir             "${OUTPUT_DIR}" \
        --nouns_file             "${NOUNS_FILE}" \
        --resolution             "${RESOLUTION}" \
        --max_length             "${MAX_LENGTH}" \
        --num_inference_steps    "${NUM_INFERENCE_STEPS}" \
        --sde_type               "${SDE_TYPE}" \
        --sde_window_size        "${SDE_WINDOW_SIZE}" \
        --sde_window_range       "${SDE_WINDOW_RANGE_START}" "${SDE_WINDOW_RANGE_END}" \
        --noise_level            "${NOISE_LEVEL}" \
        --num_epochs             "${NUM_EPOCHS}" \
        --num_batches_per_epoch  "${NUM_BATCHES_PER_EPOCH}" \
        --train_batch_size       "${TRAIN_BATCH_SIZE}" \
        --group_size             "${GROUP_SIZE}" \
        --max_count              "${MAX_COUNT}" \
        --lr                     "${LR}" \
        --weight_decay           "${WEIGHT_DECAY}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        --max_grad_norm          "${MAX_GRAD_NORM}" \
        --mixed_precision        "${MIXED_PRECISION}" \
        --contrastive_weight     "${CONTRASTIVE_WEIGHT}" \
        --temperature            "${TEMPERATURE}" \
        --num_ctr_negatives      "${NUM_CTR_NEGATIVES}" \
        --contrastive_batch_size "${CONTRASTIVE_BATCH_SIZE}" \
        --clip_range             "${CLIP_RANGE}" \
        --adv_clip_max           "${ADV_CLIP_MAX}" \
        --global_std \
        --save_every             "${SAVE_EVERY}" \
        --vis_every              "${VIS_EVERY}" \
        --seed                   "${SEED}" \
        --wandb_project          "${WANDB_PROJECT}" \
        --wandb_run              "${WANDB_RUN}" \
        ${WANDB_FLAG} \
        ${GRAD_CKPT_FLAG}
