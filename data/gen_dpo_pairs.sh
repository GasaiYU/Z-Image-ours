#!/usr/bin/env bash
# gen_dpo_pairs.sh
# 多卡并行运行 generate_dpo_pairs.py，按名词列表切片，每进程独占一张 GPU。
#
# 用法：
#   bash data/gen_dpo_pairs.sh                         # 默认 8 卡
#   N_GPUS=4 bash data/gen_dpo_pairs.sh                # 只用前 4 张
#   N_GPUS=1 GPU_START=2 bash data/gen_dpo_pairs.sh    # 只用第 2 张卡
#   SKIP_VLM=1 bash data/gen_dpo_pairs.sh              # 跳过 VLM 验证
#
# 环境变量：
#   N_GPUS        要启动的进程数（默认 8）
#   GPU_START     起始 GPU 编号（默认 0）
#   SKIP_VLM      设为 1 则跳过 VLM 验证阶段（默认 0）
#   N_EDITS       每个 (名词,数量) 组合生成几张编辑图（默认 3）
#   MIN_COUNT     编辑数量范围下界（默认 1）
#   MAX_COUNT     编辑数量范围上界（默认 10）
#   MIN_SEED_SCORE 种子图像最低可接受分数（默认 0.8）

export HF_HOME=/mmu-vcg/gaomingju/data/T2I/hf_cache/
set -euo pipefail

# ── 可调参数 ──────────────────────────────────────────────────────────────────
N_GPUS=${N_GPUS:-8}
GPU_START=${GPU_START:-0}
SKIP_VLM=${SKIP_VLM:-0}
N_EDITS=${N_EDITS:-3}
MIN_COUNT=${MIN_COUNT:-1}
MAX_COUNT=${MAX_COUNT:-10}
MIN_SEED_SCORE=${MIN_SEED_SCORE:-0.8}

JSONL=${JSONL:-"data/train_triplets/counting_triplets_minimal_origin.jsonl"}
NOUNS_FILE=${NOUNS_FILE:-"data/train_triplets/counting_nouns.txt"}
IMAGE_DIR=${IMAGE_DIR:-"data/generated_images"}
OUTDIR=${OUTDIR:-"data/dpo_edit_images"}

GEN_MODEL=${GEN_MODEL:-"Qwen/Qwen-Image-2512"}
EDIT_MODEL=${EDIT_MODEL:-"Qwen/Qwen-Image-Edit"}
VLM_MODEL=${VLM_MODEL:-"Qwen/Qwen2-VL-7B-Instruct"}

# ── 打印配置 ──────────────────────────────────────────────────────────────────
echo "========================================================"
echo "  generate_dpo_pairs.py  –  multi-GPU launcher"
echo "  N_GPUS     : ${N_GPUS}  (GPU ${GPU_START} ~ $((GPU_START + N_GPUS - 1)))"
echo "  JSONL      : ${JSONL}"
echo "  NOUNS_FILE : ${NOUNS_FILE}"
echo "  IMAGE_DIR  : ${IMAGE_DIR}"
echo "  OUTDIR     : ${OUTDIR}"
echo "  N_EDITS    : ${N_EDITS}"
echo "  COUNT range: ${MIN_COUNT} ~ ${MAX_COUNT}"
echo "  SEED SCORE : >= ${MIN_SEED_SCORE}"
echo "  SKIP_VLM   : ${SKIP_VLM}"
echo "========================================================"

SKIP_VLM_FLAG=""
if [[ "${SKIP_VLM}" == "1" ]]; then
    SKIP_VLM_FLAG="--skip_vlm"
fi

# ── 启动进程 ──────────────────────────────────────────────────────────────────
PIDS=()
for (( rank=0; rank<N_GPUS; rank++ )); do
    gpu_id=$(( GPU_START + rank ))

    CUDA_VISIBLE_DEVICES=${gpu_id} \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    python data/generate_dpo_pairs.py \
        --jsonl           "${JSONL}" \
        --nouns_file      "${NOUNS_FILE}" \
        --image_dir       "${IMAGE_DIR}" \
        --outdir          "${OUTDIR}" \
        --gen_model       "${GEN_MODEL}" \
        --edit_model      "${EDIT_MODEL}" \
        --vlm_model       "${VLM_MODEL}" \
        --min_seed_score  "${MIN_SEED_SCORE}" \
        --min_count       "${MIN_COUNT}" \
        --max_count       "${MAX_COUNT}" \
        --n_edits         "${N_EDITS}" \
        --rank            "${rank}" \
        --world_size      "${N_GPUS}" \
        ${SKIP_VLM_FLAG} \
        2>&1 | tee "logs/gen_dpo_pairs_rank${rank}.log" &

    PIDS+=($!)
    echo "  [rank=${rank}] pid=${PIDS[-1]}  GPU=${gpu_id}"
done

echo ""
echo "All ${N_GPUS} workers started. Waiting ..."
echo ""

FAILED=0
for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    if wait "$pid"; then
        echo "  [rank=${i}] pid=${pid}  DONE"
    else
        echo "  [rank=${i}] pid=${pid}  FAILED (exit $?)" >&2
        FAILED=$(( FAILED + 1 ))
    fi
done

echo ""
if [[ $FAILED -eq 0 ]]; then
    echo "All workers finished successfully. Output: ${OUTDIR}"
else
    echo "${FAILED} worker(s) failed." >&2
    exit 1
fi
