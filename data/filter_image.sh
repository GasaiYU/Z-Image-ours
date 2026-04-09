#!/usr/bin/env bash
# filter_image.sh
# 依次在后台启动 N_GPUS 个独立 Python 进程，对已生成的图像做 VLM 筛选。
# 与 gen_image.sh 完全同构：每进程 CUDA_VISIBLE_DEVICES=X 独占一张卡。
#
# 用法：
#   bash data/filter_image.sh                                      # 默认 8 卡，Qwen3-VL-8B
#   VLM_MODEL="Qwen/Qwen3-VL-32B-Instruct" bash data/filter_image.sh  # 用更大的 32B 模型
#   N_GPUS=4 bash data/filter_image.sh                             # 只用前 4 张卡
#   TASKS="counting" bash data/filter_image.sh                     # 只筛 counting
#   THRESHOLD=0.8 bash data/filter_image.sh                        # 更严格的通过阈值
export HF_HOME=/mmu-vcg/gaomingju/data/T2I/hf_cache/
set -euo pipefail

N_GPUS=${N_GPUS:-8}
GPU_START=${GPU_START:-0}
IMAGE_DIR=${IMAGE_DIR:-"data/generated_images"}
TRIPLET_DIR=${TRIPLET_DIR:-"data/train_triplets"}
VLM_MODEL=${VLM_MODEL:-"Qwen/Qwen3-VL-8B-Instruct"}
THRESHOLD=${THRESHOLD:-0.5}
TASKS=${TASKS:-""}
OVERWRITE=${OVERWRITE:-""}   # 非空则传 --overwrite

echo "========================================"
echo "  Filtering with ${N_GPUS} workers"
echo "  GPUs     : ${GPU_START} ~ $((GPU_START + N_GPUS - 1))"
echo "  VLM      : ${VLM_MODEL}"
echo "  tasks    : ${TASKS:-all}"
echo "  threshold: ${THRESHOLD}"
echo "  image_dir: ${IMAGE_DIR}"
echo "========================================"

PIDS=()
for (( rank=0; rank<N_GPUS; rank++ )); do
    gpu_id=$(( GPU_START + rank ))

    TASKS_ARG=""
    [[ -n "${TASKS}" ]] && TASKS_ARG="--tasks ${TASKS}"

    OVERWRITE_ARG=""
    [[ -n "${OVERWRITE}" ]] && OVERWRITE_ARG="--overwrite"

    CUDA_VISIBLE_DEVICES=${gpu_id} \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    python data/filter_triplet_images.py \
        --image_dir      "${IMAGE_DIR}" \
        --triplet_dir    "${TRIPLET_DIR}" \
        --vlm_model      "${VLM_MODEL}" \
        --threshold      "${THRESHOLD}" \
        --gpu_id         0 \
        --rank           "${rank}" \
        --world_size     "${N_GPUS}" \
        ${TASKS_ARG} \
        ${OVERWRITE_ARG} &

    PIDS+=($!)
    echo "  [rank=${rank}] pid=${PIDS[-1]}  GPU=${gpu_id}"
done

echo ""
echo "All workers started. Waiting for completion ..."
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
    # 合并所有 rank 的汇总 JSONL
    MERGED="${IMAGE_DIR}/filtered_all.jsonl"
    cat "${IMAGE_DIR}"/filtered_rank*.jsonl > "${MERGED}" 2>/dev/null || true
    total=$(wc -l < "${MERGED}" 2>/dev/null || echo 0)
    echo "All workers finished. Passed samples: ${total}"
    echo "Merged summary → ${MERGED}"
else
    echo "${FAILED} worker(s) failed." >&2
    exit 1
fi
