#!/usr/bin/env bash
# gen_image.sh
# 依次在后台启动 N_GPUS 个完全独立的 Python 进程，每个进程绑定一张 GPU。
# 每个进程通过 CUDA_VISIBLE_DEVICES=X 独占一张卡，用 --rank / --world_size
# 对 anchor 列表做静态切片，彼此不通信，完全绕开 Python GIL。
#
# 用法：
#   bash data/gen_image.sh                              # 默认 8 卡，生成全部 task
#   N_GPUS=4 bash data/gen_image.sh                     # 只用前 4 张卡
#   N_GPUS=1 GPU_START=3 bash data/gen_image.sh         # 只用第 3 张卡（单卡调试）
#   TASKS="counting" bash data/gen_image.sh             # 只生成 counting task
#   TASKS="counting color" bash data/gen_image.sh       # 生成 counting + color
export HF_HOME=/mmu-vcg/gaomingju/data/T2I/hf_cache/
set -euo pipefail

N_GPUS=${N_GPUS:-8}          # 要启动的进程数
GPU_START=${GPU_START:-0}    # 起始 GPU 编号（若想跳过前几张卡）
N_SAMPLES=${N_SAMPLES:-4}    # 每个 anchor 生成几张图
BATCH_SIZE=${BATCH_SIZE:-1}  # 单次推理 batch size
OUTDIR=${OUTDIR:-"data/generated_images"}
TRIPLET_DIR=${TRIPLET_DIR:-"data/train_triplets"}
# 要生成的 task，空格分隔多个，留空则生成全部
# 例：TASKS="counting" 或 TASKS="counting color"
TASKS=${TASKS:-""}

echo "========================================"
echo "  Launching ${N_GPUS} workers"
echo "  GPUs  : ${GPU_START} ~ $((GPU_START + N_GPUS - 1))"
echo "  tasks : ${TASKS:-all}"
echo "  out   : ${OUTDIR}"
echo "========================================"

PIDS=()
for (( rank=0; rank<N_GPUS; rank++ )); do
    gpu_id=$(( GPU_START + rank ))

    # 若 TASKS 非空，则拼接 --tasks 参数（支持多个 task 空格分隔）
    TASKS_ARG=""
    if [[ -n "${TASKS}" ]]; then
        TASKS_ARG="--tasks ${TASKS}"
    fi

    CUDA_VISIBLE_DEVICES=${gpu_id} \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    python data/generate_triplet_images.py \
        --triplet_dir "${TRIPLET_DIR}" \
        --outdir      "${OUTDIR}" \
        --n_samples   "${N_SAMPLES}" \
        --batch_size  "${BATCH_SIZE}" \
        --rank        "${rank}" \
        --world_size  "${N_GPUS}" \
        ${TASKS_ARG} &

    PIDS+=($!)
    echo "  [rank=${rank}] pid=${PIDS[-1]}  GPU=${gpu_id}"
done

echo ""
echo "All workers started. Waiting for completion ..."
echo ""

# 等待所有子进程结束，任何一个以非 0 退出则打印警告
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
    echo "All workers finished successfully."
else
    echo "${FAILED} worker(s) failed." >&2
    exit 1
fi
