#!/bin/bash
# Multi-seed GenEval generation + evaluation script
# Usage:
#   bash scripts/run_multi_seed_geneval.sh \
#     [--seeds "42 43 44 45 46 47 48 49"] \
#     [--gpus "0 1 2 3 4 5 6 7"] \
#     [--tags "counting"] [--eval]

# ---- Default config ----
SEEDS="42 43 44 45 46 47 48 49"
GPUS="0 1 2 3 4 5 6 7"
TAGS="counting"
RUN_EVAL=false
BASE_OUTDIR="outputs/multi_seed_geneval_naive"
MODEL_PATH="/mmu-vcg/gaomingju/workspace/T2I/Z-Image-ours/benchmarks/geneval/pretrained"
DETECTOR_MODEL="mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco"
PROMPT_FILE="benchmarks/geneval/prompts/evaluation_metadata.jsonl"

# ---- Parse arguments ----
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds)   SEEDS="$2";   shift 2 ;;
        --gpus)    GPUS="$2";    shift 2 ;;
        --tags)    TAGS="$2";    shift 2 ;;
        --outdir)  BASE_OUTDIR="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --eval)    RUN_EVAL=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=========================================="
echo "  Multi-Seed GenEval Runner"
echo "  Seeds: $SEEDS"
echo "  GPUs:  $GPUS"
echo "  Tags:  $TAGS"
echo "  Eval:  $RUN_EVAL"
echo "  OutDir: $BASE_OUTDIR"
echo "=========================================="

mkdir -p "$BASE_OUTDIR"

SUMMARY_FILE="$BASE_OUTDIR/all_seeds_summary.txt"
echo "Seed | Overall | Counting | Single Object | Two Object | Counting | Colors | Color Attrib | Position" > "$SUMMARY_FILE"
echo "------|---------|----------|---------------|------------|----------|--------|--------------|----------" >> "$SUMMARY_FILE"

read -r -a SEED_ARRAY <<< "$SEEDS"
read -r -a GPU_ARRAY <<< "$GPUS"

if [ "${#SEED_ARRAY[@]}" -ne "${#GPU_ARRAY[@]}" ]; then
    echo "ERROR: Number of seeds (${#SEED_ARRAY[@]}) must equal number of GPUs (${#GPU_ARRAY[@]})."
    echo "Example: --seeds \"42 43 44 45 46 47 48 49\" --gpus \"0 1 2 3 4 5 6 7\""
    exit 1
fi

run_one_seed() {
    local SEED="$1"
    local GPU_ID="$2"
    local SEED_DIR="$BASE_OUTDIR/seed_${SEED}"
    local EVAL_OUTFILE="$SEED_DIR/evaluation_results.jsonl"
    local LOG_FILE="$SEED_DIR/run.log"

    mkdir -p "$SEED_DIR"
    {
        echo "========== SEED $SEED on GPU $GPU_ID =========="
        echo "Output dir: $SEED_DIR"
        echo "[1/2] Generating images..."

        CUDA_VISIBLE_DEVICES="$GPU_ID" python benchmarks/geneval/generation/zimage_generate.py \
            "$PROMPT_FILE" \
            --outdir "$SEED_DIR" \
            --tags $TAGS \
            --seed "$SEED"

        if [ $? -ne 0 ]; then
            echo "ERROR: Generation failed for seed $SEED on GPU $GPU_ID."
            return 1
        fi
        echo "Generation done."

        if [ "$RUN_EVAL" = true ]; then
            echo "[2/2] Evaluating..."
            CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES="$GPU_ID" \
            python benchmarks/geneval/evaluation/evaluate_images.py \
                "$SEED_DIR" \
                --outfile "$EVAL_OUTFILE" \
                --model-path "$MODEL_PATH" \
                --options model="$DETECTOR_MODEL" detector_device=cuda clip_device=cuda

            if [ $? -ne 0 ]; then
                echo "ERROR: Evaluation failed for seed $SEED on GPU $GPU_ID."
                return 1
            fi

            echo "Scores for seed $SEED:"
            python benchmarks/geneval/evaluation/summary_scores.py "$EVAL_OUTFILE"
        else
            echo "[2/2] Skipping evaluation (pass --eval to enable)."
        fi
    } 2>&1 | tee "$LOG_FILE"
}

echo ""
echo "Launching ${#SEED_ARRAY[@]} parallel jobs..."
PIDS=()
for i in "${!SEED_ARRAY[@]}"; do
    SEED="${SEED_ARRAY[$i]}"
    GPU_ID="${GPU_ARRAY[$i]}"
    run_one_seed "$SEED" "$GPU_ID" &
    PIDS+=($!)
done

FAILED=0
for pid in "${PIDS[@]}"; do
    wait "$pid" || FAILED=1
done

if [ "$RUN_EVAL" = true ]; then
    for SEED in "${SEED_ARRAY[@]}"; do
        EVAL_OUTFILE="$BASE_OUTDIR/seed_${SEED}/evaluation_results.jsonl"
        if [ -f "$EVAL_OUTFILE" ]; then
            echo "Seed $SEED" >> "$SUMMARY_FILE"
            python benchmarks/geneval/evaluation/summary_scores.py "$EVAL_OUTFILE" 2>&1 >> "$SUMMARY_FILE"
            echo "---" >> "$SUMMARY_FILE"
        fi
    done
fi

echo ""
echo "=========================================="
echo "  All seeds done."
if [ "$RUN_EVAL" = true ]; then
    echo "  Summary saved to: $SUMMARY_FILE"
    echo ""
    echo "=== FINAL SUMMARY ==="
    cat "$SUMMARY_FILE"
fi
echo "=========================================="

if [ "$FAILED" -ne 0 ]; then
    exit 1
fi
