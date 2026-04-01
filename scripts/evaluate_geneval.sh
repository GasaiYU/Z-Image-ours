#!/bin/bash
export CUDA_LAUNCH_BLOCKING=1
# Define directories
IMAGEDIR="outputs/outputs_geneval_counting_decay"
OUTFILE="outputs/outputs_geneval_counting_decay/evaluation_results.jsonl"
MODEL_PATH="/mmu-vcg/gaomingju/workspace/T2I/Z-Image-ours/benchmarks/geneval/pretrained"
# Use Mask2Former on GPU
DETECTOR_MODEL="mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco"

# Run evaluation
python benchmarks/geneval/evaluation/evaluate_images.py \
    $IMAGEDIR \
    --outfile $OUTFILE \
    --model-path $MODEL_PATH \
    --options model=$DETECTOR_MODEL detector_device=cuda clip_device=cuda

# # Run summary scores (optional, if you want to see the final score)
# python benchmarks/geneval/evaluation/summary_scores.py $OUTFILE
