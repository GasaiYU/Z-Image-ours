#!/bin/bash

# Define directories
IMAGEDIR="outputs/outputs_geneval_sub"
OUTFILE="evaluation_results.jsonl"
MODEL_PATH="/mmu-vcg/gaomingju/workspace/T2I/Z-Image-ours/benchmarks/geneval/pretrained"

# Run evaluation
python benchmarks/geneval/evaluation/evaluate_images.py \
    $IMAGEDIR \
    --outfile $OUTFILE \
    --model-path $MODEL_PATH \
    --options detector_device=cpu clip_device=cuda

# # Run summary scores (optional, if you want to see the final score)
# python benchmarks/geneval/evaluation/summary_scores.py $OUTFILE
