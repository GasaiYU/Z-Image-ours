#!/bin/bash

# Define directories
IMAGEDIR="outputs"
OUTFILE="evaluation_results.jsonl"
MODEL_PATH="/mmu-vcg/gaomingju/workspace/T2I/Z-Image-ours/benchmarks/geneval/pretrained"

# Run evaluation
python benchmarks/geneval/evaluation/evaluate_images.py \
    $IMAGEDIR \
    --outfile $OUTFILE \
    --model-path $MODEL_PATH

# # Run summary scores (optional, if you want to see the final score)
# python benchmarks/geneval/evaluation/summary_scores.py $OUTFILE
