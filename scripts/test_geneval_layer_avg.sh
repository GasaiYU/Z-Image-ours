#!/usr/bin/env bash
# Test GenEval Counting with uniform average of all LLM layers (training-free baseline).
#
# Usage:
#   bash scripts/test_geneval_layer_avg.sh [seed]
#
# To test only a partial range of layers, edit --layer_start / --layer_end below.

SEED=${1:-42}

python benchmarks/geneval/generation/zimage_generate_layer_avg.py \
  benchmarks/geneval/prompts/evaluation_metadata.jsonl \
  --outdir  outputs/geneval_layer_avg_all_seed${SEED} \
  --tags    counting \
  --seed    ${SEED} \
  --layer_start 0 \
  --layer_end   -1
