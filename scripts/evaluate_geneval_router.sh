#!/bin/bash
# =============================================================================
# Generate GenEval images using the trained DynamicTokenRouter
#
# Usage:
#   bash scripts/evaluate_geneval_router.sh [ROUTER_CKPT] [SEED] [TAGS...]
#
# Examples:
#   # Generate counting prompts with seed=42 (default)
#   bash scripts/evaluate_geneval_router.sh checkpoints/router/router_best.pt 42
#
#   # Generate counting + color_attr with a different seed
#   bash scripts/evaluate_geneval_router.sh checkpoints/router/router_best.pt 420 counting color_attr
#
# Output:
#   outputs/geneval_router_<tags>_seed_<SEED>/   – generated images
# =============================================================================
set -euo pipefail

# ---------- Arguments ----------
ROUTER_CKPT="${1:-checkpoints/router/router_best.pt}"
SEED="${2:-42}"
shift 2 2>/dev/null || true        # consume first two positional args if present
TAGS=("${@:-counting}")            # remaining args are tags; default to "counting"

# ---------- Derived paths ----------
TAG_STR=$(IFS=_; echo "${TAGS[*]}")
OUTDIR="outputs/geneval_router_${TAG_STR}_seed_${SEED}"
METADATA="benchmarks/geneval/prompts/evaluation_metadata.jsonl"

echo "============================================================"
echo "  GenEval Router Generation"
echo "============================================================"
echo "  router_ckpt : ${ROUTER_CKPT}"
echo "  tags        : ${TAGS[*]}"
echo "  seed        : ${SEED}"
echo "  outdir      : ${OUTDIR}"
echo "============================================================"

python benchmarks/geneval/generation/zimage_generate_trained_router.py \
    "${METADATA}" \
    --router_ckpt "${ROUTER_CKPT}" \
    --outdir      "${OUTDIR}" \
    --tags        "${TAGS[@]}" \
    --seed        "${SEED}"

echo ""
echo "============================================================"
echo "  Images saved to: ${OUTDIR}"
echo "============================================================"
