python benchmarks/geneval/generation/zimage_generate_decay.py \
  benchmarks/geneval/prompts/evaluation_metadata.jsonl \
  --outdir outputs/outputs_geneval_counting_decay_L10-20_rate02_seed420 \
  --tags counting \
  --decay_rate 0.2 \
  --route_start 10 \
  --route_end 21 \
  --target_type quantity \
  --seed 420
