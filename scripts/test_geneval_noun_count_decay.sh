python benchmarks/geneval/generation/zimage_generate_noun_count_decay.py \
  benchmarks/geneval/prompts/evaluation_metadata.jsonl \
  --outdir outputs/outputs_geneval_counting_noun_count_decay_L10-20_dr03 \
  --tags counting \
  --count_rs 10  --count_re 20  --count_dr 0.2 \
  --noun_rs  10  --noun_re  20  --noun_dr  0.2 \
  --seed 42
