python train_text/train_router.py \
    --model_dir /mmu-vcg/gaomingju/workspace/T2I/Z-Image-ours/benchmarks/geneval/pretrained \
    --triplet_dir data/train_triplets \
    --output_dir checkpoints/router \
    --loss_type infonce \
    --temperature 0.07 \
    --batch_size 64 \
    --epochs 20 \
    --lambda_reg 0.1