python train_text/train_router.py \
    --triplet_dir data/train_triplets \
    --output_dir checkpoints/router \
    --loss_type supcon \
    --temperature 0.07 \
    --batch_size 64 \
    --epochs 20 \
    --lambda_reg 0.1