"""
Standalone diagnostic script: inspect text embeddings before and after context_refiner.

Usage:
    python train_text/diagnose_embeddings.py \
        --model_dir ckpts/Z-Image-Turbo \
        --triplets_jsonl data/train_triplets/counting_triplets_filtered.jsonl \
        --generated_root data/generated_images \
        --num_samples 32 \
        --num_negatives 9 \
        --device cuda

For each sampled anchor the script prints:
  - The actual anchor / positive / negative texts
  - Which tokens are included in the content mask (non-special)
  - pre-refiner  pos_sim / neg_sim / p-n
  - post-refiner pos_sim / neg_sim / p-n
  - Summary statistics over all samples
"""

import argparse
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from utils import load_from_local_dir  # noqa: E402

# ── reuse data utilities from training script ──────────────────────────────────
sys.path.append(os.path.dirname(__file__))
from train_counting_contrastive_diffusion import (   # noqa: E402
    CountingVerdictDataset,
    run_context_refiner,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def tokenize(tokenizer, texts, max_length=128, device="cpu"):
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return enc.input_ids.to(device), enc.attention_mask.to(device)


def content_mean_pool(h: torch.Tensor, ids: torch.Tensor,
                      mask: torch.Tensor, special_ids: set) -> torch.Tensor:
    """Mean pool over non-special, non-padding tokens, then L2-normalise. Returns [B, D]."""
    is_special = torch.zeros_like(ids, dtype=torch.bool)
    for sid in special_ids:
        is_special |= (ids == sid)
    content_mask = mask.bool() & ~is_special          # [B, seq]
    fm = content_mask.unsqueeze(-1).float()
    pooled = (h * fm).sum(1) / fm.sum(1).clamp(min=1.0)
    return F.normalize(pooled.float(), dim=-1)


def show_content_tokens(tokenizer, ids: torch.Tensor,
                        mask: torch.Tensor, special_ids: set) -> str:
    """Return a string listing the content (non-special, non-pad) tokens."""
    is_special = torch.zeros_like(ids, dtype=torch.bool)
    for sid in special_ids:
        is_special |= (ids == sid)
    content_mask = mask.bool() & ~is_special
    toks = [tokenizer.decode([ids[i].item()]) for i in range(len(ids))
            if content_mask[i].item()]
    return " | ".join(toks)


@torch.no_grad()
def embed(text_encoder, transformer, tokenizer, texts, special_ids, device, max_length=128):
    """Return (pre_pool [B,D], post_pool [B,D]) for a list of texts."""
    ids, mask = tokenize(tokenizer, texts, max_length, device)

    out = text_encoder(input_ids=ids, attention_mask=mask.bool(), output_hidden_states=True)
    h = out.hidden_states[-2].float()           # [B, seq, D_enc]

    pre_pool = content_mean_pool(h, ids, mask, special_ids)

    h_ref = run_context_refiner(transformer, h.to(next(transformer.parameters()).dtype), mask).float()
    post_pool = content_mean_pool(h_ref, ids, mask, special_ids)

    return pre_pool, post_pool, ids, mask


# ── main ───────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device(args.device)

    print(f"[Load] base model from {args.model_dir}")
    components = load_from_local_dir(
        args.model_dir, device="cpu", dtype=torch.bfloat16, verbose=True,
    )
    transformer  = components["transformer"].to(device).eval()
    text_encoder = components["text_encoder"].to(device).eval()
    tokenizer    = components["tokenizer"]

    for p in transformer.parameters():
        p.requires_grad_(False)
    for p in text_encoder.parameters():
        p.requires_grad_(False)

    if args.checkpoint:
        print(f"[Load] checkpoint from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        if "transformer_state_dict" in ckpt:
            missing, unexpected = transformer.load_state_dict(ckpt["transformer_state_dict"], strict=False)
            print(f"  transformer: {len(missing)} missing keys, {len(unexpected)} unexpected keys")
            if missing:
                print(f"  missing (first 5): {missing[:5]}")
        else:
            print("  [WARNING] no 'transformer_state_dict' found in checkpoint")
        ckpt_step = ckpt.get("step", "?")
        ckpt_epoch = ckpt.get("epoch", "?")
        print(f"  checkpoint step={ckpt_step}  epoch={ckpt_epoch}")
    else:
        print("[Load] no checkpoint given — using base pretrained weights")

    special_ids = set(tokenizer.all_special_ids)

    print(f"\n[Special token ids] ({len(special_ids)} total): {sorted(special_ids)[:10]} ...")
    example_special = [tokenizer.decode([sid]) for sid in sorted(special_ids)[:10]]
    print(f"[Special tokens decoded]: {example_special}")

    # ── dataset ────────────────────────────────────────────────────────────────
    print(f"\n[Dataset] loading {args.triplets_jsonl}")
    dataset = CountingVerdictDataset(
        triplets_jsonl=args.triplets_jsonl,
        generated_root=args.generated_root,
        threshold=0.8,
        resolution=512,
        num_negatives=args.num_negatives,
    )
    print(f"[Dataset] {len(dataset)} samples")

    # ── sample ─────────────────────────────────────────────────────────────────
    random.seed(args.seed)
    items = dataset.sample_text_batch(args.num_samples)

    pre_pns, post_pns = [], []

    for i, item in enumerate(items):
        anchor   = item["anchor"]
        positive = item["positive"]
        negs     = item["negatives"][:args.num_negatives]

        all_texts = [anchor, positive] + negs

        pre_all, post_all, ids_all, mask_all = embed(
            text_encoder, transformer, tokenizer, all_texts, special_ids, device,
        )

        ea_pre,  ep_pre,  en_pre  = pre_all[0],  pre_all[1],  pre_all[2:]
        ea_post, ep_post, en_post = post_all[0], post_all[1], post_all[2:]

        pre_pos_sim  = (ea_pre  * ep_pre ).sum().item()
        pre_neg_sim  = (ea_pre.unsqueeze(0)  * en_pre ).sum(-1).mean().item()
        post_pos_sim = (ea_post * ep_post).sum().item()
        post_neg_sim = (ea_post.unsqueeze(0) * en_post).sum(-1).mean().item()

        pre_pn  = pre_pos_sim  - pre_neg_sim
        post_pn = post_pos_sim - post_neg_sim
        pre_pns.append(pre_pn)
        post_pns.append(post_pn)

        if i < args.show_n:
            print(f"\n{'='*70}")
            print(f"[Sample {i}]")
            print(f"  anchor  : {anchor}")
            print(f"  positive: {positive}")
            print(f"  neg[0]  : {negs[0]}")
            print(f"  neg[1]  : {negs[1] if len(negs) > 1 else '-'}")

            a_ids, a_mask = ids_all[0], mask_all[0]
            p_ids, p_mask = ids_all[1], mask_all[1]
            n_ids, n_mask = ids_all[2], mask_all[2]

            print(f"\n  content tokens (anchor)  : {show_content_tokens(tokenizer, a_ids, a_mask, special_ids)}")
            print(f"  content tokens (positive): {show_content_tokens(tokenizer, p_ids, p_mask, special_ids)}")
            print(f"  content tokens (neg[0])  : {show_content_tokens(tokenizer, n_ids, n_mask, special_ids)}")

            print(f"\n  PRE-refiner   pos_sim={pre_pos_sim:.4f}  neg_sim={pre_neg_sim:.4f}  p-n={pre_pn:+.4f}")
            print(f"  POST-refiner  pos_sim={post_pos_sim:.4f}  neg_sim={post_neg_sim:.4f}  p-n={post_pn:+.4f}")

    # ── summary ────────────────────────────────────────────────────────────────
    import statistics
    print(f"\n{'='*70}")
    print(f"[Summary over {args.num_samples} samples]")
    print(f"  PRE-refiner   mean p-n = {statistics.mean(pre_pns):+.4f}  "
          f"std = {statistics.stdev(pre_pns):.4f}  "
          f"pos_rate = {sum(x>0 for x in pre_pns)/len(pre_pns):.2%}")
    print(f"  POST-refiner  mean p-n = {statistics.mean(post_pns):+.4f}  "
          f"std = {statistics.stdev(post_pns):.4f}  "
          f"pos_rate = {sum(x>0 for x in post_pns)/len(post_pns):.2%}")

    # ── p-n histogram ──────────────────────────────────────────────────────────
    bins = [-0.1, -0.05, -0.02, -0.01, 0.0, 0.01, 0.02, 0.05, 0.1, float("inf")]
    def histogram(vals, bins):
        counts = [0] * (len(bins) - 1)
        for v in vals:
            for j in range(len(bins) - 1):
                if bins[j] <= v < bins[j+1]:
                    counts[j] += 1
                    break
        return counts

    pre_hist  = histogram(pre_pns,  bins)
    post_hist = histogram(post_pns, bins)
    print(f"\n  p-n bucket distribution (pre / post):")
    for j in range(len(bins) - 1):
        lo = f"{bins[j]:+.2f}" if bins[j] != float("-inf") else " -inf"
        hi = f"{bins[j+1]:+.2f}" if bins[j+1] != float("inf") else "  +inf"
        bar_pre  = "█" * pre_hist[j]
        bar_post = "█" * post_hist[j]
        print(f"  [{lo}, {hi})  pre:{pre_hist[j]:3d} {bar_pre:<20s}  post:{post_hist[j]:3d} {bar_post}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir",      type=str, default="ckpts/Z-Image-Turbo")
    p.add_argument("--checkpoint",     type=str, default="",
                   help="Path to .pt checkpoint saved by training script (loads transformer_state_dict)")
    p.add_argument("--triplets_jsonl", type=str,
                   default="data/train_triplets/counting_triplets_filtered.jsonl")
    p.add_argument("--generated_root", type=str, default="data/generated_images")
    p.add_argument("--num_samples",    type=int, default=64,
                   help="Number of anchor/pos/neg triplets to evaluate")
    p.add_argument("--num_negatives",  type=int, default=9)
    p.add_argument("--show_n",         type=int, default=5,
                   help="Number of individual samples to print in detail")
    p.add_argument("--device",         type=str, default="cuda")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--max_length",     type=int, default=128)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
