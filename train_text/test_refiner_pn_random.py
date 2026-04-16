"""
Random p-n verification for refiner embeddings.

For each trial:
  - sample B triplets: (anchor, positive, negative)
  - encode pre-refiner and post-refiner sentence embeddings
  - compute p-n = sim(anchor, positive) - sim(anchor, negative)
  - report per-trial and overall statistics
"""

import argparse
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from utils import load_from_local_dir  # noqa: E402

def run_context_refiner(transformer: torch.nn.Module, token_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Project text features to transformer dim, then run context_refiner blocks."""
    model = transformer.module if hasattr(transformer, "module") else transformer
    bsz, seq_len, _ = token_hidden.shape
    device = token_hidden.device
    dtype = next(model.parameters()).dtype
    attn_mask = attention_mask.bool()

    cap_feats = model.cap_embedder(token_hidden.to(dtype))
    cap_feats = cap_feats.clone()
    cap_feats[~attn_mask] = model.cap_pad_token.to(dtype)

    pos_ids = torch.zeros((bsz, seq_len, 3), dtype=torch.int32, device=device)
    pos_ids[:, :, 0] = torch.arange(1, seq_len + 1, dtype=torch.int32, device=device).unsqueeze(0).expand(bsz, -1)
    cap_freqs = model.rope_embedder(pos_ids.view(-1, 3)).view(bsz, seq_len, -1)

    refined = cap_feats
    for layer in model.context_refiner:
        refined = layer(refined, attn_mask, cap_freqs)
    return refined


CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789"


def rand_gibberish(min_len: int = 6, max_len: int = 14) -> str:
    n = random.randint(min_len, max_len)
    return "".join(random.choice(CHARSET) for _ in range(n))


def build_triplet() -> tuple[str, str, str]:
    """
    Build a noisy/gibberish triplet for stress testing anisotropy:
      anchor:   "xxxdasda qwe12"
      positive: "a photo of xxxdasda qwe12"      (keeps anchor core)
      negative: "a photo of asd98zx qqq77"       (different gibberish core)
    """
    core_a = f"{rand_gibberish()} {rand_gibberish()}"
    core_n = f"{rand_gibberish()} {rand_gibberish()}{rand_gibberish()}{rand_gibberish()}"
    anchor = core_a
    positive = f"a photo of {core_a}"
    negative = f"a photo of {core_n}"
    return anchor, positive, negative


def pool_content(token_hidden: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, special_ids: set[int]) -> torch.Tensor:
    is_special = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in special_ids:
        is_special |= (input_ids == sid)
    content_mask = attention_mask.bool() & ~is_special
    m = content_mask.unsqueeze(-1).float()
    pooled = (token_hidden.float() * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    return F.normalize(pooled, dim=-1)


def mean_std_min_max(x: torch.Tensor) -> tuple[float, float, float, float]:
    return float(x.mean()), float(x.std()), float(x.min()), float(x.max())


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    components = load_from_local_dir(
        args.model_dir,
        device="cpu",
        dtype=torch.bfloat16,
        verbose=True,
    )
    transformer = components["transformer"].to(device).eval()
    text_encoder = components["text_encoder"].to(device).eval()
    tokenizer = components["tokenizer"]

    for p in transformer.parameters():
        p.requires_grad_(False)
    for p in text_encoder.parameters():
        p.requires_grad_(False)

    special_ids = set(tokenizer.all_special_ids)
    all_pre = []
    all_post = []

    for t in range(args.trials):
        triplets = [build_triplet() for _ in range(args.batch_size)]
        anchors = [x[0] for x in triplets]
        positives = [x[1] for x in triplets]
        negatives = [x[2] for x in triplets]

        texts = anchors + positives + negatives
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)

        with torch.no_grad():
            out = text_encoder(input_ids=input_ids, attention_mask=attention_mask.bool(), output_hidden_states=True)
            pre_tok = out.hidden_states[-2]
            post_tok = run_context_refiner(transformer, pre_tok, attention_mask)

            pre_vec = pool_content(pre_tok, input_ids, attention_mask, special_ids)
            post_vec = pool_content(post_tok, input_ids, attention_mask, special_ids)

        b = args.batch_size
        ea_pre, ep_pre, en_pre = pre_vec[:b], pre_vec[b:2 * b], pre_vec[2 * b:]
        ea_post, ep_post, en_post = post_vec[:b], post_vec[b:2 * b], post_vec[2 * b:]

        pn_pre = (ea_pre * ep_pre).sum(dim=-1) - (ea_pre * en_pre).sum(dim=-1)
        pn_post = (ea_post * ep_post).sum(dim=-1) - (ea_post * en_post).sum(dim=-1)

        all_pre.append(pn_pre.cpu())
        all_post.append(pn_post.cpu())

        pre_stats = mean_std_min_max(pn_pre)
        post_stats = mean_std_min_max(pn_post)
        print(
            f"[trial {t+1:02d}/{args.trials}] "
            f"pre_p-n mean={pre_stats[0]:+.6f} std={pre_stats[1]:.6f} min={pre_stats[2]:+.6f} max={pre_stats[3]:+.6f} | "
            f"post_p-n mean={post_stats[0]:+.6f} std={post_stats[1]:.6f} min={post_stats[2]:+.6f} max={post_stats[3]:+.6f}"
        )

    all_pre_t = torch.cat(all_pre, dim=0)
    all_post_t = torch.cat(all_post, dim=0)
    pre_all = mean_std_min_max(all_pre_t)
    post_all = mean_std_min_max(all_post_t)

    print("\n=== Overall ===")
    print(
        f"PRE  p-n mean={pre_all[0]:+.6f} std={pre_all[1]:.6f} min={pre_all[2]:+.6f} max={pre_all[3]:+.6f} "
        f"pos_rate={(all_pre_t > 0).float().mean().item() * 100:.2f}%"
    )
    print(
        f"POST p-n mean={post_all[0]:+.6f} std={post_all[1]:.6f} min={post_all[2]:+.6f} max={post_all[3]:+.6f} "
        f"pos_rate={(all_post_t > 0).float().mean().item() * 100:.2f}%"
    )

    print("\nfirst 10 PRE p-n :", [round(float(x), 6) for x in all_pre_t[:10]])
    print("first 10 POST p-n:", [round(float(x), 6) for x in all_post_t[:10]])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default="ckpts/Z-Image-Turbo")
    p.add_argument("--trials", type=int, default=10, help="How many random batches to test")
    p.add_argument("--batch_size", type=int, default=32, help="Triplets per trial")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

