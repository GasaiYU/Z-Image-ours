"""
Quick diagnostic for refiner embedding similarity.

What it does:
1) Randomly sample prompts (or read from a text file)
2) Encode with frozen text_encoder -> pre-refiner embeddings
3) Pass through transformer.context_refiner -> post-refiner embeddings
4) Compare pairwise cosine similarity statistics (off-diagonal only)
"""

import argparse
import os
import random
import sys
from typing import List

import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from utils import load_from_local_dir  # noqa: E402


NUMBER_WORDS = [
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
]

OBJECT_WORDS = [
    "cats", "dogs", "apples", "cars", "chairs", "tables", "books", "bicycles",
    "cups", "candles", "kites", "flowers", "hammers", "backpacks", "bottles",
    "socks", "pencils", "oranges", "birds", "umbrellas", "laptops", "shoes",
]

PROMPT_TEMPLATES = [
    "{n} {o}",
    "a photo of {n} {o}",
    "an image of {n} {o}",
    "a realistic photo of {n} {o}",
    "{n} {o} on a plain background",
    "{n} {o}, studio photography",
    "a detailed image of {n} {o}",
    "a clear photo of {n} {o}",
]


def format_prompt(tokenizer, text: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return text
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


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


def mean_pool_content(token_hidden: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, special_ids: set[int]) -> torch.Tensor:
    is_special = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in special_ids:
        is_special |= (input_ids == sid)
    content_mask = attention_mask.bool() & ~is_special
    float_mask = content_mask.unsqueeze(-1).float()
    denom = float_mask.sum(dim=1).clamp(min=1.0)
    pooled = (token_hidden.float() * float_mask).sum(dim=1) / denom
    return F.normalize(pooled, dim=-1)


def sample_prompts(num_samples: int, seed: int) -> List[str]:
    random.seed(seed)
    prompts: List[str] = []
    for _ in range(num_samples):
        n = random.choice(NUMBER_WORDS)
        o = random.choice(OBJECT_WORDS)
        t = random.choice(PROMPT_TEMPLATES)
        prompts.append(t.format(n=n, o=o))
    return prompts


def pairwise_stats(features: torch.Tensor) -> tuple[torch.Tensor, dict]:
    sim = features @ features.t()  # cosine because features are normalised
    n = sim.shape[0]
    off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    vals = sim[off_diag_mask]
    stats = {
        "mean": vals.mean().item(),
        "std": vals.std().item(),
        "min": vals.min().item(),
        "max": vals.max().item(),
        "p10": torch.quantile(vals, 0.10).item(),
        "p50": torch.quantile(vals, 0.50).item(),
        "p90": torch.quantile(vals, 0.90).item(),
    }
    return sim, stats


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

    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            raw_prompts = [x.strip() for x in f.readlines() if x.strip()]
        if len(raw_prompts) < args.num_samples:
            raise ValueError(f"prompts_file only has {len(raw_prompts)} lines, need >= {args.num_samples}")
        prompts = random.sample(raw_prompts, k=args.num_samples)
    else:
        prompts = sample_prompts(args.num_samples, args.seed)

    prompts = [format_prompt(tokenizer, p, args.use_chat_template) for p in prompts]

    enc = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    special_ids = set(tokenizer.all_special_ids)

    with torch.no_grad():
        out = text_encoder(input_ids=input_ids, attention_mask=attention_mask.bool(), output_hidden_states=True)
        pre_tokens = out.hidden_states[-2].detach().clone()
        post_tokens = run_context_refiner(transformer, pre_tokens, attention_mask)

        pre_vec = mean_pool_content(pre_tokens, input_ids, attention_mask, special_ids)
        post_vec = mean_pool_content(post_tokens, input_ids, attention_mask, special_ids)

    pre_sim, pre_stats = pairwise_stats(pre_vec)
    post_sim, post_stats = pairwise_stats(post_vec)

    print("\n=== Pairwise cosine stats (off-diagonal) ===")
    print(
        f"PRE  : mean={pre_stats['mean']:.6f} std={pre_stats['std']:.6f} "
        f"min={pre_stats['min']:.6f} p10={pre_stats['p10']:.6f} "
        f"p50={pre_stats['p50']:.6f} p90={pre_stats['p90']:.6f} max={pre_stats['max']:.6f}"
    )
    print(
        f"POST : mean={post_stats['mean']:.6f} std={post_stats['std']:.6f} "
        f"min={post_stats['min']:.6f} p10={post_stats['p10']:.6f} "
        f"p50={post_stats['p50']:.6f} p90={post_stats['p90']:.6f} max={post_stats['max']:.6f}"
    )

    n = pre_sim.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=pre_sim.device)
    idx_pairs = mask.nonzero(as_tuple=False)
    post_vals = post_sim[mask]

    topk = min(args.topk_pairs, post_vals.numel())
    top_idx = torch.topk(post_vals, k=topk, largest=True).indices
    low_idx = torch.topk(post_vals, k=topk, largest=False).indices

    print(f"\n=== Top-{topk} most similar pairs (POST) ===")
    for i in top_idx.tolist():
        a, b = idx_pairs[i].tolist()
        print(f"[{a:03d},{b:03d}] pre={pre_sim[a,b].item():.6f} post={post_sim[a,b].item():.6f} | {prompts[a]}  <->  {prompts[b]}")

    print(f"\n=== Top-{topk} least similar pairs (POST) ===")
    for i in low_idx.tolist():
        a, b = idx_pairs[i].tolist()
        print(f"[{a:03d},{b:03d}] pre={pre_sim[a,b].item():.6f} post={post_sim[a,b].item():.6f} | {prompts[a]}  <->  {prompts[b]}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default="ckpts/Z-Image-Turbo")
    p.add_argument("--num_samples", type=int, default=64)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompts_file", type=str, default="")
    p.add_argument("--topk_pairs", type=int, default=10)
    p.add_argument("--use_chat_template", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
