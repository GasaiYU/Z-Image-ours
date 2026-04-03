"""
Visualize per-token routing weights of counting tokens across LLM layers.
=========================================================================

Loads a trained DynamicTokenRouter checkpoint, runs a set of prompts that
contain counting words, and plots the softmax routing weights (= "which LLM
layer does each token attend to?") for:

  - counting tokens  (one / two / three / …)
  - noun tokens      (object words in the same prompt)
  - other tokens     (articles, prepositions, …)

Produces three output files in --out_dir:
  1. heatmap_<prompt_idx>.png   – per-token × per-layer weight heatmap for
                                   each prompt (columns = layers, rows = tokens)
  2. avg_by_type.png            – averaged weight curves across layers,
                                   grouped by token type (counting / noun / other)
  3. weight_stats.json          – raw numbers for further analysis

Usage:
    python train_text/visualize_router_weights.py \
        --router_ckpt checkpoints/router/router_best.pt \
        --model_dir   ckpts/Z-Image-Turbo \
        --out_dir     outputs/router_vis

    # Custom prompts:
    python train_text/visualize_router_weights.py \
        --router_ckpt checkpoints/router/router_best.pt \
        --prompts "two red apples" "three blue birds on a branch"
"""

import argparse
import json
import os
import re
import sys

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils import ensure_model_weights, load_from_local_dir
from train_router import DynamicTokenRouter   # same file, same dir


# ---------------------------------------------------------------------------
# Word banks (mirror train_router.py / zimage_generate_decay.py)
# ---------------------------------------------------------------------------
COUNTING_BANK = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
}
# Common nouns used in counting prompts – extended list
NOUN_BANK = {
    "apple", "apples", "cat", "cats", "dog", "dogs", "bird", "birds",
    "cup", "cups", "ball", "balls", "car", "cars", "tree", "trees",
    "flower", "flowers", "book", "books", "chair", "chairs", "box", "boxes",
    "star", "stars", "cloud", "clouds", "fish", "fishes", "boat", "boats",
    "house", "houses", "plane", "planes", "bottle", "bottles", "lamp", "lamps",
    "person", "people", "child", "children", "horse", "horses", "sheep",
    "banana", "bananas", "orange", "oranges", "strawberry", "strawberries",
    "butterfly", "butterflies", "elephant", "elephants", "bear", "bears",
}

DEFAULT_PROMPTS = [
    "two cats sitting on a sofa",
    "three red apples on a table",
    "four birds flying in the sky",
    "five yellow stars",
    "one blue bottle on a shelf",
    "six dogs running in a park",
    "two children playing with a ball",
    "three boats on the ocean",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def classify_token(token_str: str) -> str:
    """Return 'counting', 'noun', or 'other' for a decoded token string."""
    clean = token_str.lower().strip().replace(" ", "").replace("▁", "")
    if clean in COUNTING_BANK:
        return "counting"
    if clean in NOUN_BANK:
        return "noun"
    return "other"


def find_content_span(tokens: list[str]):
    """
    Locate the start/end of the actual prompt content inside the Qwen chat
    template tokens (skip system/role boilerplate).
    Returns (content_start, content_end) indices into `tokens`.
    """
    content_start, content_end = 0, len(tokens)
    for i, t in enumerate(tokens):
        if "user" in t.lower():
            content_start = i + 1
        elif "<|im_end|>" in t and i > content_start:
            content_end = i
            break
    return content_start, content_end


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------
@torch.no_grad()
def analyze_prompt(
    prompt: str,
    text_encoder,
    tokenizer,
    router: DynamicTokenRouter,
    device,
    max_sequence_length: int = 512,
):
    """
    Run one prompt through the text encoder + router and return:
        tokens       : list[str]           content tokens (no padding)
        token_types  : list[str]           'counting' | 'noun' | 'other'
        weights      : np.ndarray [T, L]   softmax routing weights per token
    """
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    enc = tokenizer(
        [formatted],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = enc.input_ids.to(device)        # [1, S]
    attention_mask = enc.attention_mask.to(device)   # [1, S]

    outputs = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    all_hs = outputs.hidden_states   # tuple of (L+1) tensors [1, S, D]

    # Router forward (eval mode, no grad)
    router.eval()
    _, routing_weights = router(all_hs, attention_mask=attention_mask.bool())
    # routing_weights: [1, S, num_layers]

    # Only keep valid (non-padding) tokens
    valid_mask = attention_mask[0].bool()   # [S]
    valid_ids  = input_ids[0][valid_mask]   # [T]
    rw         = routing_weights[0][valid_mask].float().cpu().numpy()   # [T, L]

    # Decode tokens
    all_tokens = [tokenizer.decode([tid]) for tid in valid_ids.tolist()]
    cs, ce = find_content_span(all_tokens)
    tokens      = all_tokens[cs:ce]
    weights     = rw[cs:ce]               # [T_content, L]
    token_types = [classify_token(t) for t in tokens]

    return tokens, token_types, weights


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def plot_heatmap(tokens, token_types, weights, prompt, save_path):
    """
    Heatmap: rows = content tokens, columns = LLM layers.
    Counting tokens are highlighted in red on the y-axis.
    """
    T, L = weights.shape
    layer_labels = [str(i + 1) for i in range(L)]   # 1-indexed

    fig, ax = plt.subplots(figsize=(max(12, L * 0.35), max(4, T * 0.4)))
    im = ax.imshow(weights, aspect="auto", cmap="viridis",
                   vmin=0, vmax=weights.max())

    ax.set_xticks(range(L))
    ax.set_xticklabels(layer_labels, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(T))

    # Color y-tick labels by token type
    yticklabels = ax.set_yticklabels(
        [t.strip() or "·" for t in tokens], fontsize=8
    )
    for label, ttype in zip(yticklabels, token_types):
        if ttype == "counting":
            label.set_color("red")
            label.set_fontweight("bold")
        elif ttype == "noun":
            label.set_color("steelblue")

    ax.set_xlabel("LLM Layer", fontsize=10)
    ax.set_ylabel("Token  (red=counting, blue=noun)", fontsize=9)
    ax.set_title(f'Routing weights — "{prompt}"', fontsize=10, pad=8)

    plt.colorbar(im, ax=ax, label="Softmax weight", fraction=0.02, pad=0.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_avg_by_type(all_results, num_layers, save_path):
    """
    Average routing weight curve per layer, grouped by token type.
    Shows the mean ± std across all prompts.
    """
    buckets = {"counting": [], "noun": [], "other": []}
    for tokens, token_types, weights in all_results:
        for ttype, w_row in zip(token_types, weights):
            buckets[ttype].append(w_row)

    colors = {"counting": "#e74c3c", "noun": "#2980b9", "other": "#7f8c8d"}
    x = np.arange(1, num_layers + 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    for ttype, color in colors.items():
        rows = np.array(buckets[ttype])   # [N, L]
        if len(rows) == 0:
            continue
        mean = rows.mean(axis=0)
        std  = rows.std(axis=0)
        ax.plot(x, mean, color=color, linewidth=2,
                label=f"{ttype}  (n={len(rows)} tokens)")
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("LLM Layer", fontsize=11)
    ax.set_ylabel("Average Routing Weight (softmax)", fontsize=11)
    ax.set_title("Counting vs. Noun vs. Other — Average Router Weight per Layer", fontsize=12)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.grid(axis="x", which="major", linestyle="--", alpha=0.4)
    ax.grid(axis="y", linestyle=":", alpha=0.3)

    # Mark the "deep" layer ([-2] = num_layers - 1, 1-indexed = num_layers)
    deep_layer = num_layers - 1    # 0-indexed among transformer layers; 1-indexed = num_layers
    ax.axvline(deep_layer, color="gray", linestyle=":", linewidth=1.2,
               label=f"default deep layer ({deep_layer})")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {save_path}")


def plot_peak_layer_hist(all_results, num_layers, save_path):
    """
    Histogram of peak-weight layers, split by token type.
    Shows which layer the router "prefers most" for each token type.
    """
    buckets = {"counting": [], "noun": [], "other": []}
    for tokens, token_types, weights in all_results:
        for ttype, w_row in zip(token_types, weights):
            peak = int(np.argmax(w_row)) + 1    # 1-indexed
            buckets[ttype].append(peak)

    colors = {"counting": "#e74c3c", "noun": "#2980b9", "other": "#7f8c8d"}
    bins   = np.arange(0.5, num_layers + 1.5, 1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
    for ax, (ttype, color) in zip(axes, colors.items()):
        data = buckets[ttype]
        if data:
            ax.hist(data, bins=bins, color=color, alpha=0.8, edgecolor="white")
            ax.axvline(np.mean(data), color="black", linestyle="--",
                       linewidth=1.5, label=f"mean={np.mean(data):.1f}")
            ax.legend(fontsize=9)
        ax.set_title(f"{ttype}  (n={len(data)})", color=color, fontsize=11)
        ax.set_xlabel("Peak Layer", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_xlim(0.5, num_layers + 0.5)

    fig.suptitle("Distribution of Peak Routing Layer per Token Type", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Visualize DynamicTokenRouter weights for counting tokens")
    p.add_argument("--router_ckpt",  type=str, required=True,
                   help="Path to trained router checkpoint (.pt)")
    p.add_argument("--model_dir",    type=str, default="ckpts/Z-Image-Turbo",
                   help="Z-Image model directory (same as training)")
    p.add_argument("--out_dir",      type=str, default="outputs/router_vis",
                   help="Directory to save visualizations")
    p.add_argument("--prompts",      type=str, nargs="+", default=None,
                   help="Custom prompts to analyze (default: built-in counting prompts)")
    p.add_argument("--max_length",   type=int, default=512,
                   help="Max tokenizer sequence length")
    p.add_argument("--no_heatmaps",  action="store_true",
                   help="Skip per-prompt heatmaps (faster for many prompts)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # ---- Load text encoder + tokenizer ----
    print("[Init] Loading text encoder …")
    model_path = ensure_model_weights(args.model_dir, verify=False)
    components = load_from_local_dir(model_path, device=device, dtype=torch.bfloat16, verbose=True)
    text_encoder = components["text_encoder"]
    tokenizer    = components["tokenizer"]
    del components["transformer"], components["vae"], components["scheduler"]
    import gc; gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad_(False)

    # ---- Load router ----
    print(f"[Init] Loading router from {args.router_ckpt} …")
    ckpt = torch.load(args.router_ckpt, map_location="cpu", weights_only=False)
    hidden_size = ckpt["hidden_size"]
    num_layers  = ckpt["num_layers"]
    mid_dim     = ckpt["mid_dim"]

    router = DynamicTokenRouter(hidden_size=hidden_size, num_layers=num_layers, mid_dim=mid_dim)
    state_dict = ckpt["router_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    router.load_state_dict(state_dict)
    router.to(device)
    router.eval()

    print(f"[Router] hidden_size={hidden_size}, num_layers={num_layers}, mid_dim={mid_dim}")
    if "epoch" in ckpt:
        print(f"[Router] epoch={ckpt['epoch']}, step={ckpt.get('step','?')}, "
              f"best_loss={ckpt.get('best_loss','N/A')}")

    # ---- Prompts ----
    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    print(f"\n[Analysis] {len(prompts)} prompts")

    # ---- Per-prompt analysis ----
    all_results = []
    stats = []

    for idx, prompt in enumerate(prompts):
        print(f"  [{idx + 1:>2}/{len(prompts)}] {prompt}")
        tokens, token_types, weights = analyze_prompt(
            prompt, text_encoder, tokenizer, router, device, args.max_length
        )
        all_results.append((tokens, token_types, weights))

        # Console summary for counting tokens in this prompt
        counting_idxs = [i for i, t in enumerate(token_types) if t == "counting"]
        for ci in counting_idxs:
            peak_layer = int(np.argmax(weights[ci])) + 1    # 1-indexed
            peak_w     = float(weights[ci].max())
            top3       = np.argsort(weights[ci])[-3:][::-1] + 1
            print(f"    counting token '{tokens[ci].strip()}': "
                  f"peak=layer {peak_layer} (w={peak_w:.3f}), "
                  f"top-3 layers={top3.tolist()}")

        # Save heatmap
        if not args.no_heatmaps:
            hmap_path = os.path.join(args.out_dir, f"heatmap_{idx:02d}.png")
            plot_heatmap(tokens, token_types, weights, prompt, hmap_path)
            print(f"    [Saved] {hmap_path}")

        # Accumulate stats
        for token, ttype, w in zip(tokens, token_types, weights):
            stats.append({
                "prompt": prompt,
                "token": token.strip(),
                "type": ttype,
                "peak_layer": int(np.argmax(w)) + 1,
                "peak_weight": float(w.max()),
                "weights": w.tolist(),
            })

    # ---- Aggregate plots ----
    print("\n[Plotting] Average weight by token type …")
    plot_avg_by_type(
        all_results, num_layers,
        os.path.join(args.out_dir, "avg_by_type.png"),
    )

    print("[Plotting] Peak-layer distribution …")
    plot_peak_layer_hist(
        all_results, num_layers,
        os.path.join(args.out_dir, "peak_layer_hist.png"),
    )

    # ---- Save raw stats ----
    stats_path = os.path.join(args.out_dir, "weight_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[Saved] {stats_path}")

    # ---- Print global summary ----
    print("\n========== Global Summary ==========")
    for ttype in ("counting", "noun", "other"):
        rows = [s for s in stats if s["type"] == ttype]
        if not rows:
            continue
        peaks = [s["peak_layer"] for s in rows]
        print(f"  {ttype:<10}  n={len(rows):>4}  "
              f"mean_peak_layer={np.mean(peaks):.2f}  "
              f"std={np.std(peaks):.2f}  "
              f"median={np.median(peaks):.1f}")
    print("====================================")
    print(f"\n[Done] All outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
