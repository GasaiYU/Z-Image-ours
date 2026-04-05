"""
Visualize average routing weights of counting tokens across LLM layers.
========================================================================

Loads a trained DynamicTokenRouter checkpoint, runs a set of prompts that
contain counting words, and plots the mean ± std softmax routing weights
across all LLM layers **for counting tokens only**.

Output: counting_avg_weight.png  +  counting_weight_stats.json

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
from collections import defaultdict

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils import ensure_model_weights, load_from_local_dir
from train_router import DynamicTokenRouter


# ---------------------------------------------------------------------------
# Counting word bank
# ---------------------------------------------------------------------------
COUNTING_BANK = {
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
}

DEFAULT_PROMPTS = [
    # two
    "two cats sitting on a sofa",
    "two red apples on a table",
    "two birds flying in the sky",
    "two yellow stars on a wall",
    "two blue bottles on a shelf",
    "two dogs running in a park",
    "two children playing with a ball",
    "two boats on the ocean",
    "two elephants standing in a field",
    "two candles on a wooden table",
    # three
    "three red apples on a table",
    "three birds flying in the sky",
    "three cats sleeping on a couch",
    "three boats sailing on the lake",
    "three horses running in a field",
    "three glasses of water on a tray",
    "three balloons floating in the air",
    "three books stacked on a desk",
    "three fish swimming in a bowl",
    "three flowers in a vase",
    # four
    "four birds perched on a fence",
    "four cars parked on a street",
    "four candles on a birthday cake",
    "four puppies playing on the grass",
    "four oranges in a basket",
    "four kites flying in the blue sky",
    "four chairs around a round table",
    "four butterflies on a flower",
    "four stars above a mountain",
    "four ducks swimming in a pond",
    # five
    "five yellow stars in the night sky",
    "five apples hanging on a tree",
    "five dogs running on the beach",
    "five children playing in a playground",
    "five boats docked at a harbor",
    "five candles lit on a table",
    "five birds sitting on a wire",
    "five balloons tied to a post",
    "five fish in a glass tank",
    "five strawberries on a white plate",
    # six / one / mixed
    "six dogs running in a park",
    "six colorful kites in the sky",
    "six cupcakes on a baking tray",
    "one blue bottle on a shelf",
    "one cat sitting on a windowsill",
    "one red umbrella in the rain",
    "three red and two blue balls on the floor",
    "two cats and one dog on a sofa",
    "four birds and two butterflies in a garden",
    "five apples and three oranges in a bowl",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def is_counting_token(token_str: str) -> bool:
    clean = token_str.lower().strip().replace(" ", "").replace("▁", "")
    return clean in COUNTING_BANK


def find_content_span(tokens: list[str]):
    """Return (start, end) of actual prompt content within chat-template tokens."""
    content_start, content_end = 0, len(tokens)
    for i, t in enumerate(tokens):
        if "user" in t.lower():
            content_start = i + 1
        elif "<|im_end|>" in t and i > content_start:
            content_end = i
            break
    return content_start, content_end


# ---------------------------------------------------------------------------
# Core: extract routing weights for counting tokens in one prompt
# ---------------------------------------------------------------------------
@torch.no_grad()
def get_counting_weights(
    prompt: str,
    text_encoder,
    tokenizer,
    router: DynamicTokenRouter,
    device,
    max_sequence_length: int = 512,
) -> list[tuple[str, np.ndarray]]:
    """
    Returns a list of (token_str, weight_vector) for every counting token
    found in the prompt.  weight_vector is shape [num_layers].
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
    input_ids      = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)

    outputs = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    _, routing_weights, _ = router(outputs.hidden_states, attention_mask=attention_mask.bool())
    # routing_weights: [1, S, L]

    valid_mask = attention_mask[0].bool()
    valid_ids  = input_ids[0][valid_mask].tolist()
    rw         = routing_weights[0][valid_mask].float().cpu().numpy()   # [T, L]

    all_tokens = [tokenizer.decode([tid]) for tid in valid_ids]
    cs, ce     = find_content_span(all_tokens)

    results = []
    for i in range(cs, ce):
        if is_counting_token(all_tokens[i]):
            results.append((all_tokens[i].strip(), rw[i]))   # ([L],)
    return results


# ---------------------------------------------------------------------------
# Counting swap analysis
# ---------------------------------------------------------------------------
SWAP_TARGETS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def replace_counting_word(prompt: str, original: str, replacement: str) -> str:
    """Replace the first whole-word occurrence of `original` with `replacement` (case-insensitive)."""
    return re.sub(r'(?<!\w)' + re.escape(original) + r'(?!\w)',
                  replacement, prompt, count=1, flags=re.IGNORECASE)


@torch.no_grad()
def get_counting_token_features(
    prompt: str,
    text_encoder,
    tokenizer,
    router: DynamicTokenRouter,
    device,
    max_sequence_length: int = 512,
):
    """
    Run prompt through text encoder + router.
    Returns for every counting token found:
        (token_str, routing_weights[L], fused_embed[D])
    """
    messages  = [{"role": "user", "content": prompt}]
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
    input_ids      = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)

    outputs = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    fused_embeds, routing_weights, _ = router(
        outputs.hidden_states, attention_mask=attention_mask.bool()
    )
    # routing_weights: [1, S, L]   fused_embeds: [1, S, D]

    valid_mask = attention_mask[0].bool()
    valid_ids  = input_ids[0][valid_mask].tolist()
    rw         = routing_weights[0][valid_mask].float().cpu().numpy()   # [T, L]
    fe         = fused_embeds[0][valid_mask].float().cpu().numpy()      # [T, D]

    all_tokens = [tokenizer.decode([tid]) for tid in valid_ids]
    cs, ce     = find_content_span(all_tokens)

    results = []
    for i in range(cs, ce):
        if is_counting_token(all_tokens[i]):
            results.append((all_tokens[i].strip(), rw[i], fe[i]))
    return results


@torch.no_grad()
def get_decision_feat(
    prompt: str,
    text_encoder,
    tokenizer,
    router: DynamicTokenRouter,
    device,
    max_sequence_length: int = 512,
) -> list[tuple[str, np.ndarray]]:
    """
    Extract the raw decision_feat (MLP input = L2-normalised h_1) for every
    counting token in prompt.  Returns list of (token_str, feat [D]).
    """
    messages  = [{"role": "user", "content": prompt}]
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
    input_ids      = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)

    outputs = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    all_hs = outputs.hidden_states   # tuple [num_layers+1] of [1, S, D]

    # decision_feat = L2-normalised h_1  (mirrors train_router.py router.forward)
    h1 = all_hs[1][0].float().cpu().numpy()           # [S, D]
    norm = np.linalg.norm(h1, axis=-1, keepdims=True) + 1e-8
    decision_feat = h1 / norm                          # [S, D]

    valid_mask = attention_mask[0].bool().cpu()
    valid_ids  = input_ids[0][valid_mask].tolist()
    df_valid   = decision_feat[valid_mask.numpy()]     # [T, D]

    all_tokens = [tokenizer.decode([tid]) for tid in valid_ids]
    cs, ce     = find_content_span(all_tokens)

    results = []
    for i in range(cs, ce):
        if is_counting_token(all_tokens[i]):
            results.append((all_tokens[i].strip(), df_valid[i]))
    return results


def plot_decision_feat_similarity(variants_df, base_prompt, out_dir):
    """
    For each variant (count word swap), extract decision_feat (L2-normalised h_1)
    and plot a cosine-similarity heatmap across all count-word variants.
    """
    os.makedirs(out_dir, exist_ok=True)

    words = [v["word"] for v in variants_df]
    feats = np.stack([v["df"] for v in variants_df])   # [N, D]

    norms = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    sim   = norms @ norms.T   # [N, N]

    vmin = sim[~np.eye(len(words), dtype=bool)].min() - 0.01

    fig, ax = plt.subplots(figsize=(max(6, len(words) * 0.8), max(5, len(words) * 0.7)))
    sns.heatmap(sim, annot=True, fmt=".3f", cmap="coolwarm",
                xticklabels=words, yticklabels=words,
                vmin=vmin, vmax=1.0, ax=ax,
                linewidths=0.5, linecolor="gray")
    ax.set_title(
        f'decision_feat (L2-norm h₁) Cosine Similarity\nbase: "{base_prompt}"',
        fontsize=10,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    p = os.path.join(out_dir, "decision_feat_similarity.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {p}")

    # Console summary
    mask = ~np.eye(len(words), dtype=bool)
    off  = sim[mask]
    print(f"\n[Decision Feat h₁ Similarity]  "
          f"min={off.min():.4f}  max={off.max():.4f}  mean={off.mean():.4f}")


def run_swap_analysis(
    base_prompt: str,
    text_encoder,
    tokenizer,
    router: DynamicTokenRouter,
    device,
    max_sequence_length: int = 512,
    swap_targets: list = None,
):
    """
    Detect counting word(s) in base_prompt, then replace with every word in
    swap_targets. Collect routing weights + fused embeddings for each variant.

    Returns
    -------
    variants : list of dicts with keys
        'word'  – the counting word in this variant (str)
        'prompt' – full prompt (str)
        'rw'    – routing_weights [L] np.ndarray for the counting token
        'fe'    – fused_embed [D] np.ndarray for the counting token
    found_word : str – the original counting word detected
    """
    if swap_targets is None:
        swap_targets = SWAP_TARGETS

    # Detect which counting word is in the base prompt
    found_word = None
    for w in swap_targets:
        if re.search(r'(?<!\w)' + re.escape(w) + r'(?!\w)', base_prompt, re.IGNORECASE):
            found_word = w
            break
    if found_word is None:
        raise ValueError(f"No counting word from {swap_targets} found in: '{base_prompt}'")

    print(f"  Base counting word detected: '{found_word}'")

    variants = []
    for target in swap_targets:
        prompt = replace_counting_word(base_prompt, found_word, target)
        feats  = get_counting_token_features(
            prompt, text_encoder, tokenizer, router, device, max_sequence_length
        )
        if not feats:
            print(f"  [skip] counting token not found after swap → '{target}'")
            continue
        tok, rw, fe = feats[0]   # take the first counting token
        variants.append({"word": target, "prompt": prompt, "rw": rw, "fe": fe})
        peak = int(rw.argmax()) + 1
        print(f"  '{target}':  peak=layer {peak}  (w={rw.max():.3f})")

    return variants, found_word


def plot_swap_results(variants, base_prompt, found_word, num_layers, out_dir):
    """
    Three figures:
    1. Routing-weight curves per variant (one line per count word).
    2. Fused-embedding cosine-similarity heatmap across all variants.
    3. Peak-layer bar chart per variant.
    """
    os.makedirs(out_dir, exist_ok=True)
    words  = [v["word"]  for v in variants]
    rws    = np.stack([v["rw"] for v in variants])   # [N, L]
    fes    = np.stack([v["fe"] for v in variants])   # [N, D]
    x      = np.arange(1, num_layers)   # n_route = num_layers - 1
    cmap   = plt.get_cmap("tab10")
    deep_layer = num_layers - 1

    # ── 1. Routing weight curves ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (word, rw) in enumerate(zip(words, rws)):
        lw    = 2.5 if word == found_word else 1.5
        ls    = "-"  if word == found_word else "--"
        alpha = 1.0  if word == found_word else 0.7
        ax.plot(x, rw, color=cmap(i % 10), linewidth=lw,
                linestyle=ls, alpha=alpha, label=f'"{word}"')

    ax.axvline(deep_layer, color="#2c3e50", linestyle=":", linewidth=1.2,
               label=f"default deep layer ({deep_layer})")
    ax.set_xlabel("LLM Layer", fontsize=12)
    ax.set_ylabel("Routing Weight (softmax)", fontsize=12)
    ax.set_title(
        f"Routing Weights per Counting Word — Swap Variants\n"
        f'base: "{base_prompt}"  (original word: "{found_word}")',
        fontsize=11,
    )
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.grid(axis="x", which="major", linestyle="--", alpha=0.35)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    ax.set_xlim(0.5, num_layers - 0.5)
    plt.tight_layout()
    p = os.path.join(out_dir, "swap_routing_curves.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {p}")

    # ── 2. Fused-embedding cosine similarity heatmap ────────────────────────
    norms  = fes / (np.linalg.norm(fes, axis=1, keepdims=True) + 1e-8)
    sim    = norms @ norms.T   # [N, N]

    fig, ax = plt.subplots(figsize=(max(6, len(words) * 0.8), max(5, len(words) * 0.7)))
    vmin = sim[~np.eye(len(words), dtype=bool)].min() - 0.01
    sns.heatmap(
        sim, annot=True, fmt=".3f", cmap="coolwarm",
        xticklabels=words, yticklabels=words,
        vmin=vmin, vmax=1.0, ax=ax,
        linewidths=0.5, linecolor="gray",
    )
    # Highlight original word row/col
    orig_idx = words.index(found_word) if found_word in words else 0
    ax.add_patch(plt.Rectangle((0, orig_idx), len(words), 1,
                                fill=False, edgecolor="lime", lw=3))
    ax.add_patch(plt.Rectangle((orig_idx, 0), 1, len(words),
                                fill=False, edgecolor="lime", lw=3))
    ax.set_title(
        f'Fused Embedding Cosine Similarity — Counting Token\n'
        f'base: "{base_prompt}"',
        fontsize=10,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    p = os.path.join(out_dir, "swap_embed_similarity.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {p}")

    # ── 3. Peak-layer bar chart ─────────────────────────────────────────────
    peaks  = [int(rw.argmax()) + 1 for rw in rws]
    colors = [cmap(i % 10) for i in range(len(words))]

    fig, ax = plt.subplots(figsize=(max(8, len(words) * 0.9), 4))
    bars = ax.bar(words, peaks, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(deep_layer, color="#2c3e50", linestyle="--", linewidth=1.2,
               label=f"default deep layer ({deep_layer})")
    for bar, peak in zip(bars, peaks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                str(peak), ha="center", va="bottom", fontsize=9)
    # Mark original word bar
    if found_word in words:
        bars[words.index(found_word)].set_edgecolor("red")
        bars[words.index(found_word)].set_linewidth(2.5)
    ax.set_xlabel("Counting Word", fontsize=12)
    ax.set_ylabel("Peak Routing Layer", fontsize=12)
    ax.set_ylim(0, num_layers + 1)
    ax.set_title(
        f'Peak Routing Layer per Counting Word\n'
        f'base: "{base_prompt}"  (red border = original)',
        fontsize=11,
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    p = os.path.join(out_dir, "swap_peak_layer.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {p}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_counting_by_word(all_token_weights, num_layers, save_path):
    """
    One curve per distinct counting word (one / two / three / …).
    Shows how routing preferences differ across different counting words.
    """
    from collections import defaultdict

    word_buckets = defaultdict(list)
    for word, w in all_token_weights:
        word_buckets[word].append(w)

    # Sort words numerically where possible, then alphabetically
    _order = ["one","1","two","2","three","3","four","4",
              "five","5","six","6","seven","7","eight","8","nine","9","ten","10"]
    words_sorted = sorted(
        word_buckets.keys(),
        key=lambda w: _order.index(w) if w in _order else 99
    )

    cmap = plt.get_cmap("tab10")
    x    = np.arange(1, num_layers)   # n_route = num_layers - 1

    fig, ax = plt.subplots(figsize=(14, 5))
    for idx, word in enumerate(words_sorted):
        rows = np.stack(word_buckets[word])   # [N, L]
        mean = rows.mean(axis=0)
        std  = rows.std(axis=0)
        color = cmap(idx % 10)
        ax.plot(x, mean, color=color, linewidth=2,
                label=f'"{word}"  (n={len(rows)})')
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12)

    deep_layer = num_layers - 1
    ax.axvline(deep_layer, color="#2c3e50", linestyle="--", linewidth=1.2,
               label=f"default deep layer ({deep_layer})")

    ax.set_xlabel("LLM Layer", fontsize=12)
    ax.set_ylabel("Average Routing Weight (softmax)", fontsize=12)
    ax.set_title("Routing Weight per Layer — Breakdown by Counting Word", fontsize=12)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.grid(axis="x", which="major", linestyle="--", alpha=0.35)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.legend(fontsize=9, ncol=2)
    ax.set_xlim(0.5, num_layers - 0.5)
    plt.close(fig)
    print(f"[Saved] {save_path}")


def plot_counting_avg(all_token_weights, num_layers, save_path):
    """
    Bar + error-bar chart: average softmax weight per LLM layer for all
    collected counting tokens.  Mean ± std shown as error bars.
    """
    if not all_token_weights:
        print("[Warning] No counting tokens found – nothing to plot.")
        return

    mat  = np.stack([w for _, w in all_token_weights], axis=0)   # [N, L]
    mean = mat.mean(axis=0)   # [L]
    std  = mat.std(axis=0)    # [L]
    x    = np.arange(1, num_layers)   # n_route = num_layers - 1

    fig, ax = plt.subplots(figsize=(14, 5))

    # Bar chart for mean
    ax.bar(x, mean, color="#e74c3c", alpha=0.75, width=0.8, label="mean weight")
    # Error bars for std
    ax.errorbar(x, mean, yerr=std, fmt="none", ecolor="#922b21",
                elinewidth=1.2, capsize=2, alpha=0.7)

    ax.set_xlabel("LLM Layer", fontsize=12)
    ax.set_ylabel("Average Routing Weight (softmax)", fontsize=12)
    ax.set_title(
        f"Counting Tokens — Average Router Weight per Layer\n"
        f"(n={len(all_token_weights)} tokens from {len(DEFAULT_PROMPTS)} prompts, "
        f"words: {sorted({t for t, _ in all_token_weights})})",
        fontsize=11,
    )

    # Mark default deep layer (hidden_states[-2] = layer num_layers - 1, 1-indexed)
    deep_layer = num_layers - 1
    ax.axvline(deep_layer, color="#2c3e50", linestyle="--", linewidth=1.5,
               label=f"default deep layer ({deep_layer})")

    # Annotate peak layer
    peak = int(mean.argmax()) + 1
    ax.annotate(
        f"peak: layer {peak}\n(w={mean[peak-1]:.3f})",
        xy=(peak, mean[peak - 1]),
        xytext=(peak + max(1, num_layers // 10), mean[peak - 1] * 0.85),
        arrowprops=dict(arrowstyle="->", color="#7f8c8d"),
        fontsize=10, color="#c0392b",
    )

    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
    ax.grid(axis="x", which="major", linestyle="--", alpha=0.35)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_xlim(0.5, num_layers - 0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Average routing weights for counting tokens")
    p.add_argument("--router_ckpt", type=str, required=True,
                   help="Path to trained router checkpoint (.pt)")
    p.add_argument("--model_dir",   type=str, default="ckpts/Z-Image-Turbo",
                   help="Z-Image model directory")
    p.add_argument("--out_dir",     type=str, default="outputs/router_vis",
                   help="Directory to save outputs")
    p.add_argument("--prompts",     type=str, nargs="+", default=None,
                   help="Custom prompts (default: built-in counting prompts)")
    p.add_argument("--max_length",  type=int, default=512,
                   help="Max tokenizer sequence length")
    # Swap mode
    p.add_argument("--swap", action="store_true",
                   help="Swap-analysis mode: replace the counting word with every other "
                        "number word and compare routing weights + fused embeddings")
    p.add_argument("--swap_prompt", type=str,
                   default="two cats sitting on a sofa",
                   help="Base prompt for swap analysis (must contain a counting word)")
    p.add_argument("--swap_targets", type=str, nargs="+", default=None,
                   help="Which counting words to swap in (default: one–ten)")
    p.add_argument("--diag", action="store_true",
                   help="Diagnostic mode: visualise the raw decision_feat similarity "
                        "across counting word swaps (runs alongside --swap automatically)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Device] {device}")

    # ---- Load text encoder + tokenizer ----
    print("[Init] Loading text encoder …")
    model_path   = ensure_model_weights(args.model_dir, verify=False)
    components   = load_from_local_dir(model_path, device=device, dtype=torch.bfloat16, verbose=True)
    text_encoder = components["text_encoder"]
    tokenizer    = components["tokenizer"]
    del components["transformer"], components["vae"], components["scheduler"]
    import gc; gc.collect()

    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad_(False)

    # ---- Load router ----
    print(f"[Init] Loading router from {args.router_ckpt} …")
    ckpt       = torch.load(args.router_ckpt, map_location="cpu", weights_only=False)
    num_layers = ckpt["num_layers"]
    mid_dim    = ckpt["mid_dim"]

    router = DynamicTokenRouter(
        hidden_size=ckpt["hidden_size"], num_layers=num_layers, mid_dim=mid_dim
    )
    state_dict = ckpt["router_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    router.load_state_dict(state_dict)
    router.to(device)
    router.eval()

    print(f"[Router] hidden_size={ckpt['hidden_size']}, num_layers={num_layers}, mid_dim={mid_dim}")
    if "epoch" in ckpt:
        print(f"[Router] epoch={ckpt['epoch']}, step={ckpt.get('step','?')}, "
              f"best_loss={ckpt.get('best_loss','N/A')}")

    # ================================================================
    # Mode A: swap analysis
    # ================================================================
    if args.swap:
        swap_out = os.path.join(args.out_dir, "swap")
        os.makedirs(swap_out, exist_ok=True)
        print(f"\n[Swap Analysis] base prompt: '{args.swap_prompt}'")
        swap_targets = args.swap_targets if args.swap_targets else SWAP_TARGETS

        variants, found_word = run_swap_analysis(
            args.swap_prompt, text_encoder, tokenizer, router,
            device, args.max_length, swap_targets,
        )
        plot_swap_results(variants, args.swap_prompt, found_word, num_layers, swap_out)

        # ---- Decision-feat diagnostic (always runs in swap mode) ----------
        print("\n[Diagnostic] Extracting raw decision_feat for each variant …")
        diag_out = os.path.join(swap_out, "decision_feat_diag")
        variants_df = []
        for target in (args.swap_targets or SWAP_TARGETS):
            prompt  = replace_counting_word(args.swap_prompt, found_word, target)
            results = get_decision_feat(
                prompt, text_encoder, tokenizer, router, device, args.max_length
            )
            if results:
                tok, df = results[0]
                variants_df.append({"word": target, "df": df})
                print(f"  '{target}': decision_feat shape={df.shape}")
        if variants_df:
            plot_decision_feat_similarity(variants_df, args.swap_prompt, diag_out)

        # Save raw data
        raw = [{"word": v["word"], "prompt": v["prompt"],
                "peak_layer": int(v["rw"].argmax()) + 1,
                "peak_weight": float(v["rw"].max()),
                "rw": v["rw"].tolist()} for v in variants]
        with open(os.path.join(swap_out, "swap_stats.json"), "w") as f:
            json.dump(raw, f, indent=2)
        print(f"[Saved] {os.path.join(swap_out, 'swap_stats.json')}")
        print(f"\n[Done] Swap outputs in: {swap_out}")
        return

    # ================================================================
    # Mode B: average weight across many prompts (default)
    # ================================================================
    prompts = args.prompts if args.prompts else DEFAULT_PROMPTS
    print(f"\n[Analysis] {len(prompts)} prompts")

    all_token_weights = []   # list of (word_str, np.ndarray[L])
    for idx, prompt in enumerate(prompts):
        found = get_counting_weights(
            prompt, text_encoder, tokenizer, router, device, args.max_length
        )
        for word, w in found:
            peak = int(w.argmax()) + 1
            print(f"  [{idx+1:>2}] '{prompt}'  →  '{word}'  peak=layer {peak}  (w={w.max():.3f})")
        all_token_weights.extend(found)

    print(f"\n[Summary] {len(all_token_weights)} counting tokens collected in total.")

    # ---- Plot ----
    plot_counting_avg(
        all_token_weights, num_layers,
        os.path.join(args.out_dir, "counting_avg_weight.png"),
    )

    print("[Plotting] Per-word breakdown …")
    plot_counting_by_word(
        all_token_weights, num_layers,
        os.path.join(args.out_dir, "counting_by_word.png"),
    )

    # ---- Save raw data ----
    stats = [
        {"token": t, "peak_layer": int(w.argmax()) + 1,
         "peak_weight": float(w.max()), "weights": w.tolist()}
        for t, w in all_token_weights
    ]
    stats_path = os.path.join(args.out_dir, "counting_weight_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[Saved] {stats_path}")

    # ---- Console summary ----
    if all_token_weights:
        mat  = np.stack([w for _, w in all_token_weights])
        mean = mat.mean(axis=0)
        peak = int(mean.argmax()) + 1
        print(f"\n  mean peak layer : {peak}")
        print(f"  mean weight at peak : {mean[peak-1]:.4f}")
        top5 = (mean.argsort()[::-1][:5] + 1).tolist()
        print(f"  top-5 layers by mean weight : {top5}")

    print(f"\n[Done] Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
