"""
diagnose_prompt.py
==================
Dedicated diagnostic for why Z-Image fails on a specific counting prompt
(default: "a photo of four computer keyboards").

What it does
------------
1. Token-level layer scan
   For every LLM layer, extract the feature of the counting token ("four") and
   a reference noun token ("keyboards").  Report their cosine similarity and
   feature norm layer-by-layer so we can see where "four" becomes
   indistinguishable from other numerals.

2. Numeral confusion matrix (per layer)
   For ONE specific layer (--focus_layer, default=1) plot a cosine-similarity
   heatmap of "four" vs ["one","two","three","four","five","six","seven","eight",
   "nine","ten"] obtained from minimal prompts like "a photo of X keyboard".
   Shows whether the LLM actually separates counting words at each layer.

3. Decay weight distribution
   For each tested decay_rate, show the per-layer weight curve and highlight
   where "four" token lands relative to the weight mass.

4. Fused-embedding numeral confusion (per decay setting)
   Same confusion matrix as (2) but after applying the decay fusion, for
   multiple (route_start, route_end, decay_rate) settings.  Directly shows
   whether and how well the fusion helps separate numerals.

5. Baseline vs. fused feature norm ratio
   If the fused "four" embedding is very close to deep baseline, the fusion
   is having no practical effect.  We plot the L2 distance and cos-sim between
   baseline and each fused version to quantify this.

All figures are saved under --out_dir (default: diagnose_results/).
"""

import argparse
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils import ensure_model_weights, load_from_local_dir


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

COUNTING_WORDS = ["one", "two", "three", "four", "five",
                  "six", "seven", "eight", "nine", "ten"]

MINIMAL_TEMPLATE = "a photo of {num} computer keyboards"


def _encode(text, text_encoder, tokenizer, device, max_seq_len=512):
    """Return (hidden_states_tuple, input_ids, attention_mask, tokens_list)."""
    messages = [{"role": "user", "content": text}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    enc = tokenizer(
        [formatted], padding="max_length", max_length=max_seq_len,
        truncation=True, return_tensors="pt",
    )
    input_ids      = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device).bool()

    with torch.no_grad():
        out = text_encoder(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True,
        )
    hs = out.hidden_states   # tuple: len = num_layers + 1

    valid_ids = input_ids[0][attention_mask[0]]
    tokens    = [tokenizer.decode([tid]) for tid in valid_ids.tolist()]
    return hs, input_ids, attention_mask, tokens


def _find_content_span(tokens):
    """Return (content_start, content_end) indices inside the chat template."""
    cs, ce = 0, len(tokens)
    for i, t in enumerate(tokens):
        if "user" in t.lower():
            cs = i + 1
        elif "<|im_end|>" in t and i > cs:
            ce = i
            break
    return cs, ce


def _find_token_idx(tokens, word):
    """Find first token index (in full token list) matching `word`."""
    for i, t in enumerate(tokens):
        clean = t.lower().strip().replace(" ", "")
        if clean == word.lower() or word.lower() in clean:
            return i
    return -1


def _cosine(a, b):
    a = a / (np.linalg.norm(a) + 1e-8)
    b = b / (np.linalg.norm(b) + 1e-8)
    return float(a @ b)


# ──────────────────────────────────────────────────────────────────────────────
# Part 1 – Token-level layer scan for the target prompt
# ──────────────────────────────────────────────────────────────────────────────

def part1_layer_scan(prompt, count_word, noun_word, text_encoder, tokenizer,
                     device, out_dir):
    print("\n[Part 1] Layer-scan for target prompt …")
    hs, input_ids, attn_mask, tokens = _encode(
        prompt, text_encoder, tokenizer, device)
    num_layers = len(hs) - 1   # hs[0]=embedding, hs[1..num_layers]=transformer

    cs, ce = _find_content_span(tokens)
    content = tokens[cs:ce]

    cidx_count = _find_token_idx(content, count_word)
    cidx_noun  = _find_token_idx(content, noun_word)
    if cidx_count == -1:
        print(f"  [warn] '{count_word}' not found in tokens: {content}")
        return
    if cidx_noun == -1:
        print(f"  [warn] '{noun_word}' not found in tokens: {content}")
        cidx_noun = None

    full_count = cs + cidx_count
    full_noun  = cs + cidx_noun if cidx_noun is not None else None
    print(f"  '{count_word}' → token idx {full_count} ('{tokens[full_count].strip()}')")
    if full_noun:
        print(f"  '{noun_word}'  → token idx {full_noun} ('{tokens[full_noun].strip()}')")

    norms_count, norms_noun = [], []
    sim_count_noun = []
    for l in range(1, num_layers + 1):
        feat_c = hs[l][0, full_count, :].float().cpu().numpy()
        norms_count.append(np.linalg.norm(feat_c))
        if full_noun is not None:
            feat_n = hs[l][0, full_noun, :].float().cpu().numpy()
            norms_noun.append(np.linalg.norm(feat_n))
            sim_count_noun.append(_cosine(feat_c, feat_n))

    layers = np.arange(1, num_layers + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(layers, norms_count, label=f'"{count_word}" norm', linewidth=2, color="#e74c3c")
    if norms_noun:
        axes[0].plot(layers, norms_noun, label=f'"{noun_word}" norm', linewidth=2,
                     color="#3498db", linestyle="--")
    axes[0].set_xlabel("LLM Layer")
    axes[0].set_ylabel("Feature L2 Norm")
    axes[0].set_title(f"Feature Norm across Layers\nprompt: \"{prompt}\"")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if sim_count_noun:
        axes[1].plot(layers, sim_count_noun, color="#9b59b6", linewidth=2)
        axes[1].axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        axes[1].set_xlabel("LLM Layer")
        axes[1].set_ylabel(f'Cosine Sim("{count_word}", "{noun_word}")')
        axes[1].set_title(f'Cosine Similarity between "{count_word}" and "{noun_word}"\nacross layers')
        axes[1].set_ylim(0.0, 1.05)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].set_visible(False)

    plt.tight_layout()
    p = os.path.join(out_dir, "part1_layer_scan.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {p}")


# ──────────────────────────────────────────────────────────────────────────────
# Part 2 – Per-layer numeral confusion matrix
# ──────────────────────────────────────────────────────────────────────────────

def part2_numeral_confusion_per_layer(count_word, text_encoder, tokenizer,
                                      device, out_dir, layers_to_plot=None):
    """
    Build prompts "a photo of X computer keyboards" for all counting words.
    For each layer in `layers_to_plot`, plot a cosine-sim heatmap.
    """
    print("\n[Part 2] Numeral confusion matrix per layer …")
    if layers_to_plot is None:
        layers_to_plot = [1, 5, 10, 15, 20, 25, 30, 35]

    # Collect hidden states for each numeral
    all_hs = {}   # word → hs tuple
    all_idx = {}  # word → token index of numeral in full seq

    for w in COUNTING_WORDS:
        prompt = MINIMAL_TEMPLATE.format(num=w)
        hs, input_ids, attn_mask, tokens = _encode(
            prompt, text_encoder, tokenizer, device)
        cs, ce = _find_content_span(tokens)
        tidx = _find_token_idx(tokens[cs:ce], w)
        if tidx == -1:
            print(f"  [skip] '{w}' not found in tokens {tokens[cs:ce]}")
            continue
        all_hs[w]  = hs
        all_idx[w] = cs + tidx

    words    = list(all_hs.keys())
    num_layers_total = len(next(iter(all_hs.values()))) - 1
    valid_layers = [l for l in layers_to_plot if 1 <= l <= num_layers_total]

    n_plots = len(valid_layers)
    ncols   = min(4, n_plots)
    nrows   = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 4.0))
    axes = np.array(axes).flatten()

    for ax_idx, layer in enumerate(valid_layers):
        feats = []
        for w in words:
            f = all_hs[w][layer][0, all_idx[w], :].float().cpu().numpy()
            feats.append(f)
        feats = np.stack(feats)
        norms = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        sim   = norms @ norms.T

        vmin = sim[~np.eye(len(words), dtype=bool)].min() - 0.01
        sns.heatmap(sim, annot=True, fmt=".3f", cmap="coolwarm",
                    xticklabels=words, yticklabels=words,
                    vmin=vmin, vmax=1.0, ax=axes[ax_idx],
                    linewidths=0.4, linecolor="gray", annot_kws={"size": 7})
        target_idx = words.index(count_word) if count_word in words else 0
        axes[ax_idx].add_patch(plt.Rectangle(
            (0, target_idx), len(words), 1, fill=False, edgecolor="lime", lw=2))
        axes[ax_idx].add_patch(plt.Rectangle(
            (target_idx, 0), 1, len(words), fill=False, edgecolor="lime", lw=2))
        axes[ax_idx].set_title(f"Layer {layer}", fontsize=10)
        axes[ax_idx].tick_params(axis='x', rotation=45)

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Numeral cosine-similarity (counting token) across layers\n"
        f"template: \"{MINIMAL_TEMPLATE.format(num='X')}\"  |  target: \"{count_word}\"",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    p = os.path.join(out_dir, "part2_numeral_confusion_by_layer.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {p}")


# ──────────────────────────────────────────────────────────────────────────────
# Part 3 – Decay weight distribution for multiple settings
# ──────────────────────────────────────────────────────────────────────────────

def part3_decay_weight_curves(decay_configs, out_dir):
    """
    decay_configs: list of (route_start, route_end, decay_rate)
    Just plots the weight distribution (no model needed).
    """
    print("\n[Part 3] Decay weight distributions …")
    fig, ax = plt.subplots(figsize=(12, 5))
    cmap = plt.get_cmap("tab10")

    for i, (rs, re_, dr) in enumerate(decay_configs):
        if dr is None:
            # Hard-replace: all weight on rs
            x = np.array([rs])
            w = np.array([1.0])
            label = f"hard-replace layer {rs}"
        else:
            n = re_ - rs
            w = np.exp(-dr * np.arange(n))
            w = w / w.sum()
            x = np.arange(rs, re_)
            label = f"L[{rs},{re_}) rate={dr}  (peak→layer {rs}, tail→{re_-1})"
        ax.plot(x, w, marker="o", markersize=4, linewidth=1.8,
                color=cmap(i % 10), label=label)

    ax.set_xlabel("LLM Layer Index")
    ax.set_ylabel("Normalised Weight")
    ax.set_title("Decay weight distribution for tested configurations")
    ax.legend(fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "part3_decay_weight_curves.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {p}")


# ──────────────────────────────────────────────────────────────────────────────
# Part 4 – Fused-embedding numeral confusion per decay config
# ──────────────────────────────────────────────────────────────────────────────

def _build_fused(hs, token_idx, route_start, route_end, decay_rate, device):
    """Compute decay-fused feature for one token.
    decay_rate=None means hard-replace with the shallowest layer in range.
    """
    total = len(hs)
    rs  = max(1, route_start)
    re_ = max(rs + 1, min(route_end, total - 1))
    if decay_rate is None:
        # Hard-replace: use only route_start layer
        return hs[rs][0, token_idx, :].float().cpu().numpy()
    layers = hs[rs:re_]
    n = len(layers)
    w = torch.exp(-decay_rate * torch.arange(n, dtype=torch.float32, device=device))
    w = w / w.sum()
    fused = torch.zeros_like(layers[0][0, token_idx, :].float())
    for wi, l in zip(w, layers):
        fused += wi * l[0, token_idx, :].float()
    return fused.cpu().numpy()


def part4_fused_confusion(count_word, text_encoder, tokenizer, device,
                           decay_configs, out_dir):
    print("\n[Part 4] Fused-embedding numeral confusion per decay config …")

    # Collect hidden states once per numeral
    all_hs  = {}
    all_idx = {}
    all_deep = {}  # baseline deep embedding
    for w in COUNTING_WORDS:
        prompt = MINIMAL_TEMPLATE.format(num=w)
        hs, input_ids, attn_mask, tokens = _encode(
            prompt, text_encoder, tokenizer, device)
        cs, ce = _find_content_span(tokens)
        tidx = _find_token_idx(tokens[cs:ce], w)
        if tidx == -1:
            continue
        all_hs[w]   = hs
        all_idx[w]  = cs + tidx
        all_deep[w] = hs[-2][0, cs + tidx, :].float().cpu().numpy()

    words = list(all_hs.keys())

    # One subplot per decay config + one for the deep baseline
    configs_with_baseline = [("baseline\n(layer -2)", None, None, None)] + \
                            [(f"hard-replace\nlayer {rs}" if dr is None
                              else f"L[{rs},{re_})\nrate={dr}", rs, re_, dr)
                             for rs, re_, dr in decay_configs]
    n_plots = len(configs_with_baseline)
    ncols   = min(3, n_plots)
    nrows   = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 5.0, nrows * 4.5))
    axes = np.array(axes).flatten()

    target_idx = words.index(count_word) if count_word in words else 0

    for ax_idx, cfg in enumerate(configs_with_baseline):
        label, rs, re_, dr = cfg
        if rs is None:
            feats = np.stack([all_deep[w] for w in words])
        else:
            feats = np.stack([
                _build_fused(all_hs[w], all_idx[w], rs, re_, dr, device)
                for w in words
            ])
        norms = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        sim   = norms @ norms.T
        vmin  = sim[~np.eye(len(words), dtype=bool)].min() - 0.01

        off_diag = sim[~np.eye(len(words), dtype=bool)]
        mean_sim  = off_diag.mean()

        sns.heatmap(sim, annot=True, fmt=".3f", cmap="coolwarm",
                    xticklabels=words, yticklabels=words,
                    vmin=vmin, vmax=1.0, ax=axes[ax_idx],
                    linewidths=0.4, linecolor="gray", annot_kws={"size": 7})
        axes[ax_idx].add_patch(plt.Rectangle(
            (0, target_idx), len(words), 1, fill=False, edgecolor="lime", lw=2))
        axes[ax_idx].add_patch(plt.Rectangle(
            (target_idx, 0), 1, len(words), fill=False, edgecolor="lime", lw=2))
        axes[ax_idx].set_title(
            f"{label}\noff-diag mean={mean_sim:.4f}", fontsize=9)
        axes[ax_idx].tick_params(axis='x', rotation=45)

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    fig.suptitle(
        f"Fused embedding numeral confusion — target: \"{count_word}\"\n"
        f"template: \"{MINIMAL_TEMPLATE.format(num='X')}\"",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    p = os.path.join(out_dir, "part4_fused_confusion.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {p}")


# ──────────────────────────────────────────────────────────────────────────────
# Part 5 – Baseline vs fused distance (quantify fusion effect)
# ──────────────────────────────────────────────────────────────────────────────

def part5_fusion_delta(count_word, text_encoder, tokenizer, device,
                       decay_configs, out_dir):
    """
    For each decay config, compute:
      - L2 distance between fused and deep-baseline "four" embedding
      - Cosine similarity between fused and deep-baseline
    across all counting words, to see if fusion actually moves the embeddings.
    """
    print("\n[Part 5] Fusion delta (fused vs. baseline) …")

    all_hs  = {}
    all_idx = {}
    all_deep = {}
    for w in COUNTING_WORDS:
        prompt = MINIMAL_TEMPLATE.format(num=w)
        hs, input_ids, attn_mask, tokens = _encode(
            prompt, text_encoder, tokenizer, device)
        cs, ce = _find_content_span(tokens)
        tidx = _find_token_idx(tokens[cs:ce], w)
        if tidx == -1:
            continue
        all_hs[w]   = hs
        all_idx[w]  = cs + tidx
        all_deep[w] = hs[-2][0, cs + tidx, :].float().cpu().numpy()

    words = list(all_hs.keys())

    fig, (ax_l2, ax_cos) = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.get_cmap("tab10")

    for i, (rs, re_, dr) in enumerate(decay_configs):
        l2_dists  = []
        cos_sims  = []
        for w in words:
            fused = _build_fused(all_hs[w], all_idx[w], rs, re_, dr, device)
            base  = all_deep[w]
            l2_dists.append(np.linalg.norm(fused - base))
            cos_sims.append(_cosine(fused, base))
        label = f"hard-replace L{rs}" if dr is None else f"L[{rs},{re_}) rate={dr}"
        x = np.arange(len(words))
        ax_l2.bar(x + i * 0.15, l2_dists, width=0.13,
                  color=cmap(i % 10), label=label, alpha=0.8)
        ax_cos.bar(x + i * 0.15, cos_sims, width=0.13,
                   color=cmap(i % 10), label=label, alpha=0.8)

    # Highlight target word
    if count_word in words:
        ti = words.index(count_word)
        for ax in (ax_l2, ax_cos):
            ax.axvspan(ti - 0.5, ti + 0.5 + len(decay_configs) * 0.15,
                       color="yellow", alpha=0.18, label=f'"{count_word}"')

    ax_l2.set_xticks(np.arange(len(words)))
    ax_l2.set_xticklabels(words, rotation=30)
    ax_l2.set_ylabel("L2(fused, deep_baseline)")
    ax_l2.set_title("How much does fusion move each numeral?\n(larger = bigger shift from baseline)")
    ax_l2.legend(fontsize=8)
    ax_l2.grid(axis="y", alpha=0.3)

    ax_cos.set_xticks(np.arange(len(words)))
    ax_cos.set_xticklabels(words, rotation=30)
    ax_cos.set_ylabel("CosSim(fused, deep_baseline)")
    ax_cos.set_title("Direction similarity: fused vs. baseline\n(closer to 1 = fusion barely changed direction)")
    ax_cos.set_ylim(0.9, 1.01)
    ax_cos.legend(fontsize=8)
    ax_cos.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    p = os.path.join(out_dir, "part5_fusion_delta.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {p}")


# ──────────────────────────────────────────────────────────────────────────────
# Part 6 – Off-diagonal similarity summary table
# ──────────────────────────────────────────────────────────────────────────────

def part6_summary_table(count_word, text_encoder, tokenizer, device,
                         decay_configs, out_dir):
    """Print (and save) a table: config → mean/min off-diagonal cosine sim."""
    print("\n[Part 6] Summary table …")
    all_hs  = {}
    all_idx = {}
    all_deep = {}
    for w in COUNTING_WORDS:
        prompt = MINIMAL_TEMPLATE.format(num=w)
        hs, input_ids, attn_mask, tokens = _encode(
            prompt, text_encoder, tokenizer, device)
        cs, ce = _find_content_span(tokens)
        tidx = _find_token_idx(tokens[cs:ce], w)
        if tidx == -1:
            continue
        all_hs[w]   = hs
        all_idx[w]  = cs + tidx
        all_deep[w] = hs[-2][0, cs + tidx, :].float().cpu().numpy()

    words = list(all_hs.keys())

    rows = []

    def sim_stats(feats):
        norms = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        sim   = norms @ norms.T
        mask  = ~np.eye(len(words), dtype=bool)
        return sim[mask].mean(), sim[mask].min(), sim[mask].max()

    # Baseline
    feats = np.stack([all_deep[w] for w in words])
    mn, mi, mx = sim_stats(feats)
    rows.append(("deep_baseline (layer -2)", "-", "-", "-", f"{mn:.4f}", f"{mi:.4f}", f"{mx:.4f}"))

    # Each layer raw (no fusion)
    num_layers_total = len(next(iter(all_hs.values()))) - 1
    for l in [1, 5, 8, 10, 12, 15, 20, 25, 30, num_layers_total]:
        if l > num_layers_total:
            continue
        feats = np.stack([all_hs[w][l][0, all_idx[w], :].float().cpu().numpy() for w in words])
        mn, mi, mx = sim_stats(feats)
        rows.append((f"raw layer {l}", "-", "-", "-", f"{mn:.4f}", f"{mi:.4f}", f"{mx:.4f}"))

    # Each decay config
    for rs, re_, dr in decay_configs:
        feats = np.stack([
            _build_fused(all_hs[w], all_idx[w], rs, re_, dr, device)
            for w in words
        ])
        mn, mi, mx = sim_stats(feats)
        cfg_name = f"hard-replace layer {rs}" if dr is None else f"decay L[{rs},{re_}) rate={dr}"
        rows.append((cfg_name, rs, re_, str(dr), f"{mn:.4f}", f"{mi:.4f}", f"{mx:.4f}"))

    header = f"{'Config':<40} {'rs':>4} {'re':>4} {'dr':>6}  {'mean':>8} {'min':>8} {'max':>8}"
    lines  = [header, "-" * len(header)]
    for r in rows:
        lines.append(f"{r[0]:<40} {str(r[1]):>4} {str(r[2]):>4} {str(r[3]):>6}  {r[4]:>8} {r[5]:>8} {r[6]:>8}")
    table_str = "\n".join(lines)
    print(table_str)

    txt_path = os.path.join(out_dir, "part6_summary_table.txt")
    with open(txt_path, "w") as f:
        f.write(f"Numeral separation summary — target: '{count_word}'\n")
        f.write(f"template: \"{MINIMAL_TEMPLATE.format(num='X')}\"\n\n")
        f.write(table_str + "\n")
    print(f"  [saved] {txt_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Diagnose counting-prompt failure")
    p.add_argument("--prompt", type=str,
                   default="a photo of four computer keyboards",
                   help="The failing prompt to diagnose")
    p.add_argument("--count_word", type=str, default="four",
                   help="The counting word in the prompt")
    p.add_argument("--noun_word", type=str, default="keyboards",
                   help="The noun in the prompt (for layer-scan cos-sim)")
    p.add_argument("--out_dir", type=str, default="diagnose_results",
                   help="Directory to write all figures and tables")
    p.add_argument("--model_dir", type=str, default="ckpts/Z-Image-Turbo")
    p.add_argument("--max_seq_len", type=int, default=512)
    # Which layers to show in the per-layer confusion matrix (Part 2)
    p.add_argument("--scan_layers", type=int, nargs="+",
                   default=[1, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35],
                   help="Layers to include in Part-2 confusion grid")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading models …")
    model_path = ensure_model_weights(args.model_dir, verify=False)
    components = load_from_local_dir(model_path, device=device, dtype=torch.bfloat16)
    text_encoder = components["text_encoder"]
    tokenizer    = components["tokenizer"]

    # Decay configurations to compare — sweep range and rate
    decay_configs = [
        # (route_start, route_end, decay_rate)
        ( 1, 36, 0.1),   # original: all layers, slow decay
        ( 1, 36, 0.3),   # all layers, faster decay
        (10, 21, 0.05),  # mid-range, nearly uniform
        (10, 21, 0.2),   # mid-range, moderate
        (10, 21, 0.5),   # mid-range, strong (biased toward layer 10)
        (10, 21, 1.0),   # very strong → almost hard-select layer 10
        (10, 21, 2.0),   # extreme → effectively layer 10 only
        ( 8, 20, 0.2),   # slightly wider mid-range
        ( 5, 15, 0.2),   # shallower mid-range
        (10, 11, None),  # hard-replace layer 10 (upper bound of any decay)
    ]

    # ── Run all parts ──────────────────────────────────────────────────────
    part1_layer_scan(
        args.prompt, args.count_word, args.noun_word,
        text_encoder, tokenizer, device, args.out_dir,
    )
    part2_numeral_confusion_per_layer(
        args.count_word, text_encoder, tokenizer, device,
        args.out_dir, layers_to_plot=args.scan_layers,
    )
    part3_decay_weight_curves(decay_configs, args.out_dir)

    part4_fused_confusion(
        args.count_word, text_encoder, tokenizer, device,
        decay_configs, args.out_dir,
    )
    part5_fusion_delta(
        args.count_word, text_encoder, tokenizer, device,
        decay_configs, args.out_dir,
    )
    part6_summary_table(
        args.count_word, text_encoder, tokenizer, device,
        decay_configs, args.out_dir,
    )

    print(f"\n✓ All done.  Results → {args.out_dir}/")
    print("Key files:")
    print("  part1_layer_scan.png          — norm & cos-sim across layers")
    print("  part2_numeral_confusion_by_layer.png — raw confusion per layer")
    print("  part3_decay_weight_curves.png — weight mass per config")
    print("  part4_fused_confusion.png     — confusion after fusion (vs baseline)")
    print("  part5_fusion_delta.png        — how much fusion moves each numeral")
    print("  part6_summary_table.txt       — off-diag similarity summary")


if __name__ == "__main__":
    main()
