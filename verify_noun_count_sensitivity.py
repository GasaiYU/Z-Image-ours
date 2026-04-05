"""
verify_noun_count_sensitivity.py
=================================
Verify hypothesis:
  "Easy-to-count nouns (cups, dogs...) absorb the counting signal from the
   count word more effectively than hard nouns (keyboards, monitors...)."

Method
------
For each noun, encode prompts "a photo of {count} {noun}" for all counting
words.  Extract the noun token's hidden_states[-2] embedding for each count.

If the noun strongly absorbs the count signal:
  → its embedding should CHANGE with the count word
  → pairwise cosine similarity across counts will be LOW

If the noun ignores the count signal:
  → its embedding barely changes with the count word
  → pairwise cosine similarity across counts will be HIGH (≈ baseline similarity)

Two metrics are reported per noun:
  1. mean off-diagonal cosine similarity of noun embeddings across counts
     (lower = more count-sensitive)
  2. std of noun embeddings across counts
     (higher = more count-sensitive)

We also plot per-noun sensitivity and a heatmap per noun for visual inspection.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from utils import ensure_model_weights, load_from_local_dir


COUNT_WORDS = ["one", "two", "three", "four", "five",
               "six", "seven", "eight", "nine", "ten"]

# Easy nouns (common in counting contexts in training data)
EASY_NOUNS = ["cups", "dogs", "cats", "birds", "balls",
              "apples", "flowers", "chairs", "cars", "books"]

# Hard nouns (rare in counting contexts)
HARD_NOUNS = ["keyboards", "monitors", "briefcases", "microscopes",
              "trophies", "saxophones", "telescopes", "hammers"]


# ── helpers ──────────────────────────────────────────────────────────────────

def _encode(text, text_encoder, tokenizer, device, max_seq_len=512):
    messages  = [{"role": "user", "content": text}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    enc  = tokenizer([formatted], padding="max_length", max_length=max_seq_len,
                     truncation=True, return_tensors="pt")
    ids  = enc.input_ids.to(device)
    mask = enc.attention_mask.to(device).bool()
    with torch.no_grad():
        out = text_encoder(input_ids=ids, attention_mask=mask,
                           output_hidden_states=True)
    valid  = ids[0][mask[0]]
    tokens = [tokenizer.decode([t]) for t in valid.tolist()]
    return out.hidden_states, tokens, ids, mask


def _content_span(tokens):
    cs, ce = 0, len(tokens)
    for i, t in enumerate(tokens):
        if "user" in t.lower():
            cs = i + 1
        elif "<|im_end|>" in t and i > cs:
            ce = i; break
    return cs, ce


def _find_last_token_idx(tokens, word):
    """Find the LAST token in the content that matches `word` (handles multi-token words)."""
    for i in range(len(tokens) - 1, -1, -1):
        clean = tokens[i].lower().strip().replace(" ", "").replace("▁", "")
        if word.lower() in clean or clean in word.lower():
            return i
    return -1


def get_noun_embeddings(noun, count_words, text_encoder, tokenizer, device,
                        layer=-2, max_seq_len=512):
    """
    For each count word, encode "a photo of {count} {noun}" and
    extract the noun token embedding at `layer`.

    Returns: dict {count_word: np.array [D]}
    """
    template = "a photo of {} {}"
    embeddings = {}

    for count in count_words:
        prompt = template.format(count, noun)
        hs, tokens, ids, mask = _encode(prompt, text_encoder, tokenizer,
                                        device, max_seq_len)
        cs, ce = _content_span(tokens)
        content = tokens[cs:ce]

        # Find last token of the noun (handle multi-token nouns like "keyboards")
        # Strategy: find which content tokens correspond to the noun word
        noun_tidx = -1
        for i in range(len(content) - 1, -1, -1):
            clean = content[i].lower().strip().replace(" ", "").replace("▁", "")
            if clean and clean in noun.lower():
                noun_tidx = i
                break

        if noun_tidx == -1:
            # Fallback: take the last content token before padding
            noun_tidx = len(content) - 1

        full_idx = cs + noun_tidx
        feat = hs[layer][0, full_idx, :].float().cpu().numpy()
        embeddings[count] = feat

    return embeddings


def count_sensitivity(embeddings):
    """
    Given {count: embedding}, compute:
      - mean off-diagonal cosine similarity (lower = more sensitive to count)
      - std of embeddings (higher = more sensitive)
    """
    words = list(embeddings.keys())
    feats = np.stack([embeddings[w] for w in words])
    norms = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
    sim   = norms @ norms.T
    mask  = ~np.eye(len(words), dtype=bool)
    mean_sim = sim[mask].mean()
    std_feat = feats.std(axis=0).mean()   # mean per-dim std
    return mean_sim, std_feat, sim, words


# ── main analysis ─────────────────────────────────────────────────────────────

def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading models …")
    model_path = ensure_model_weights(args.model_dir, verify=False)
    components = load_from_local_dir(model_path, device=device, dtype=torch.bfloat16)
    text_encoder = components["text_encoder"]
    tokenizer    = components["tokenizer"]

    all_nouns   = EASY_NOUNS + HARD_NOUNS
    results     = {}   # noun → (mean_sim, std_feat)
    sim_matrices = {}  # noun → sim matrix

    print(f"\nAnalysing {len(all_nouns)} nouns × {len(COUNT_WORDS)} counts …")
    for noun in all_nouns:
        print(f"  {noun} …", end=" ", flush=True)
        embeddings = get_noun_embeddings(
            noun, COUNT_WORDS, text_encoder, tokenizer, device,
            layer=args.layer, max_seq_len=args.max_seq_len)
        mean_sim, std_feat, sim, words = count_sensitivity(embeddings)
        results[noun]      = (mean_sim, std_feat)
        sim_matrices[noun] = (sim, words)
        tag = "EASY" if noun in EASY_NOUNS else "HARD"
        print(f"mean_sim={mean_sim:.4f}  std={std_feat:.4f}  [{tag}]")

    # ── Plot 1: sensitivity bar chart ────────────────────────────────────────
    nouns_sorted = sorted(results.keys(), key=lambda n: results[n][0])
    mean_sims    = [results[n][0] for n in nouns_sorted]
    colors       = ["#3498db" if n in EASY_NOUNS else "#e74c3c"
                    for n in nouns_sorted]

    fig, ax = plt.subplots(figsize=(max(12, len(all_nouns) * 0.7), 5))
    bars = ax.bar(range(len(nouns_sorted)), mean_sims, color=colors, alpha=0.85)
    ax.set_xticks(range(len(nouns_sorted)))
    ax.set_xticklabels(nouns_sorted, rotation=40, ha="right", fontsize=10)
    ax.set_ylabel("Mean off-diagonal cosine similarity across count words\n"
                  "(LOWER = noun embedding changes more with count = absorbs count signal better)")
    ax.set_title(f"Noun Count-Sensitivity  (layer {args.layer})\n"
                 f"Blue=easy nouns  Red=hard nouns\n"
                 f"template: 'a photo of {{count}} {{noun}}'")
    ax.axhline(np.mean(mean_sims), color="gray", linestyle="--", linewidth=1,
               label=f"mean={np.mean(mean_sims):.4f}")

    # Group labels
    easy_mean = np.mean([results[n][0] for n in EASY_NOUNS if n in results])
    hard_mean = np.mean([results[n][0] for n in HARD_NOUNS if n in results])
    ax.text(0.02, 0.95,
            f"Easy nouns avg: {easy_mean:.4f}\nHard nouns avg: {hard_mean:.4f}\nDiff: {hard_mean - easy_mean:+.4f}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(args.out_dir, "noun_count_sensitivity_bar.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[saved] {p}")

    # ── Plot 2: per-noun cosine similarity heatmaps (grid) ──────────────────
    n_plots = len(all_nouns)
    ncols   = 5
    nrows   = (n_plots + ncols - 1) // ncols
    fig2, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.5, nrows * 3.2))
    axes = np.array(axes).flatten()

    for idx, noun in enumerate(nouns_sorted):
        sim, words = sim_matrices[noun]
        vmin = sim[~np.eye(len(words), dtype=bool)].min() - 0.01
        tag  = "EASY" if noun in EASY_NOUNS else "HARD"
        mean_s = results[noun][0]
        sns.heatmap(sim, annot=True, fmt=".3f", cmap="coolwarm",
                    xticklabels=words, yticklabels=words,
                    vmin=vmin, vmax=1.0, ax=axes[idx],
                    linewidths=0.3, linecolor="gray",
                    annot_kws={"size": 5})
        axes[idx].set_title(f"{noun} [{tag}]\nmean={mean_s:.4f}",
                            fontsize=8,
                            color="#1a5276" if tag == "EASY" else "#922b21")
        axes[idx].tick_params(labelsize=5)

    for ax in axes[n_plots:]:
        ax.set_visible(False)

    fig2.suptitle(
        f"Per-noun cosine similarity of noun-token embeddings across count words\n"
        f"(each cell = cos-sim between 'a photo of ROW {'{noun}'}' and 'a photo of COL {'{noun}'}')\n"
        f"Lower off-diagonal = noun better distinguishes different counts",
        fontsize=9, y=1.01)
    plt.tight_layout()
    p2 = os.path.join(args.out_dir, "noun_count_sensitivity_heatmaps.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[saved] {p2}")

    # ── Plot 3: easy vs hard comparison (just cups vs keyboards) ─────────────
    spotlight = {n: sim_matrices[n] for n in ["cups", "keyboards"]
                 if n in sim_matrices}
    if len(spotlight) == 2:
        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
        for ax, noun in zip(axes3, ["cups", "keyboards"]):
            sim, words = spotlight[noun]
            vmin = sim[~np.eye(len(words), dtype=bool)].min() - 0.01
            sns.heatmap(sim, annot=True, fmt=".3f", cmap="coolwarm",
                        xticklabels=words, yticklabels=words,
                        vmin=vmin, vmax=1.0, ax=ax,
                        linewidths=0.5, linecolor="gray",
                        annot_kws={"size": 8})
            tag = "EASY" if noun in EASY_NOUNS else "HARD"
            ax.set_title(
                f'"{noun}"  [{tag}]\nmean off-diag sim = {results[noun][0]:.4f}',
                fontsize=11)
            ax.tick_params(labelsize=8)
        fig3.suptitle(
            "Spotlight: cups (easy) vs keyboards (hard)\n"
            "Noun token embedding cos-similarity across count words\n"
            "template: 'a photo of {count} {noun}'",
            fontsize=11)
        plt.tight_layout()
        p3 = os.path.join(args.out_dir, "spotlight_cups_vs_keyboards.png")
        plt.savefig(p3, dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"[saved] {p3}")

    # ── Text summary ─────────────────────────────────────────────────────────
    lines = [
        "Noun Count-Sensitivity Analysis",
        "================================",
        f"Layer: {args.layer}",
        f"Template: 'a photo of {{count}} {{noun}}'",
        f"Count words: {COUNT_WORDS}",
        "",
        f"{'Noun':<20} {'Category':<10} {'Mean Sim':>10} {'Std':>10}",
        "-" * 55,
    ]
    for noun in nouns_sorted:
        ms, sf = results[noun]
        tag = "EASY" if noun in EASY_NOUNS else "HARD"
        lines.append(f"{noun:<20} {tag:<10} {ms:>10.4f} {sf:>10.6f}")

    lines += [
        "",
        f"Easy nouns avg mean_sim : {easy_mean:.4f}",
        f"Hard nouns avg mean_sim : {hard_mean:.4f}",
        f"Difference (hard-easy)  : {hard_mean - easy_mean:+.4f}",
        "",
        "Interpretation:",
        "  If hard_mean >> easy_mean: hypothesis CONFIRMED",
        "    Hard nouns don't absorb counting signal → their embeddings are",
        "    nearly identical regardless of count word → DiT can't distinguish",
        "  If hard_mean ≈ easy_mean: hypothesis REJECTED",
        "    The problem is NOT in text encoding but in DiT visual knowledge",
    ]

    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    print(f"\n[saved] {os.path.join(args.out_dir, 'summary.txt')}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir",    default="noun_count_sensitivity")
    p.add_argument("--model_dir",  default="ckpts/Z-Image-Turbo")
    p.add_argument("--layer",      type=int, default=-2,
                   help="Which hidden layer to extract noun embedding from (-2=default deep, 10=most discriminative)")
    p.add_argument("--max_seq_len", type=int, default=512)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
