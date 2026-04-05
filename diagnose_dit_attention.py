"""
diagnose_dit_attention.py
=========================
Diagnose WHY counting word modifications to text embeddings fail to influence
the DiT output, for a specific failing prompt.

Three hypotheses are tested:

H1 – The DiT cross-attention barely attends to the counting token.
     → Register forward hooks on every DiT transformer block to capture
       cross-attention weights, run one denoising step, then visualise
       per-token attention averaged across heads and layers.

H2 – Layer-10 feature has abnormal L2 norm → DiT suppresses it.
     → Compare norm of baseline vs. layer-10 feature; also try
       "norm-preserved replacement": replace direction but keep baseline norm.

H3 – Direction shift is too small; need extrapolation beyond layer-10.
     → Try fused = baseline + scale * (layer10 - baseline) with scale > 1.
       If scale=2 or 3 visually changes counting, H3 is confirmed.

Outputs (all under --out_dir):
  h1_dit_cross_attention.png      per-token attention heatmap
  h1_token_attention_bar.png      bar chart: how much each token is attended to
  h2_norm_comparison.png          norm of each numeral at each layer
  h2_blend_directions.png         cosine-sim confusion for norm-preserved blends
  h3_extrapolation_confusion.png  confusion for extrapolated embeddings
  summary.txt                     text summary of all three findings
"""

import argparse
import os
import sys
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils import ensure_model_weights, load_from_local_dir


COUNTING_WORDS = ["one", "two", "three", "four", "five",
                  "six", "seven", "eight", "nine", "ten"]
MINIMAL_TEMPLATE = "a photo of {num} computer keyboards"


# ──────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────────────────────

def _encode(text, text_encoder, tokenizer, device, max_seq_len=512):
    messages  = [{"role": "user", "content": text}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    enc = tokenizer([formatted], padding="max_length", max_length=max_seq_len,
                    truncation=True, return_tensors="pt")
    ids  = enc.input_ids.to(device)
    mask = enc.attention_mask.to(device).bool()
    with torch.no_grad():
        out = text_encoder(input_ids=ids, attention_mask=mask,
                           output_hidden_states=True)
    hs     = out.hidden_states
    valid  = ids[0][mask[0]]
    tokens = [tokenizer.decode([t]) for t in valid.tolist()]
    return hs, ids, mask, tokens


def _content_span(tokens):
    cs, ce = 0, len(tokens)
    for i, t in enumerate(tokens):
        if "user" in t.lower():
            cs = i + 1
        elif "<|im_end|>" in t and i > cs:
            ce = i; break
    return cs, ce


def _find_idx(tokens, word):
    for i, t in enumerate(tokens):
        clean = t.lower().strip().replace(" ", "")
        if clean == word.lower() or word.lower() in clean:
            return i
    return -1


def _cosine_matrix(feats):
    n = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8
    f = feats / n
    return f @ f.T


def _off_diag_mean(sim):
    mask = ~np.eye(sim.shape[0], dtype=bool)
    return sim[mask].mean()


# ──────────────────────────────────────────────────────────────────────────────
# H1 – DiT cross-attention analysis
# ──────────────────────────────────────────────────────────────────────────────

def h1_dit_attention(prompt, count_word, components, device, out_dir,
                     max_seq_len=512, num_steps=1, seed=42):
    """
    Run one denoising step and capture cross-attention weights from the DiT.
    Visualise which text tokens the DiT actually attends to.
    """
    print("\n[H1] Capturing DiT cross-attention …")

    text_encoder = components["text_encoder"]
    tokenizer    = components["tokenizer"]
    transformer  = components["transformer"]
    vae          = components["vae"]
    scheduler    = components["scheduler"]

    # ── Encode text ──────────────────────────────────────────────────────────
    hs, ids, mask, tokens = _encode(prompt, text_encoder, tokenizer, device, max_seq_len)
    deep_embeds = hs[-2].to(dtype=next(transformer.parameters()).dtype)

    # ── Hook setup ───────────────────────────────────────────────────────────
    attn_maps = []   # list of [heads, img_tokens, text_tokens]

    def _make_hook(name):
        def hook(module, input, output):
            # Most DiT implementations return (out, attn_weight) or just out.
            # We try to capture the attention weight if present.
            if isinstance(output, tuple) and len(output) >= 2:
                w = output[1]
                if w is not None and w.dim() == 4:
                    # [B, heads, img_seq, txt_seq]
                    attn_maps.append(w[0].detach().float().cpu())
        return hook

    hooks = []
    for name, module in transformer.named_modules():
        # look for cross-attention layers (name contains "cross" or module has
        # both "to_k" and "to_q" child modules and is not a self-attention)
        if "cross_attn" in name.lower() or "crossattn" in name.lower():
            if hasattr(module, "forward"):
                hooks.append(module.register_forward_hook(_make_hook(name)))

    if not hooks:
        # Fallback: hook every Attention module and filter by text-conditioning
        for name, module in transformer.named_modules():
            cls_name = type(module).__name__.lower()
            if "attention" in cls_name and hasattr(module, "to_k"):
                hooks.append(module.register_forward_hook(_make_hook(name)))

    # ── Single denoising step ─────────────────────────────────────────────────
    torch.manual_seed(seed)
    latent_size = 128  # 1024 / 8
    latents = torch.randn(1, 16, latent_size, latent_size,
                          device=device,
                          dtype=next(transformer.parameters()).dtype)

    # Prepare text conditioning (monkey-patch to inject our embedding)
    original_fwd = text_encoder.forward
    _captured_embed = {"v": deep_embeds, "mask": mask}

    def patched_fwd(*args, **kwargs):
        class O:
            pass
        o = O()
        dummy_hs = list(hs)
        dummy_hs[-2] = _captured_embed["v"]
        o.hidden_states = tuple(dummy_hs)
        return o

    text_encoder.forward = patched_fwd
    try:
        scheduler.set_timesteps(max(num_steps, 20))
        t = scheduler.timesteps[:1]
        with torch.no_grad():
            _ = transformer(
                hidden_states=latents,
                timestep=t,
                encoder_hidden_states=deep_embeds,
                encoder_attention_mask=mask.to(deep_embeds.dtype),
                return_dict=False,
            )
    except Exception as e:
        print(f"  [warn] DiT forward failed: {e}")
        print("  Trying alternate call signature …")
        try:
            with torch.no_grad():
                _ = transformer(
                    hidden_states=latents,
                    timestep=t.float(),
                    encoder_hidden_states=deep_embeds,
                    return_dict=False,
                )
        except Exception as e2:
            print(f"  [warn] Second attempt also failed: {e2}")
    finally:
        text_encoder.forward = original_fwd
        for h in hooks:
            h.remove()

    if not attn_maps:
        print("  [warn] No cross-attention maps captured. "
              "The DiT may not expose attention weights. Skipping H1 plot.")
        with open(os.path.join(out_dir, "h1_note.txt"), "w") as f:
            f.write("Cross-attention maps not accessible via hooks in this DiT implementation.\n"
                    "H1 cannot be directly visualised.\n")
        return

    # ── Aggregate maps ────────────────────────────────────────────────────────
    # Stack: [num_layers, heads, img_tokens, txt_tokens]
    try:
        stacked = torch.stack(attn_maps, dim=0)   # [L, H, I, T]
    except RuntimeError:
        # different shapes across layers – average each separately
        txt_len = attn_maps[0].shape[-1]
        stacked = torch.stack([a for a in attn_maps if a.shape[-1] == txt_len])

    # Mean over layers and heads → [img_tokens, txt_tokens]
    mean_attn = stacked.mean(dim=(0, 1)).numpy()   # [I, T]
    # Mean over image tokens → [txt_tokens]
    per_token_attn = mean_attn.mean(axis=0)        # [T]

    # Decode valid text tokens
    valid_tokens = tokens  # already valid only

    # Align txt dim
    txt_len = min(per_token_attn.shape[0], len(valid_tokens))
    per_token_attn = per_token_attn[:txt_len]
    valid_tokens   = valid_tokens[:txt_len]
    token_labels   = [t.replace("▁", "").strip() or f"[{i}]"
                      for i, t in enumerate(valid_tokens)]

    # Bar chart: per-token attention
    cs, ce = _content_span(valid_tokens)
    content_labels = token_labels[cs:ce]
    content_attn   = per_token_attn[cs:ce]

    count_tidx = _find_idx(valid_tokens[cs:ce], count_word)

    fig, ax = plt.subplots(figsize=(max(10, len(content_labels) * 0.45), 5))
    colors = ["#e74c3c" if i == count_tidx else "#3498db"
              for i in range(len(content_labels))]
    ax.bar(range(len(content_labels)), content_attn, color=colors)
    ax.set_xticks(range(len(content_labels)))
    ax.set_xticklabels(content_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Cross-Attention Weight")
    ax.set_title(
        f"DiT Cross-Attention per Token (averaged over layers & heads)\n"
        f"prompt: \"{prompt}\"  |  red = \"{count_word}\"")
    ax.grid(axis="y", alpha=0.3)
    if count_tidx >= 0:
        ax.annotate(f"↑ {count_word}\n({content_attn[count_tidx]:.4f})",
                    xy=(count_tidx, content_attn[count_tidx]),
                    xytext=(count_tidx + 1, content_attn[count_tidx] * 1.1),
                    arrowprops=dict(arrowstyle="->"),
                    fontsize=9, color="#c0392b")
    plt.tight_layout()
    p = os.path.join(out_dir, "h1_token_attention_bar.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {p}")

    # Summary print
    if count_tidx >= 0:
        rank = sorted(range(len(content_attn)),
                      key=lambda i: content_attn[i], reverse=True).index(count_tidx) + 1
        total_mass = content_attn[count_tidx] / content_attn.sum()
        print(f"  '{count_word}' attention: {content_attn[count_tidx]:.4f}  "
              f"(rank {rank}/{len(content_attn)},  "
              f"{total_mass*100:.1f}% of total content attention)")
    return per_token_attn, token_labels


# ──────────────────────────────────────────────────────────────────────────────
# H2 – Norm analysis + norm-preserved replacement
# ──────────────────────────────────────────────────────────────────────────────

def h2_norm_analysis(count_word, text_encoder, tokenizer, device, out_dir,
                     focus_layers=(1, 5, 8, 10, 12, 15, 20, 25, 30, 35)):
    print("\n[H2] Norm analysis across layers …")

    all_hs  = {}
    all_idx = {}
    for w in COUNTING_WORDS:
        prompt = MINIMAL_TEMPLATE.format(num=w)
        hs, ids, mask, tokens = _encode(prompt, text_encoder, tokenizer, device)
        cs, ce = _content_span(tokens)
        tidx = _find_idx(tokens[cs:ce], w)
        if tidx == -1:
            continue
        all_hs[w]  = hs
        all_idx[w] = cs + tidx

    words      = list(all_hs.keys())
    num_layers = len(next(iter(all_hs.values()))) - 1
    valid_foci = [l for l in focus_layers if 1 <= l <= num_layers]

    # ── Plot norms per layer ──────────────────────────────────────────────────
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, w in enumerate(words):
        norms = [all_hs[w][l][0, all_idx[w], :].float().norm().item()
                 for l in range(1, num_layers + 1)]
        color = "#e74c3c" if w == count_word else cmap(i % 10)
        lw    = 2.5 if w == count_word else 1.0
        ax.plot(range(1, num_layers + 1), norms, color=color,
                linewidth=lw, label=f'"{w}"', alpha=0.85)

    ax.set_xlabel("LLM Layer")
    ax.set_ylabel("Feature L2 Norm")
    ax.set_title(f"Feature Norm per Layer for each Counting Word\n"
                 f"template: \"{MINIMAL_TEMPLATE.format(num='X')}\"")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "h2_norm_by_layer.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {p}")

    # ── Norm-preserved confusion at each focus layer ──────────────────────────
    # Replace direction of token with layer L, but keep the baseline L2 norm.
    # This isolates whether the norm or the direction is the problem.
    target_idx = words.index(count_word) if count_word in words else 0

    n_plots = len(valid_foci) + 1  # +1 for baseline
    ncols   = min(4, n_plots)
    nrows   = (n_plots + ncols - 1) // ncols
    fig2, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 4.5, nrows * 4.0))
    axes = np.array(axes).flatten()

    # Baseline
    feats_base = np.stack([all_hs[w][-2][0, all_idx[w], :].float().cpu().numpy()
                           for w in words])
    sim_base   = _cosine_matrix(feats_base)
    sns.heatmap(sim_base, annot=True, fmt=".3f", cmap="coolwarm",
                xticklabels=words, yticklabels=words,
                vmin=sim_base[~np.eye(len(words), dtype=bool)].min() - 0.01,
                vmax=1.0, ax=axes[0], linewidths=0.3, annot_kws={"size":6})
    axes[0].set_title(f"baseline layer -2\nmean={_off_diag_mean(sim_base):.4f}", fontsize=9)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].add_patch(plt.Rectangle((0, target_idx), len(words), 1,
                                     fill=False, edgecolor="lime", lw=2))

    for ax_i, layer in enumerate(valid_foci, start=1):
        feats = []
        for w in words:
            f_layer    = all_hs[w][layer][0, all_idx[w], :].float().cpu().numpy()
            f_baseline = all_hs[w][-2][0, all_idx[w], :].float().cpu().numpy()
            # Replace direction, keep baseline norm
            base_norm  = np.linalg.norm(f_baseline)
            layer_dir  = f_layer / (np.linalg.norm(f_layer) + 1e-8)
            feats.append(layer_dir * base_norm)
        feats = np.stack(feats)
        sim   = _cosine_matrix(feats)
        mean_sim = _off_diag_mean(sim)
        sns.heatmap(sim, annot=True, fmt=".3f", cmap="coolwarm",
                    xticklabels=words, yticklabels=words,
                    vmin=sim[~np.eye(len(words), dtype=bool)].min() - 0.01,
                    vmax=1.0, ax=axes[ax_i], linewidths=0.3, annot_kws={"size":6})
        axes[ax_i].set_title(f"norm-preserved layer {layer}\nmean={mean_sim:.4f}", fontsize=9)
        axes[ax_i].tick_params(axis='x', rotation=45)
        axes[ax_i].add_patch(plt.Rectangle((0, target_idx), len(words), 1,
                                            fill=False, edgecolor="lime", lw=2))

    for ax in axes[n_plots:]:
        ax.set_visible(False)
    fig2.suptitle(f"Norm-Preserved Direction Swap — confusion matrix\n"
                  f"(direction from layer L, norm from baseline layer -2)",
                  fontsize=10, y=1.01)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "h2_norm_preserved_confusion.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [saved] {p2}")


# ──────────────────────────────────────────────────────────────────────────────
# H3 – Feature extrapolation beyond layer-10 direction
# ──────────────────────────────────────────────────────────────────────────────

def h3_extrapolation(count_word, text_encoder, tokenizer, device, out_dir,
                     shallow_layer=10, scales=(0.5, 1.0, 1.5, 2.0, 3.0, 5.0)):
    """
    fused = baseline + scale * (layer_shallow - baseline)
    scale=0 → pure baseline
    scale=1 → pure layer_shallow (hard-replace)
    scale>1 → extrapolate further in the direction of layer_shallow
    """
    print(f"\n[H3] Extrapolation analysis (layer {shallow_layer}) …")

    all_hs  = {}
    all_idx = {}
    all_deep = {}
    for w in COUNTING_WORDS:
        prompt = MINIMAL_TEMPLATE.format(num=w)
        hs, ids, mask, tokens = _encode(prompt, text_encoder, tokenizer, device)
        cs, ce = _content_span(tokens)
        tidx = _find_idx(tokens[cs:ce], w)
        if tidx == -1:
            continue
        all_hs[w]   = hs
        all_idx[w]  = cs + tidx
        all_deep[w] = hs[-2][0, all_idx[w], :].float().cpu().numpy()

    words      = list(all_hs.keys())
    target_idx = words.index(count_word) if count_word in words else 0

    n_plots = len(scales)
    ncols   = min(3, n_plots)
    nrows   = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 4.0))
    axes = np.array(axes).flatten()

    summary_rows = []
    for ax_i, scale in enumerate(scales):
        feats = []
        for w in words:
            base  = all_deep[w]
            layer = all_hs[w][shallow_layer][0, all_idx[w], :].float().cpu().numpy()
            fused = base + scale * (layer - base)
            feats.append(fused)
        feats    = np.stack(feats)
        sim      = _cosine_matrix(feats)
        mean_sim = _off_diag_mean(sim)
        summary_rows.append((scale, mean_sim, sim[~np.eye(len(words),dtype=bool)].min()))

        sns.heatmap(sim, annot=True, fmt=".3f", cmap="coolwarm",
                    xticklabels=words, yticklabels=words,
                    vmin=sim[~np.eye(len(words), dtype=bool)].min() - 0.01,
                    vmax=1.0, ax=axes[ax_i], linewidths=0.3, annot_kws={"size":6})
        axes[ax_i].set_title(f"scale={scale:.1f}  mean={mean_sim:.4f}", fontsize=9)
        axes[ax_i].tick_params(axis='x', rotation=45)
        axes[ax_i].add_patch(plt.Rectangle((0, target_idx), len(words), 1,
                                            fill=False, edgecolor="lime", lw=2))

    for ax in axes[n_plots:]:
        ax.set_visible(False)
    fig.suptitle(
        f"H3: Extrapolation  fused = baseline + scale×(layer{shallow_layer} - baseline)\n"
        f"scale=0→baseline  scale=1→hard-replace  scale>1→extrapolate",
        fontsize=10, y=1.01)
    plt.tight_layout()
    p = os.path.join(out_dir, "h3_extrapolation_confusion.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {p}")

    # Print scale vs mean_sim curve
    print(f"  Extrapolation scale → off-diag mean similarity:")
    for sc, mn, mi in summary_rows:
        print(f"    scale={sc:.1f}  mean={mn:.4f}  min={mi:.4f}")

    return summary_rows


# ──────────────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────────────

def write_summary(prompt, count_word, out_dir, h3_rows):
    lines = [
        f"Diagnosis summary for: \"{prompt}\"",
        f"Counting word: \"{count_word}\"",
        "=" * 60,
        "",
        "H1 (DiT attention): See h1_token_attention_bar.png",
        "   If the bar for the counting word is near-zero compared to",
        "   other tokens, the DiT literally ignores it → text-embedding",
        "   modification cannot fix the problem.",
        "",
        "H2 (Norm analysis): See h2_norm_by_layer.png + h2_norm_preserved_confusion.png",
        "   If norm-preserved confusion << baseline confusion, the issue",
        "   was partly that layer-10 has different norm (OOD for DiT).",
        "   Norm-preserved blend should be tried for generation.",
        "",
        "H3 (Extrapolation): scale → off-diag mean similarity",
    ]
    for sc, mn, mi in (h3_rows or []):
        lines.append(f"   scale={sc:.1f}  mean={mn:.4f}  min={mi:.4f}")
    lines += [
        "",
        "   If similarity keeps dropping as scale increases but generation",
        "   doesn't change, the bottleneck is in the DiT (H1 likely).",
        "",
        "Recommended next experiments (in order):",
        "  1. Check h1_token_attention_bar.png first.",
        "  2. If H1 shows counting token is suppressed:",
        "     → Try attention re-weighting at DiT level (not text encoder)",
        "     → Or use negative CFG: negative_prompt='zero keyboards'",
        "  3. If H2 shows norm-preserved helps in feature space:",
        "     → Use blend mode with norm normalisation in generation script",
        "  4. If H3 shows scale=3+ gives much better feature separation:",
        "     → Try generation with large-scale extrapolation",
        "     → But watch for visual artifacts (may be too OOD for DiT)",
    ]
    txt_path = os.path.join(out_dir, "summary.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[saved] {txt_path}")
    print("\n".join(lines))


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", default="a photo of four computer keyboards")
    p.add_argument("--count_word", default="four")
    p.add_argument("--shallow_layer", type=int, default=10)
    p.add_argument("--out_dir", default="diagnose_dit")
    p.add_argument("--model_dir", default="ckpts/Z-Image-Turbo")
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--skip_h1", action="store_true",
                   help="Skip H1 (DiT forward pass) if it's too slow or fails")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print("Loading models …")
    model_path = ensure_model_weights(args.model_dir, verify=False)
    components = load_from_local_dir(model_path, device=device, dtype=torch.bfloat16)

    h3_rows = None

    if not args.skip_h1:
        h1_dit_attention(
            args.prompt, args.count_word, components, device,
            args.out_dir, args.max_seq_len,
        )

    h2_norm_analysis(
        args.count_word, components["text_encoder"], components["tokenizer"],
        device, args.out_dir,
    )

    h3_rows = h3_extrapolation(
        args.count_word, components["text_encoder"], components["tokenizer"],
        device, args.out_dir,
        shallow_layer=args.shallow_layer,
    )

    write_summary(args.prompt, args.count_word, args.out_dir, h3_rows)

    print(f"\n✓ Done. → {args.out_dir}/")


if __name__ == "__main__":
    main()
