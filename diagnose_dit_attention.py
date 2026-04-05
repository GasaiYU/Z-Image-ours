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
                     max_seq_len=512, seed=42):
    """
    Capture image→text attention weights from the DiT.

    Architecture notes:
      ZImageTransformer2DModel concatenates image tokens and caption tokens into
      a single unified sequence [img_tokens | cap_tokens], then runs full
      self-attention on it.  There is no separate cross-attention module.

      dispatch_attention() calls F.scaled_dot_product_attention (or Flash
      Attention) which does not return attention weights.  We temporarily
      monkey-patch dispatch_attention to use manual softmax so we can capture
      the attention map, then restore it immediately after one forward pass.
    """
    print("\n[H1] Capturing DiT unified-attention (img→text subblock) …")

    import utils.attention as _attn_mod
    from utils.attention import dispatch_attention as _orig_dispatch

    text_encoder = components["text_encoder"]
    tokenizer    = components["tokenizer"]
    transformer  = components["transformer"]
    scheduler    = components["scheduler"]

    # ── Encode text ──────────────────────────────────────────────────────────
    hs, ids, mask, tokens = _encode(prompt, text_encoder, tokenizer, device, max_seq_len)
    deep_embeds = hs[-2].to(dtype=next(transformer.parameters()).dtype)

    # Count valid caption tokens
    valid_ids   = ids[0][mask[0]]
    valid_tokens = [tokenizer.decode([t]) for t in valid_ids.tolist()]
    cap_len = len(valid_ids)   # number of valid caption tokens (padded to max_seq_len)

    # Storage for captured attention submatrices [heads, img_tokens, cap_tokens]
    captured = []
    _x_len_ref = [None]  # will be filled once we know img token count

    def _manual_dispatch(query, key, value, attn_mask=None,
                         dropout_p=0.0, is_causal=False, scale=None, backend=None):
        """Manual softmax attention that captures the attention map."""
        # query/key/value: [B_or_seqlen, n_heads, head_dim]  (variable-length batch)
        # For variable-length (list) input the shapes may differ; handle tensor case only.
        if not isinstance(query, torch.Tensor):
            return _orig_dispatch(query, key, value, attn_mask=attn_mask,
                                  dropout_p=dropout_p, is_causal=is_causal,
                                  scale=scale, backend=backend)

        # query: [S, H, D]  (packed variable-length) OR [B, S, H, D]
        # Only capture if we have both img and cap tokens (unified layer call).
        # Heuristic: total seq len > cap_len  →  unified sequence
        seq_len = query.shape[-3] if query.dim() == 4 else query.shape[0]
        if seq_len > cap_len and _x_len_ref[0] is not None:
            x_len = _x_len_ref[0]
            # Compute attention map manually using float32
            q = query.float()
            k = key.float()
            v = value.float()

            if q.dim() == 3:  # [S, H, D] (packed)
                # Can't easily extract sub-blocks without knowing batch offsets;
                # skip this call.
                pass
            else:
                # [B, S, H, D] → transpose to [B, H, S, D]
                q = q.transpose(1, 2)
                k = k.transpose(1, 2)
                v = v.transpose(1, 2)
                sc = scale if scale is not None else (q.shape[-1] ** -0.5)
                scores = torch.einsum("bhid,bhjd->bhij", q, k) * sc
                if attn_mask is not None:
                    am = attn_mask
                    if am.dtype == torch.bool:
                        scores = scores.masked_fill(~am.unsqueeze(1).unsqueeze(1), float("-inf"))
                    else:
                        scores = scores + am
                weights = torch.softmax(scores, dim=-1)   # [B, H, S, S]
                # Extract img_tokens → cap_tokens subblock
                # unified order: [img(x_len) | cap(cap_len)]
                img_to_cap = weights[0, :, :x_len, x_len:x_len + cap_len]  # [H, img_len, cap_len]
                captured.append(img_to_cap.detach().cpu())

        return _orig_dispatch(query, key, value, attn_mask=attn_mask,
                              dropout_p=dropout_p, is_causal=is_causal,
                              scale=scale, backend=backend)

    # ── Prepare minimal latent for one forward step ───────────────────────────
    torch.manual_seed(seed)
    # Use a small latent: 64×64 pix / 8 = 8×8 latent patches
    # Actual size doesn't matter much — we only want attention statistics.
    latent_h, latent_w = 128, 128
    n_latent_ch = 16
    dtype = next(transformer.parameters()).dtype
    latents = torch.randn(1, n_latent_ch, latent_h, latent_w,
                          device=device, dtype=dtype)

    # Monkey-patch
    _attn_mod.dispatch_attention = _manual_dispatch

    try:
        scheduler.set_timesteps(20)
        t = scheduler.timesteps[:1].to(device)

        with torch.no_grad():
            result = transformer(
                hidden_states=latents,
                timestep=t,
                encoder_hidden_states=deep_embeds,
                encoder_attention_mask=mask.to(dtype=dtype),
                return_dict=False,
            )
        # After the call we know x_len from the captured data
        if captured:
            x_len_inferred = captured[0].shape[1]
            _x_len_ref[0] = x_len_inferred
    except TypeError:
        # Try alternate signature (some pipeline wrappers differ)
        try:
            with torch.no_grad():
                prompt_embeds = deep_embeds  # [1, S, D]
                result = transformer(
                    hidden_states=latents,
                    timestep=t,
                    encoder_hidden_states=prompt_embeds,
                    return_dict=False,
                )
        except Exception as e:
            print(f"  [warn] DiT forward failed: {e}")
    except Exception as e:
        print(f"  [warn] DiT forward failed: {e}")
    finally:
        _attn_mod.dispatch_attention = _orig_dispatch

    if not captured:
        # The transformer uses a different internal call path (e.g. variable-length
        # packing). Fall back: hook ZImageAttention directly.
        print("  [info] Batch-dispatch path missed; trying ZImageAttention hook …")
        captured2 = []

        def _attn_hook(module, inp, out):
            # inp[0]: hidden_states [B or S, seq, dim] or packed
            # We can't recover Q,K from output alone; skip silently.
            pass

        hooks2 = []
        from src.zimage.transformer import ZImageAttention
        for m in transformer.modules():
            if isinstance(m, ZImageAttention):
                hooks2.append(m.register_forward_hook(_attn_hook))

        print("  [warn] Variable-length packing prevents standard batch attention capture.")
        print("         Writing architecture note instead.")
        for h in hooks2:
            h.remove()

        with open(os.path.join(out_dir, "h1_architecture_note.txt"), "w") as f:
            f.write(
                "ZImageTransformer2DModel Architecture Note\n"
                "===========================================\n\n"
                "This DiT does NOT use cross-attention.\n"
                "Instead, image tokens and caption tokens are concatenated into\n"
                "a unified sequence: [img_tokens | cap_tokens].\n"
                "Full bidirectional self-attention is applied on this unified seq.\n\n"
                "Attention weights are not returned by dispatch_attention.\n"
                "Flash Attention / SDPA backends discard the weight matrix.\n\n"
                "To capture attention weights, the architecture needs to be run\n"
                "with NATIVE_MATH backend (no Flash), using packed variable-length\n"
                "sequences. This requires patching at the varlen attention level.\n\n"
                "Key implication for the counting problem:\n"
                "  The text tokens live in the same sequence as image tokens.\n"
                "  'four' token participates in self-attention with every image patch.\n"
                "  Whether 'four' is influential depends on whether the Q/K dot\n"
                "  products between image patches and 'four' are large.\n"
                "  This is determined by the deep text feature (hidden_states[-2])\n"
                "  that goes into the unified sequence.\n"
            )
        print(f"  [saved] {os.path.join(out_dir, 'h1_architecture_note.txt')}")
        return None, valid_tokens

    # ── Aggregate captured maps ───────────────────────────────────────────────
    # captured: list of [H, img_len, cap_len] per unified layer
    try:
        stacked = torch.stack(captured, dim=0)   # [L, H, img_len, cap_len]
    except RuntimeError:
        min_cap = min(c.shape[-1] for c in captured)
        min_img = min(c.shape[-2] for c in captured)
        stacked = torch.stack([c[:, :min_img, :min_cap] for c in captured], dim=0)

    # Mean over layers and heads → [img_len, cap_len]
    mean_attn = stacked.mean(dim=(0, 1)).numpy()
    # Mean over image positions → [cap_len]: how much each cap token is attended to
    per_cap_attn = mean_attn.mean(axis=0)   # [cap_len]

    # Align with valid_tokens
    n = min(len(per_cap_attn), len(valid_tokens))
    per_cap_attn  = per_cap_attn[:n]
    valid_tokens  = valid_tokens[:n]

    cs, ce = _content_span(valid_tokens)
    content_tokens = valid_tokens[cs:ce]
    content_attn   = per_cap_attn[cs:ce]
    token_labels   = [t.replace("▁", "").strip() or f"[{i}]"
                      for i, t in enumerate(content_tokens)]

    count_tidx = _find_idx(content_tokens, count_word)

    # Bar chart
    fig, ax = plt.subplots(figsize=(max(10, len(token_labels) * 0.5), 5))
    colors = ["#e74c3c" if i == count_tidx else "#3498db"
              for i in range(len(token_labels))]
    ax.bar(range(len(token_labels)), content_attn, color=colors)
    ax.set_xticks(range(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Attention Weight (img patches → this token, avg over layers & heads)")
    ax.set_title(
        f"DiT Unified Self-Attention: Image Patches → Caption Token\n"
        f"prompt: \"{prompt}\"  |  red = \"{count_word}\"")
    ax.grid(axis="y", alpha=0.3)
    if count_tidx >= 0:
        ax.annotate(
            f"↑ \"{count_word}\"\n({content_attn[count_tidx]:.4f})",
            xy=(count_tidx, content_attn[count_tidx]),
            xytext=(count_tidx + max(1, len(token_labels)//8),
                    content_attn[count_tidx] * 1.15),
            arrowprops=dict(arrowstyle="->"), fontsize=9, color="#c0392b")
    plt.tight_layout()
    p = os.path.join(out_dir, "h1_token_attention_bar.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {p}")

    # Heatmap: [img_patches × cap_tokens] (average over heads, first layer)
    first_layer_map = stacked[0].mean(dim=0).numpy()  # [img_len, cap_len]
    fig2, ax2 = plt.subplots(figsize=(max(8, n * 0.5), 5))
    im = ax2.imshow(first_layer_map[:, :n], aspect="auto", cmap="viridis")
    ax2.set_yticks([])
    ax2.set_xticks(range(n))
    ax2.set_xticklabels(
        [t.replace("▁","").strip() or f"[{i}]" for i, t in enumerate(valid_tokens[:n])],
        rotation=45, ha="right", fontsize=7)
    ax2.set_xlabel("Caption Token")
    ax2.set_ylabel("Image Patch (flattened)")
    ax2.set_title(f"Attention map — layer 0, avg over heads\n\"{prompt}\"")
    if count_tidx >= 0:
        full_count_idx = cs + count_tidx
        ax2.axvline(full_count_idx, color="red", linewidth=2, alpha=0.7,
                    label=f'"{count_word}"')
        ax2.legend(fontsize=8)
    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "h1_attention_heatmap.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  [saved] {p2}")

    # Print summary
    if count_tidx >= 0 and len(content_attn) > 0:
        rank = sorted(range(len(content_attn)),
                      key=lambda i: content_attn[i], reverse=True).index(count_tidx) + 1
        mass = content_attn[count_tidx] / (content_attn.sum() + 1e-8)
        print(f"\n  *** \"{count_word}\" attention summary ***")
        print(f"      mean weight : {content_attn[count_tidx]:.5f}")
        print(f"      rank        : {rank} / {len(content_attn)} content tokens")
        print(f"      % of mass   : {mass*100:.1f}%")
        if rank > len(content_attn) // 2:
            print("  ⚠  LOW attention rank — image patches barely look at this token.")
            print("     Text-embedding modification alone is unlikely to fix counting.")
        else:
            print("  ✓  Reasonable attention rank — counting token is noticed by DiT.")

    return per_cap_attn, valid_tokens


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
