"""
compare_dit_attention.py
========================
Compare DiT self-attention (unified image+caption sequence) between two prompts,
e.g. counting + easy noun vs counting + hard noun.

Intended workflow (same idea as a diagnose_dit_attention-style probe):
  - Encode two prompts through the text tower to cap_feats (as in inference).
  - Run one DiT forward at a fixed timestep with shared random latents (fair comparison).
  - In each main DiT block, compute attention softmax explicitly and record, for each
    target caption key position, the mean over heads of attention from *real image*
    queries to that key (then average across layers for the heatmap).

Usage
-----
  python compare_dit_attention.py \\
      --prompt_good "a photo of four cups" \\
      --prompt_bad  "a photo of four computer keyboards" \\
      --focus_word four \\
      --noun_good cups \\
      --noun_bad keyboards \\
      --out_dir compare_dit_attention_out
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import types
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from config import DEFAULT_HEIGHT, DEFAULT_INFERENCE_STEPS, DEFAULT_MAX_SEQUENCE_LENGTH, DEFAULT_WIDTH, SEQ_MULTI_OF
from utils import ensure_model_weights, load_from_local_dir
from utils.attention import _process_mask
from zimage.pipeline import calculate_shift, retrieve_timesteps
from zimage.transformer import ZImageAttention


# ──────────────────────────────────────────────────────────────────────────────
# Text helpers (same pattern as transplant_count_token.py)
# ──────────────────────────────────────────────────────────────────────────────


def _content_span(tokens: List[str]) -> Tuple[int, int]:
    cs, ce = 0, len(tokens)
    for i, t in enumerate(tokens):
        if "user" in t.lower():
            cs = i + 1
        elif "<|redacted_im_end|>" in t and i > cs:
            ce = i
            break
    return cs, ce


def _find_idx(tokens: List[str], word: str) -> int:
    w = word.lower()
    for i, t in enumerate(tokens):
        clean = t.lower().strip().replace(" ", "")
        if clean == w or w in clean:
            return i
    return -1


def encode_prompt_embeds(
    prompt: str,
    text_encoder,
    tokenizer,
    device: str,
    max_length: int,
):
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    enc = tokenizer(
        [formatted],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    ids = enc.input_ids.to(device)
    mask = enc.attention_mask.to(device).bool()
    with torch.no_grad():
        out = text_encoder(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
        )
    pe = out.hidden_states[-2][0][mask[0]]
    toks = [tokenizer.decode([t]) for t in ids[0][mask[0]].tolist()]
    return pe, toks


def resolve_token_index(tokens: List[str], word: str, label: str) -> int:
    cs, ce = _content_span(tokens)
    span = tokens[cs:ce]
    rel = _find_idx(span, word)
    if rel < 0:
        raise ValueError(f"Could not find {label!r} ({word!r}) in content tokens: {span}")
    return cs + rel


def image_grid_meta(
    latent_5d: torch.Tensor,
    patch_size: int,
    f_patch_size: int,
) -> Tuple[int, int, int, int, int]:
    """Returns (padded_image_len, F_t, H_t, W_t, ori_image_len)."""
    _, _, f_dim, h, w = latent_5d.shape
    ft = f_dim // f_patch_size
    ht = h // patch_size
    wt = w // patch_size
    ori = ft * ht * wt
    pad_len = (-ori) % SEQ_MULTI_OF
    return ori + pad_len, ft, ht, wt, ori


# ──────────────────────────────────────────────────────────────────────────────
# DiT attention capture (main transformer.layers only)
# ──────────────────────────────────────────────────────────────────────────────


def _apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast("cuda", enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)
        return x_out.type_as(x_in)


def capturing_attention_forward(
    self: ZImageAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    freqs_cis: Optional[torch.Tensor] = None,
):
    """Same as ZImageAttention.forward but records marginal image→key attention."""
    query = self.to_q(hidden_states)
    key = self.to_k(hidden_states)
    value = self.to_v(hidden_states)

    query = query.unflatten(-1, (self.n_heads, -1))
    key = key.unflatten(-1, (self.n_kv_heads, -1))
    value = value.unflatten(-1, (self.n_kv_heads, -1))

    if self.norm_q is not None:
        query = self.norm_q(query)
    if self.norm_k is not None:
        key = self.norm_k(key)

    if freqs_cis is not None:
        query = _apply_rotary_emb(query, freqs_cis)
        key = _apply_rotary_emb(key, freqs_cis)

    if self.n_kv_heads != self.n_heads:
        r = self.n_heads // self.n_kv_heads
        key = key.repeat_interleave(r, dim=2)
        value = value.repeat_interleave(r, dim=2)

    dtype = query.dtype
    query_f = query.float()
    key_f = key.float()
    value_f = value.float()

    q = query_f.transpose(1, 2)
    k = key_f.transpose(1, 2)
    scale = 1.0 / math.sqrt(self.head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if attention_mask is not None:
        add_mask = _process_mask(attention_mask, scores.dtype)
        scores = scores + add_mask

    probs = torch.softmax(scores, dim=-1)

    cap = getattr(self, "_dit_capture", None)
    if cap is not None:
        li = self._dit_layer_idx
        n_img = self._dit_n_img_tokens
        key_pos: Dict[str, int] = self._dit_key_pos
        for name, k_idx in key_pos.items():
            # [H, n_img] mean over batch 0
            col = probs[0, :, :n_img, k_idx].mean(dim=0).float().cpu().numpy()
            cap.setdefault(li, {})[name] = col

    v = value_f.transpose(1, 2)
    hidden_states = torch.matmul(probs, v)
    hidden_states = hidden_states.transpose(1, 2).contiguous().flatten(2, 3)
    hidden_states = hidden_states.to(dtype)
    return self.to_out[0](hidden_states)


def install_capture_hooks(transformer) -> List:
    originals = []
    for li, layer in enumerate(transformer.layers):
        attn = layer.attention
        originals.append(attn.forward)
        attn.forward = types.MethodType(capturing_attention_forward, attn)
        attn._dit_layer_idx = li
    return originals


def uninstall_capture_hooks(transformer, originals: List):
    for layer, orig in zip(transformer.layers, originals):
        attn = layer.attention
        attn.forward = orig
        for attr in ("_dit_capture", "_dit_layer_idx", "_dit_n_img_tokens", "_dit_key_pos"):
            if hasattr(attn, attr):
                delattr(attn, attr)


def configure_capture(
    transformer,
    capture: Dict[int, Dict[str, np.ndarray]],
    n_img_tokens: int,
    key_pos: Dict[str, int],
):
    for layer in transformer.layers:
        attn = layer.attention
        attn._dit_capture = capture
        attn._dit_n_img_tokens = n_img_tokens
        attn._dit_key_pos = key_pos


def run_dit_forward(
    transformer,
    latents_5d: torch.Tensor,
    timestep_1d: torch.Tensor,
    cap_feat: torch.Tensor,
    dtype: torch.dtype,
):
    """Single-sample forward matching zimage.pipeline (no CFG)."""
    device = latents_5d.device
    x = latents_5d.to(dtype)
    t = timestep_1d.to(device)
    cap = [cap_feat.to(dtype)]
    x_list = list(x.unbind(dim=0))
    transformer(x_list, t, cap)


def build_latents_and_timestep(
    transformer,
    scheduler,
    device: str,
    height: int,
    width: int,
    seed: int,
    step_index: int,
    num_inference_steps: int,
    dtype: torch.dtype,
):
    vae_scale_factor = 8 * 2
    height_latent = 2 * (int(height) // vae_scale_factor)
    width_latent = 2 * (int(width) // vae_scale_factor)
    shape = (1, transformer.in_channels, height_latent, width_latent)
    g = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(shape, generator=g, device=device, dtype=torch.float32)
    image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    scheduler.sigma_min = 0.0
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler, num_inference_steps, device, sigmas=None, mu=mu
    )
    if step_index < 0:
        step_index += len(timesteps)
    step_index = max(0, min(step_index, len(timesteps) - 1))
    t = timesteps[step_index]
    timestep = t.expand(latents.shape[0])
    timestep = (1000 - timestep) / 1000
    latent_in = latents.to(dtype).unsqueeze(2)
    return latent_in, timestep.to(device)


def marginal_to_2d(vec_1d: np.ndarray, ft: int, ht: int, wt: int) -> np.ndarray:
    """Average over frame axis so shape is (ht, wt)."""
    x = vec_1d[: ft * ht * wt].reshape(ft, ht, wt)
    return x.mean(axis=0)


def aggregate_layers(capture: Dict[int, Dict[str, np.ndarray]], name: str) -> np.ndarray:
    arrs = [capture[li][name] for li in sorted(capture.keys()) if name in capture[li]]
    if not arrs:
        raise RuntimeError(f"No capture data for {name!r}")
    return np.mean(np.stack(arrs, axis=0), axis=0)


def plot_comparison(
    good_four: np.ndarray,
    bad_four: np.ndarray,
    good_noun: np.ndarray,
    bad_noun: np.ndarray,
    ft: int,
    ht: int,
    wt: int,
    prompt_good: str,
    prompt_bad: str,
    focus_word: str,
    noun_good: str,
    noun_bad: str,
    out_path: str,
):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    maps = [
        (good_four, f'"{focus_word}" ← {noun_good} prompt'),
        (bad_four, f'"{focus_word}" ← {noun_bad} prompt'),
        (good_noun, f'"{noun_good}"'),
        (bad_noun, f'"{noun_bad}"'),
    ]
    allv = np.concatenate([m[0].ravel() for m in maps])
    vmin, vmax = float(np.percentile(allv, 5)), float(np.percentile(allv, 95))
    if vmax <= vmin:
        vmin, vmax = float(allv.min()), float(allv.max()) + 1e-8

    for ax, (grid, title) in zip(axes.ravel(), maps):
        g2 = marginal_to_2d(grid, ft, ht, wt)
        im = ax.imshow(g2, cmap="magma", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label="mean attn (img queries → key)")
    fig.suptitle(
        "DiT unified attention: mean over heads & main layers\n"
        f"Good: {prompt_good}\nBad: {prompt_bad}",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def scalar_summary(name: str, vec: np.ndarray) -> float:
    return float(np.mean(vec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt_good", type=str, default="a photo of four cups")
    ap.add_argument("--prompt_bad", type=str, default="a photo of four computer keyboards")
    ap.add_argument("--focus_word", type=str, default="four")
    ap.add_argument("--noun_good", type=str, default="cups")
    ap.add_argument("--noun_bad", type=str, default="keyboards")
    ap.add_argument("--model_dir", type=str, default="ckpts/Z-Image-Turbo")
    ap.add_argument("--out_dir", type=str, default="compare_dit_attention_out")
    ap.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    ap.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    ap.add_argument("--max_length", type=int, default=DEFAULT_MAX_SEQUENCE_LENGTH)
    ap.add_argument("--num_inference_steps", type=int, default=DEFAULT_INFERENCE_STEPS)
    ap.add_argument("--step_index", type=int, default=0, help="Scheduler step index for capture (0 = first)")
    ap.add_argument("--seed", type=int, default=0, help="Shared latent seed for both prompts")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)

    model_path = ensure_model_weights(args.model_dir, verify=False)
    components = load_from_local_dir(model_path, device=device, dtype=torch.bfloat16, compile=False)
    transformer = components["transformer"]
    text_encoder = components["text_encoder"]
    tokenizer = components["tokenizer"]
    scheduler = components["scheduler"]
    dit_dtype = next(transformer.parameters()).dtype

    patch_size = transformer.all_patch_size[0]
    f_patch_size = transformer.all_f_patch_size[0]

    latent_5d, timestep_1d = build_latents_and_timestep(
        transformer,
        scheduler,
        device,
        args.height,
        args.width,
        args.seed,
        args.step_index,
        args.num_inference_steps,
        dit_dtype,
    )

    padded_img_len, ft, ht, wt, n_img_ori = image_grid_meta(latent_5d, patch_size, f_patch_size)

    # ── Encode both prompts ─────────────────────────────────────────────────
    pe_g, tok_g = encode_prompt_embeds(
        args.prompt_good, text_encoder, tokenizer, device, args.max_length
    )
    pe_b, tok_b = encode_prompt_embeds(
        args.prompt_bad, text_encoder, tokenizer, device, args.max_length
    )

    idx_four_g = resolve_token_index(tok_g, args.focus_word, "focus_word (good)")
    idx_four_b = resolve_token_index(tok_b, args.focus_word, "focus_word (bad)")
    idx_ng = resolve_token_index(tok_g, args.noun_good, "noun_good")
    # allow "keyboard" substring for tokenizers that split "keyboards"
    try:
        idx_nb = resolve_token_index(tok_b, args.noun_bad, "noun_bad")
    except ValueError:
        idx_nb = resolve_token_index(tok_b, args.noun_bad.rstrip("s"), "noun_bad (fallback)")

    key_g = {
        "four": padded_img_len + idx_four_g,
        "noun": padded_img_len + idx_ng,
    }
    key_b = {
        "four": padded_img_len + idx_four_b,
        "noun": padded_img_len + idx_nb,
    }

    print(f"Device: {device}")
    print(f"Image tokens (real / padded): {n_img_ori} / {padded_img_len}  grid (F,H,W)=({ft},{ht},{wt})")
    print(f"Good prompt cap len={len(pe_g)} | '{args.focus_word}' cap_idx={idx_four_g} | '{args.noun_good}' cap_idx={idx_ng}")
    print(f"Bad  prompt cap len={len(pe_b)} | '{args.focus_word}' cap_idx={idx_four_b} | '{args.noun_bad}' cap_idx={idx_nb}")
    print(f"Unified key positions (good): {key_g}")
    print(f"Unified key positions (bad):  {key_b}")

    originals = install_capture_hooks(transformer)
    try:
        # Good prompt
        cap: Dict[int, Dict[str, np.ndarray]] = {}
        configure_capture(transformer, cap, n_img_ori, key_g)
        with torch.no_grad():
            run_dit_forward(transformer, latent_5d, timestep_1d, pe_g, dit_dtype)
        good_four = aggregate_layers(cap, "four")
        good_noun = aggregate_layers(cap, "noun")
        sum_g_four = scalar_summary("four", good_four)
        sum_g_noun = scalar_summary("noun", good_noun)

        # Bad prompt (same latents / timestep)
        cap = {}
        configure_capture(transformer, cap, n_img_ori, key_b)
        with torch.no_grad():
            run_dit_forward(transformer, latent_5d, timestep_1d, pe_b, dit_dtype)
        bad_four = aggregate_layers(cap, "four")
        bad_noun = aggregate_layers(cap, "noun")
        sum_b_four = scalar_summary("four", bad_four)
        sum_b_noun = scalar_summary("noun", bad_noun)
    finally:
        uninstall_capture_hooks(transformer, originals)

    print("\n--- Scalar mean (image queries → key), averaged over heads & main DiT layers ---")
    print(f"  '{args.focus_word}' (good / bad): {sum_g_four:.6e} / {sum_b_four:.6e}  (ratio bad/good: {sum_b_four / (sum_g_four + 1e-12):.4f})")
    print(f"  noun      (good / bad): {sum_g_noun:.6e} / {sum_b_noun:.6e}  (ratio bad/good: {sum_b_noun / (sum_g_noun + 1e-12):.4f})")

    out_png = os.path.join(args.out_dir, "dit_attention_compare.png")
    plot_comparison(
        good_four,
        bad_four,
        good_noun,
        bad_noun,
        ft,
        ht,
        wt,
        args.prompt_good,
        args.prompt_bad,
        args.focus_word,
        args.noun_good,
        args.noun_bad,
        out_png,
    )
    print(f"\nSaved figure: {out_png}")

    np.savez(
        os.path.join(args.out_dir, "dit_attention_vectors.npz"),
        good_four=good_four,
        bad_four=bad_four,
        good_noun=good_noun,
        bad_noun=bad_noun,
        padded_img_len=padded_img_len,
        n_img_ori=n_img_ori,
    )


if __name__ == "__main__":
    main()
