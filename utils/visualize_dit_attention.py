"""
visualize_dit_attention.py
==========================
Visualize the self-attention maps of the DiT (ZImageTransformer2DModel)
between image patches and text tokens, across different layers and
denoising timesteps.

Background
----------
Z-Image's DiT uses FULL self-attention on a unified sequence:
    [image_patches (x_len) | text_tokens (cap_len)]
There is NO separate cross-attention. To get the "cross-attention-like" map
(how each image patch attends to each text token), we extract the sub-matrix:
    attn_weights[image_patch_idx, text_token_idx]
from the full self-attention matrix.

For proper softmax normalization, we compute logits between each image patch
query and ALL (image+text) keys, apply softmax, then keep only the text-token
columns. This is done in chunks to avoid OOM.

Usage
-----
    cd /path/to/Z-Image-ours
    python utils/visualize_dit_attention.py \\
        --prompt "a photo of four computer keyboards" \\
        --target_tokens four keyboards \\
        --vis_layers 5 10 15 20 25 \\
        --vis_timestep_indices 0 2 5 7 \\
        --outdir outputs/dit_attention

Output
------
outputs/dit_attention/<prompt>/
    generated.png              - the generated image
    t00_layer05_four.png       - heatmap for token "four" at step 0, layer 5
    ...
    summary_four.png           - grid: rows=timesteps, cols=layers
    summary_keyboards.png
"""

import argparse
import math
import os
import re
import sys
from typing import Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_lightning import seed_everything

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC  = os.path.join(_ROOT, "src")
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import ensure_model_weights, load_from_local_dir, set_attention_backend
from zimage import generate
from zimage.transformer import ZImageAttention, apply_rotary_emb

torch.set_grad_enabled(False)


# ── token utilities ────────────────────────────────────────────────────────────

def content_span(tokens: List[str]) -> Tuple[int, int]:
    """Return [start, end) indices of the user-message content tokens."""
    cs, ce = 0, len(tokens)
    for i, t in enumerate(tokens):
        if "user" in t.lower():
            cs = i + 1
        elif "<|im_end|>" in t and i > cs:
            ce = i
            break
    return cs, ce


def find_token_indices(
    valid_ids: torch.Tensor,
    tokenizer,
    target_words: List[str],
) -> Dict[str, List[int]]:
    """
    Map each target word to the indices of its matching tokens inside cap_feats.

    cap_feats for sample 0 = text_encoder_output[valid_mask], so the index into
    cap_feats equals the position in the list of decoded valid tokens.
    A single word may span multiple sub-word tokens (all indices are returned).
    """
    tokens = [tokenizer.decode([t]) for t in valid_ids.tolist()]
    cs, ce = content_span(tokens)
    content_tokens = tokens[cs:ce]

    result: Dict[str, List[int]] = {w: [] for w in target_words}

    # Build a concatenated string from cleaned sub-words for substring search
    for word in target_words:
        w_clean = word.lower().replace(" ", "")
        for i, t in enumerate(content_tokens):
            clean = t.lower().strip().replace(" ", "").replace("▁", "").replace("Ġ", "")
            if clean == w_clean or re.search(r"\b" + re.escape(w_clean) + r"\b", clean):
                result[word].append(cs + i)  # absolute index in valid tokens

    return result


# ── attention capture infrastructure ──────────────────────────────────────────

class AttentionCapture:
    """
    Stores cross-attention-like maps captured during the denoising loop.

    maps[timestep_idx][layer_idx] = FloatTensor [x_len, cap_len]
        rows  = image patch tokens
        cols  = text (cap) tokens
    """

    def __init__(self, capture_layers: List[int], capture_timestep_indices: Set[int]):
        self.capture_layers = set(capture_layers)
        self.capture_timestep_indices = set(capture_timestep_indices)
        self.timestep_idx: int = 0
        self.x_lens: List[int] = []
        self.cap_lens: List[int] = []
        self.maps: Dict[int, Dict[int, torch.Tensor]] = {}

    def should_capture(self, layer_idx: int) -> bool:
        return (
            self.timestep_idx in self.capture_timestep_indices
            and layer_idx in self.capture_layers
            and bool(self.x_lens)
        )

    def store(self, layer_idx: int, cross_map: torch.Tensor) -> None:
        t = self.timestep_idx
        if t not in self.maps:
            self.maps[t] = {}
        self.maps[t][layer_idx] = cross_map


def _compute_cross_attn_chunked(
    query: torch.Tensor,   # [B, H, S, D_h]  after RoPE
    key:   torch.Tensor,   # [B, H, S, D_h]  after RoPE
    x_len: int,
    cap_len: int,
    chunk_size: int = 512,
) -> torch.Tensor:
    """
    Compute the image-patch → text-token attention (head-averaged) in chunks
    to avoid materialising the full S×S attention matrix.

    Returns FloatTensor [x_len, cap_len] on CPU.
    """
    scale = 1.0 / math.sqrt(query.shape[-1])
    # batch item 0 only
    q_img = query[0, :, :x_len, :]                   # [H, x_len, D_h]
    k_all = key[0, :, : x_len + cap_len, :]          # [H, S_all, D_h]

    n_chunks = math.ceil(x_len / chunk_size)
    cross_chunks: List[torch.Tensor] = []

    for ci in range(n_chunks):
        s = ci * chunk_size
        e = min(s + chunk_size, x_len)

        q_chunk = q_img[:, s:e, :]                   # [H, chunk, D_h]
        logits = torch.matmul(q_chunk, k_all.transpose(-2, -1)) * scale   # [H, chunk, S_all]
        attn_w = torch.softmax(logits.float(), dim=-1)                     # [H, chunk, S_all]

        # Keep only text-token columns and average over heads
        cross_chunk = attn_w[:, :, x_len: x_len + cap_len]                # [H, chunk, cap_len]
        cross_chunks.append(cross_chunk.mean(dim=0).cpu())                 # [chunk, cap_len]

    return torch.cat(cross_chunks, dim=0)   # [x_len, cap_len]


def install_hooks(
    transformer,
    capture: AttentionCapture,
) -> Tuple[Dict[int, Tuple], object]:
    """
    Monkey-patch the transformer to capture attention maps during generation.

    Returns (patched_attns, original_pae) for cleanup.
    """

    # ── 1. patchify_and_embed: capture x_len and cap_len ──────────────────────
    original_pae = transformer.patchify_and_embed

    def patched_pae(all_image, all_cap_feats, patch_size, f_patch_size):
        result = original_pae(all_image, all_cap_feats, patch_size, f_patch_size)
        x_embed, cap_embed = result[0], result[1]
        capture.x_lens  = [len(v) for v in x_embed]
        capture.cap_lens = [len(v) for v in cap_embed]
        return result

    transformer.patchify_and_embed = patched_pae

    # ── 2. ZImageAttention.forward: compute + store cross-attn map ────────────
    patched_attns: Dict[int, Tuple] = {}

    for layer_idx, layer in enumerate(transformer.layers):
        if layer_idx not in capture.capture_layers:
            continue

        attn = layer.attention
        original_attn_fwd = attn.forward  # bound method

        def make_patched_fwd(layer_id: int, orig_fwd, attn_module: ZImageAttention):
            def patched_fwd(
                hidden_states: torch.Tensor,
                attention_mask=None,
                freqs_cis=None,
            ) -> torch.Tensor:
                # Always run the real forward first (for correctness / grad flow)
                output = orig_fwd(hidden_states, attention_mask, freqs_cis)

                if capture.should_capture(layer_id):
                    with torch.no_grad():
                        # Re-compute Q, K with RoPE in float32
                        hs = hidden_states.float()
                        q = attn_module.to_q(hs)
                        k = attn_module.to_k(hs)

                        q = q.unflatten(-1, (attn_module.n_heads,    attn_module.head_dim))
                        k = k.unflatten(-1, (attn_module.n_kv_heads, attn_module.head_dim))

                        if attn_module.norm_q is not None:
                            q = attn_module.norm_q(q)
                        if attn_module.norm_k is not None:
                            k = attn_module.norm_k(k)

                        if freqs_cis is not None:
                            q = apply_rotary_emb(q, freqs_cis)
                            k = apply_rotary_emb(k, freqs_cis)

                        # [B, S, H, D_h] -> [B, H, S, D_h]
                        q = q.permute(0, 2, 1, 3)
                        k = k.permute(0, 2, 1, 3)

                        x_len   = capture.x_lens[0]
                        cap_len = capture.cap_lens[0]

                        cross_map = _compute_cross_attn_chunked(q, k, x_len, cap_len)
                        capture.store(layer_id, cross_map)

                return output

            return patched_fwd

        patched_fwd = make_patched_fwd(layer_idx, original_attn_fwd, attn)
        patched_attns[layer_idx] = (attn, original_attn_fwd)
        attn.forward = patched_fwd   # set instance attribute (shadows class method)

    return patched_attns, original_pae


def remove_hooks(
    transformer,
    patched_attns: Dict[int, Tuple],
    original_pae,
) -> None:
    """Restore all patched methods."""
    transformer.patchify_and_embed = original_pae
    for layer_idx, (attn, orig_fwd) in patched_attns.items():
        attn.forward = orig_fwd   # restore bound method


# ── visualisation ──────────────────────────────────────────────────────────────

def _get_attn_for_word(
    cross_map: torch.Tensor,   # [x_len, cap_len]
    indices: List[int],
) -> Optional[np.ndarray]:
    """Average attention over all sub-token indices for one word. Returns 1-D array."""
    cols = [cross_map[:, ti] for ti in indices if ti < cross_map.shape[1]]
    if not cols:
        return None
    attn = torch.stack(cols, dim=0).mean(dim=0).numpy()   # [x_len]
    return attn


def _spatial_map(
    attn_1d: np.ndarray,
    grid_h: int,
    grid_w: int,
) -> np.ndarray:
    """Reshape head-averaged attention [x_len, ...] to [grid_h, grid_w]."""
    n_real = grid_h * grid_w
    arr = attn_1d[:n_real].reshape(grid_h, grid_w)
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def visualize_attention(
    capture: AttentionCapture,
    token_indices: Dict[str, List[int]],
    img_h: int,
    img_w: int,
    patch_size: int,
    vae_scale: int,
    outdir: str,
    gen_image: Optional[Image.Image] = None,
) -> None:
    """Save per-(timestep, layer, word) heatmaps and per-word summary grids."""
    os.makedirs(outdir, exist_ok=True)

    # Patch grid dimensions
    latent_h = img_h // vae_scale
    latent_w = img_w // vae_scale
    grid_h   = latent_h // patch_size
    grid_w   = latent_w // patch_size
    print(f"  Patch grid: {grid_h}×{grid_w} = {grid_h * grid_w} patches")

    if gen_image is not None:
        gen_image.save(os.path.join(outdir, "generated.png"))

    # ── individual heatmaps ────────────────────────────────────────────────────
    for t_idx, layer_maps in sorted(capture.maps.items()):
        for layer_idx, cross_map in sorted(layer_maps.items()):
            for word, indices in token_indices.items():
                if not indices:
                    continue
                attn_1d = _get_attn_for_word(cross_map, indices)
                if attn_1d is None:
                    continue
                spatial = _spatial_map(attn_1d, grid_h, grid_w)

                if gen_image is not None:
                    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
                    axes[0].imshow(gen_image)
                    axes[0].set_title("Generated Image")
                    axes[0].axis("off")
                    ax = axes[1]
                else:
                    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

                im = ax.imshow(spatial, cmap="hot", interpolation="bilinear",
                               vmin=0, vmax=1)
                ax.set_title(f'Token: "{word}"  |  Layer {layer_idx}  |  Step {t_idx}',
                             fontsize=10)
                ax.axis("off")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                fname = f"t{t_idx:02d}_layer{layer_idx:02d}_{word.replace(' ', '_')}.png"
                plt.tight_layout()
                plt.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved: {fname}")

    # ── summary grids (one per word) ───────────────────────────────────────────
    t_idxs     = sorted(capture.maps.keys())
    all_layers = sorted({l for tm in capture.maps.values() for l in tm.keys()})

    if not t_idxs or not all_layers:
        return

    nrows, ncols = len(t_idxs), len(all_layers)

    for word, indices in token_indices.items():
        if not indices:
            print(f"  Warning: token not found for '{word}', skipping summary.")
            continue

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(max(2.5 * ncols, 4), max(2.5 * nrows, 4)),
            squeeze=False,
        )
        fig.suptitle(f'Attention to "{word}"  (rows=timestep, cols=layer)', fontsize=12)

        for ri, t_idx in enumerate(t_idxs):
            for ci, layer_idx in enumerate(all_layers):
                ax = axes[ri][ci]
                cross_map = capture.maps.get(t_idx, {}).get(layer_idx)
                if cross_map is None:
                    ax.set_visible(False)
                    continue

                attn_1d = _get_attn_for_word(cross_map, indices)
                if attn_1d is None:
                    ax.set_visible(False)
                    continue

                spatial = _spatial_map(attn_1d, grid_h, grid_w)
                ax.imshow(spatial, cmap="hot", interpolation="bilinear",
                          vmin=0, vmax=1)
                ax.set_title(f"L{layer_idx}", fontsize=7)
                if ci == 0:
                    ax.set_ylabel(f"t={t_idx}", fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        summary_path = os.path.join(outdir, f"summary_{word.replace(' ', '_')}.png")
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Summary: summary_{word.replace(' ', '_')}.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main(opt: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    seed_everything(opt.seed)

    model_path = ensure_model_weights("ckpts/Z-Image-Turbo", verify=False)
    components  = load_from_local_dir(model_path, device=device,
                                      dtype=torch.bfloat16, compile=False)
    attn_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")
    set_attention_backend(attn_backend)

    tokenizer    = components["tokenizer"]
    transformer  = components["transformer"]

    # ── find target token positions in cap_feats ───────────────────────────────
    messages  = [{"role": "user", "content": opt.prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    enc = tokenizer(
        [formatted],
        padding="max_length",
        max_length=opt.max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    valid_ids = enc.input_ids[0][enc.attention_mask[0].bool()]
    token_indices = find_token_indices(valid_ids, tokenizer, opt.target_tokens)

    print("\nToken positions in cap_feats:")
    decoded = [tokenizer.decode([t]) for t in valid_ids.tolist()]
    for word, idxs in token_indices.items():
        matched = [decoded[i] for i in idxs]
        print(f"  '{word}' → indices {idxs}  =  {matched}")

    # ── configure capture ──────────────────────────────────────────────────────
    n_layers   = len(transformer.layers)
    vis_layers = opt.vis_layers if opt.vis_layers else [
        l for l in [0, 5, 10, 15, 20, 25, n_layers - 1] if l < n_layers
    ]
    vis_layers = sorted(set(l for l in vis_layers if 0 <= l < n_layers))
    vis_steps  = set(opt.vis_timestep_indices)

    print(f"\nCapturing layers:        {vis_layers}")
    print(f"Capturing step indices:  {sorted(vis_steps)}")

    capture = AttentionCapture(vis_layers, vis_steps)

    # ── wrap transformer.forward to track timestep index ──────────────────────
    original_transformer_fwd = transformer.forward
    call_count = [0]

    def counting_fwd(x, t, cap_feats, patch_size=2, f_patch_size=1):
        capture.timestep_idx = call_count[0]
        call_count[0] += 1
        return original_transformer_fwd(x, t, cap_feats, patch_size, f_patch_size)

    transformer.forward = counting_fwd

    # ── install attention hooks ────────────────────────────────────────────────
    patched_attns, original_pae = install_hooks(transformer, capture)

    print(f"\nGenerating: '{opt.prompt}'")
    try:
        generator = torch.Generator(device).manual_seed(opt.seed)
        images = generate(
            prompt=[opt.prompt],
            **components,
            height=opt.H,
            width=opt.W,
            num_inference_steps=opt.steps,
            guidance_scale=opt.scale,
            generator=generator,
            max_sequence_length=opt.max_sequence_length,
        )
        gen_image = images[0]
    finally:
        remove_hooks(transformer, patched_attns, original_pae)
        transformer.forward = original_transformer_fwd

    # ── visualise ──────────────────────────────────────────────────────────────
    # vae_scale = vae_scale_factor * 2 = 8 * 2 = 16
    vae_scale = 16
    total_captures = sum(len(v) for v in capture.maps.values())
    print(f"\nTotal captured maps: {total_captures}")

    slug = re.sub(r"[^a-z0-9]+", "_", opt.prompt.lower())[:50].strip("_")
    outdir = os.path.join(opt.outdir, slug)

    visualize_attention(
        capture, token_indices,
        img_h=opt.H, img_w=opt.W,
        patch_size=2, vae_scale=vae_scale,
        outdir=outdir,
        gen_image=gen_image,
    )
    print(f"\nDone. Results saved to: {outdir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize DiT cross-attention maps for target tokens."
    )
    p.add_argument("--prompt", type=str,
                   default="a photo of four computer keyboards")
    p.add_argument("--target_tokens", type=str, nargs="+",
                   default=["four", "keyboards"],
                   help="Words to visualize attention for")
    p.add_argument("--outdir",  type=str, default="outputs/dit_attention")
    p.add_argument("--steps",   type=int, default=8)
    p.add_argument("--H",       type=int, default=1024)
    p.add_argument("--W",       type=int, default=1024)
    p.add_argument("--scale",   type=float, default=0.0)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--vis_layers", type=int, nargs="+", default=None,
                   help="Layer indices to capture. Default: 0,5,10,15,20,25,29")
    p.add_argument("--vis_timestep_indices", type=int, nargs="+", default=[0, 2, 5, 7],
                   help="Denoising step indices to capture (0=first step)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
