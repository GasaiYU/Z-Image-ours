"""
test_noun_layer_injection.py
============================
Experiment: inject Layer-10 (count-aware) noun embedding into the deep embedding.

Motivation
----------
Causal attention means the COUNT token's embedding is context-independent.
But the NOUN token CAN see the count word on its left, so at intermediate
layers (e.g. Layer 10) the noun embedding IS count-aware.

For "easy" nouns (cups, dogs) this happens naturally because the LLM has
seen many "N cups / N dogs" patterns in training.
For "hard" nouns (keyboards, monitors) the LLM has seen very few such
patterns, so the noun's deep embedding ignores the count word.

Strategy
--------
For the noun token at position p in prompt "a photo of {count} {noun}":

  Method A – Hard Replace:
    final_embeds[p] = hidden_states[layer_src][p]

  Method B – Blend (interpolation):
    final_embeds[p] = (1-α) * hidden_states[-2][p]
                    + α    * hidden_states[layer_src][p]

  Method C – Direction Injection (norm-preserving):
    dir  = normalize(hidden_states[layer_src][p])
    norm = ||hidden_states[-2][p]||
    final_embeds[p] = hidden_states[-2][p] + α * dir * norm

We sweep α ∈ {0.1, 0.3, 0.5, 0.7, 1.0} and layer_src ∈ {8, 10, 12}
and generate images for each setting.

  Method D – Decay (weighted average over a layer range):
    weights[i] = exp(-decay_rate * i) / Z   (i=0 is route_start, highest weight)
    fused[p]   = Σ_i weights[i] * hidden_states[route_start+i][p]
    final_embeds[p] = fused[p]              (hard replace with the fused vector)

  Method E – Decay + Blend (softer variant):
    final_embeds[p] = (1-α) * hidden_states[-2][p] + α * fused[p]

Output
------
  <out_dir>/
    baseline/                   # no modification
    hard_L{layer}/              # hard replace with layer L
    blend_L{layer}_a{alpha}/    # blend
    inject_L{layer}_a{alpha}/   # direction injection
    summary_grid.png            # big comparison grid
"""

import argparse
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont
from pytorch_lightning import seed_everything

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from utils import ensure_model_weights, load_from_local_dir
from zimage.pipeline import generate


# ── text encoding helpers ─────────────────────────────────────────────────────

def encode_all_layers(prompt, text_encoder, tokenizer, device, max_seq_len=512):
    """Encode prompt; return all hidden states + token list."""
    messages  = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    enc = tokenizer(
        [formatted], padding="max_length", max_length=max_seq_len,
        truncation=True, return_tensors="pt",
    )
    ids  = enc.input_ids.to(device)
    mask = enc.attention_mask.to(device).bool()
    with torch.no_grad():
        out = text_encoder(input_ids=ids, attention_mask=mask,
                           output_hidden_states=True)
    valid  = ids[0][mask[0]]
    tokens = [tokenizer.decode([t]) for t in valid.tolist()]
    return out.hidden_states, tokens, ids, mask


def content_span(tokens):
    cs, ce = 0, len(tokens)
    for i, t in enumerate(tokens):
        if "user" in t.lower():
            cs = i + 1
        elif "<|im_end|>" in t and i > cs:
            ce = i; break
    return cs, ce


def find_noun_indices(tokens, nouns, cs, ce):
    """
    Find the full-sequence indices of all sub-tokens that belong to ANY word in
    `nouns` (a list of strings, e.g. ["computer", "keyboards"]).
    Returns list of (full_idx, sub_token), deduplicated and sorted.
    """
    content = tokens[cs:ce]
    hits = {}
    for noun in nouns:
        for i, t in enumerate(content):
            clean = t.lower().strip().replace(" ", "").replace("▁", "")
            if clean and (clean in noun.lower() or noun.lower() in clean):
                full_idx = cs + i
                hits[full_idx] = t   # deduplicate by index
    if not hits:
        # Fallback: last content token
        hits = {ce - 1: tokens[ce - 1]}
    return sorted(hits.items())   # [(full_idx, sub_token), ...]


# ── embedding modification ────────────────────────────────────────────────────

def build_modified_embeds(hs, noun_indices, method, layer_src, alpha,
                          route_start=8, route_end=13, decay_rate=0.3):
    """
    Returns modified deep embeds (hidden_states[-2]).

    method      : "baseline" | "hard" | "blend" | "inject" | "decay" | "decay_blend"
    layer_src   : int   – source layer for hard/blend/inject
    alpha       : float – blend/inject/decay_blend strength (0=no change, 1=full replace)
    route_start : int   – first layer (inclusive) for decay range
    route_end   : int   – last  layer (exclusive) for decay range
    decay_rate  : float – exponential decay rate (higher = more weight on route_start)
    """
    deep   = hs[-2].clone()          # [1, S, D]
    device = deep.device

    if method == "baseline":
        return deep

    # Pre-compute decayed fused feature if needed
    def _decay_fused(full_idx):
        rs = max(0, route_start)
        re = min(route_end, len(hs) - 1)
        layers = hs[rs:re]
        n = len(layers)
        w = torch.exp(-decay_rate * torch.arange(n, device=device,
                                                  dtype=torch.float32))
        w = w / w.sum()
        fused = torch.zeros_like(deep[0, full_idx, :], dtype=torch.float32)
        for i, lyr in enumerate(layers):
            fused += w[i] * lyr[0, full_idx, :].float()
        return fused.to(deep.dtype)

    for full_idx, _ in noun_indices:
        d = deep[0, full_idx, :]                          # [D]

        if method in ("decay", "decay_blend"):
            s = _decay_fused(full_idx)                    # weighted avg over range
        else:
            s = hs[layer_src][0, full_idx, :].to(d.dtype)

        if method == "hard":
            deep[0, full_idx, :] = s

        elif method == "blend":
            deep[0, full_idx, :] = (1 - alpha) * d + alpha * s

        elif method == "inject":
            s_norm = s / (s.norm() + 1e-8)
            deep[0, full_idx, :] = d + alpha * s_norm * d.norm()

        elif method == "decay":
            # Hard-replace with the decayed weighted average
            deep[0, full_idx, :] = s

        elif method == "decay_blend":
            # Softly blend decayed feature into the deep embedding
            deep[0, full_idx, :] = (1 - alpha) * d + alpha * s

    return deep


# ── generation ────────────────────────────────────────────────────────────────

def gen_image(components, prompt, modified_embeds, seed, device):
    """Generate one image with monkey-patched text encoder."""
    seed_everything(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    original_fwd = components["text_encoder"].forward

    def patched_fwd(input_ids, attention_mask, **kwargs):
        class O: pass
        o = O()
        o.hidden_states = [None] * 40
        o.hidden_states[-2] = modified_embeds.to(
            next(components["text_encoder"].parameters()).dtype)
        return o

    components["text_encoder"].forward = patched_fwd
    try:
        imgs = generate(
            transformer=components["transformer"],
            vae=components["vae"],
            text_encoder=components["text_encoder"],
            tokenizer=components["tokenizer"],
            scheduler=components["scheduler"],
            prompt=[prompt],
            height=1024, width=1024,
            num_inference_steps=8,
            guidance_scale=0.0,
            generator=generator,
        )
    finally:
        components["text_encoder"].forward = original_fwd

    return imgs[0]


# ── image helpers ─────────────────────────────────────────────────────────────

def add_label(img, text, font_size=22):
    """Overlay a text label on a PIL image (top-left corner)."""
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                                  font_size)
    except Exception:
        font = ImageFont.load_default()
    margin = 6
    bbox = draw.textbbox((margin, margin), text, font=font)
    draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2],
                   fill=(0, 0, 0, 180))
    draw.text((margin, margin), text, fill=(255, 255, 255), font=font)
    return out


def make_grid(images, labels, ncols=6, cell=256, title=""):
    """Build a flat image grid from a list of PIL images."""
    assert len(images) == len(labels)
    n     = len(images)
    nrows = (n + ncols - 1) // ncols
    pad   = 4
    header = 40 if title else 0
    W = ncols * (cell + pad) + pad
    H = header + nrows * (cell + pad) + pad
    grid = Image.new("RGB", (W, H), color=(30, 30, 30))
    if title:
        draw = ImageDraw.Draw(grid)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except Exception:
            font = ImageFont.load_default()
        draw.text((pad, 6), title, fill=(220, 220, 220), font=font)
    for i, (img, lbl) in enumerate(zip(images, labels)):
        r, c = divmod(i, ncols)
        x = pad + c * (cell + pad)
        y = header + pad + r * (cell + pad)
        thumb = img.resize((cell, cell), Image.LANCZOS)
        thumb = add_label(thumb, lbl, font_size=14)
        grid.paste(thumb, (x, y))
    return grid


# ── main ──────────────────────────────────────────────────────────────────────

def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading models …")
    model_path = ensure_model_weights(args.model_dir, verify=False)
    components = load_from_local_dir(model_path, device=device,
                                     dtype=torch.bfloat16)
    te = components["text_encoder"]
    tk = components["tokenizer"]

    # ── encode prompt once, get all hidden states ──────────────────────────
    print(f"\nEncoding: '{args.prompt}'")
    hs, tokens, ids, mask = encode_all_layers(
        args.prompt, te, tk, device, args.max_seq_len)
    cs, ce = content_span(tokens)
    noun_indices = find_noun_indices(tokens, args.nouns, cs, ce)

    print(f"Target nouns {args.nouns} found at full-sequence positions: "
          f"{[f'{idx}({t.strip()!r})' for idx, t in noun_indices]}")
    print(f"Total LLM layers (including embedding): {len(hs)}")

    # ── define experiment configs ──────────────────────────────────────────
    # Each entry: (method, layer_src, alpha, route_start, route_end, decay_rate)
    # layer_src / route_start / route_end / decay_rate only used by relevant methods
    configs = [("baseline", 0, 0.0, 0, 0, 0.0)]

    for lyr in args.src_layers:
        configs.append(("hard", lyr, 1.0, 0, 0, 0.0))

    for lyr in args.src_layers:
        for a in args.alphas:
            configs.append(("blend",  lyr, a, 0, 0, 0.0))
            configs.append(("inject", lyr, a, 0, 0, 0.0))

    # Decay configs: sweep decay_rate over fixed route range
    for dr in args.decay_rates:
        configs.append(("decay",       0, 1.0, args.route_start, args.route_end, dr))
        for a in args.alphas:
            configs.append(("decay_blend", 0, a,   args.route_start, args.route_end, dr))

    print(f"\n{len(configs)} configurations × {args.num_seeds} seeds "
          f"= {len(configs) * args.num_seeds} images\n")

    # ── generate ───────────────────────────────────────────────────────────
    all_images = []   # (label, seed, pil_image)

    for method, lyr, alpha, rs, re, dr in configs:
        if method == "baseline":
            label = "baseline"
        elif method == "hard":
            label = f"hard_L{lyr}"
        elif method == "blend":
            label = f"blend_L{lyr}_a{alpha:.1f}"
        elif method == "inject":
            label = f"inject_L{lyr}_a{alpha:.1f}"
        elif method == "decay":
            label = f"decay_L{rs}-{re}_dr{dr}"
        else:  # decay_blend
            label = f"decayblend_L{rs}-{re}_dr{dr}_a{alpha:.1f}"

        sub_dir = os.path.join(args.out_dir, label)
        os.makedirs(sub_dir, exist_ok=True)

        embeds = build_modified_embeds(hs, noun_indices, method, lyr, alpha,
                                       route_start=rs, route_end=re,
                                       decay_rate=dr)

        for seed in range(args.start_seed, args.start_seed + args.num_seeds):
            print(f"  [{label}] seed={seed} …", end=" ", flush=True)
            img = gen_image(components, args.prompt, embeds, seed, device)
            img_path = os.path.join(sub_dir, f"seed{seed}.png")
            img.save(img_path)
            all_images.append((label, seed, img))
            print("done")

    # ── build summary grid ─────────────────────────────────────────────────
    # Layout: rows = configs, columns = seeds
    print("\nBuilding summary grid …")
    n_cfg  = len(configs)
    n_seed = args.num_seeds
    cell   = 192
    pad    = 3
    lbl_w  = 200
    header = 50

    W = lbl_w + n_seed * (cell + pad) + pad
    H = header + n_cfg * (cell + pad) + pad
    grid = Image.new("RGB", (W, H), color=(20, 20, 20))

    draw = ImageDraw.Draw(grid)
    try:
        font_big  = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font_big = font_small = ImageFont.load_default()

    # Title
    draw.text((pad, 8), f"Noun Layer Injection: '{args.prompt}'  nouns={args.nouns}",
              fill=(220, 220, 220), font=font_big)

    # Column headers (seeds)
    for j, seed in enumerate(range(args.start_seed,
                                   args.start_seed + args.num_seeds)):
        x = lbl_w + pad + j * (cell + pad) + cell // 2 - 20
        draw.text((x, header - 18), f"seed={seed}", fill=(180, 180, 180),
                  font=font_small)

    # Row labels + images
    img_idx = 0
    for i, (method, lyr, alpha, rs, re, dr) in enumerate(configs):
        if method == "baseline":
            row_lbl = "BASELINE"
            row_color = (100, 200, 100)
        elif method == "hard":
            row_lbl = f"hard L{lyr}"
            row_color = (200, 100, 100)
        elif method == "blend":
            row_lbl = f"blend L{lyr} α={alpha:.1f}"
            row_color = (100, 150, 220)
        elif method == "inject":
            row_lbl = f"inject L{lyr} α={alpha:.1f}"
            row_color = (220, 160, 60)
        elif method == "decay":
            row_lbl = f"decay [{rs},{re}) dr={dr}"
            row_color = (180, 100, 220)
        else:  # decay_blend
            row_lbl = f"decay_blend [{rs},{re}) dr={dr} α={alpha:.1f}"
            row_color = (220, 180, 100)

        y_top = header + pad + i * (cell + pad)
        draw.text((pad, y_top + cell // 2 - 8), row_lbl,
                  fill=row_color, font=font_small)

        for j in range(n_seed):
            _, _, img = all_images[img_idx]; img_idx += 1
            x = lbl_w + pad + j * (cell + pad)
            thumb = img.resize((cell, cell), Image.LANCZOS)
            grid.paste(thumb, (x, y_top))

    grid_path = os.path.join(args.out_dir, "summary_grid.png")
    grid.save(grid_path)
    print(f"[saved] {grid_path}")

    print(f"All outputs in: {args.out_dir}/")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt",       default="a photo of four computer keyboards")
    p.add_argument("--nouns",        nargs="+", default=["computer", "keyboards"],
                   help="Words whose tokens will be modified (supports multiple)")
    p.add_argument("--model_dir",    default="ckpts/Z-Image-Turbo")
    p.add_argument("--out_dir",      default="noun_injection_results")
    # single-layer methods
    p.add_argument("--src_layers",   type=int, nargs="+", default=[8, 10, 12],
                   help="Source layers for hard/blend/inject methods")
    p.add_argument("--alphas",       type=float, nargs="+", default=[0.3, 0.5, 0.7],
                   help="Blend / inject / decay_blend strengths")
    # decay method
    p.add_argument("--route_start",  type=int, default=8,
                   help="First layer (inclusive) for decay range")
    p.add_argument("--route_end",    type=int, default=13,
                   help="Last layer (exclusive) for decay range, e.g. 13 = layers 8-12")
    p.add_argument("--decay_rates",  type=float, nargs="+", default=[0.1, 0.3, 0.5],
                   help="Exponential decay rates for decay/decay_blend methods")
    # generation
    p.add_argument("--num_seeds",    type=int, default=4)
    p.add_argument("--start_seed",   type=int, default=42)
    p.add_argument("--max_seq_len",  type=int, default=512)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
