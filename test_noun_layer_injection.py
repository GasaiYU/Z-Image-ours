"""
test_noun_layer_injection.py
============================
Apply exponential-decay layer fusion to target tokens (count words and/or nouns)
and compare against baseline generation.

For each target token, replace its deep embedding (hidden_states[-2]) with a
weighted average over a range of intermediate layers:

    weights[i] = exp(-decay_rate * i) / Z   (i=0 → route_start, highest weight)
    fused[p]   = Σ_i  weights[i] * hidden_states[route_start + i][p]

Count words and nouns can use independent layer ranges and decay rates.
"""

import argparse
import os
import sys

import torch
from PIL import Image, ImageDraw, ImageFont
from pytorch_lightning import seed_everything

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
from utils import ensure_model_weights, load_from_local_dir
from zimage.pipeline import generate

QUANTITY_BANK = [
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
]


# ── text encoding ─────────────────────────────────────────────────────────────

def encode_all_layers(prompt, text_encoder, tokenizer, device, max_seq_len=512):
    messages  = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    enc = tokenizer([formatted], padding="max_length", max_length=max_seq_len,
                    truncation=True, return_tensors="pt")
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


def find_token_indices(tokens, words, cs, ce):
    """Return sorted list of (full_seq_idx, token_str) for tokens matching any word."""
    content = tokens[cs:ce]
    hits = {}
    for word in words:
        for i, t in enumerate(content):
            clean = t.lower().strip().replace(" ", "").replace("▁", "")
            if clean and (clean in word.lower() or word.lower() in clean):
                hits[cs + i] = t
    return sorted(hits.items())


# ── decay fusion ──────────────────────────────────────────────────────────────

def decay_fuse(hs, full_idx, route_start, route_end, decay_rate, device, ref_dtype):
    """Compute decay-weighted average of hs[route_start:route_end] at full_idx."""
    rs = max(0, route_start)
    re = min(route_end, len(hs) - 1)
    layers = hs[rs:re]
    n = len(layers)
    w = torch.exp(-decay_rate * torch.arange(n, device=device, dtype=torch.float32))
    w = w / w.sum()
    fused = torch.zeros_like(hs[-2][0, full_idx, :], dtype=torch.float32)
    for i, lyr in enumerate(layers):
        fused += w[i] * lyr[0, full_idx, :].float()
    return fused.to(ref_dtype)


def build_decay_embeds(hs, count_indices, noun_indices,
                       count_rs, count_re, count_dr,
                       noun_rs, noun_re, noun_dr):
    """
    Apply decay fusion to count tokens and/or noun tokens.
    Pass empty list to skip a group.
    """
    deep   = hs[-2].clone()
    device = deep.device
    dtype  = deep.dtype

    for full_idx, _ in count_indices:
        deep[0, full_idx, :] = decay_fuse(
            hs, full_idx, count_rs, count_re, count_dr, device, dtype)

    for full_idx, _ in noun_indices:
        deep[0, full_idx, :] = decay_fuse(
            hs, full_idx, noun_rs, noun_re, noun_dr, device, dtype)

    return deep


# ── generation ────────────────────────────────────────────────────────────────

def gen_image(components, prompt, embeds, seed, device):
    seed_everything(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    orig = components["text_encoder"].forward

    def patched(input_ids, attention_mask, **kwargs):
        class O: pass
        o = O()
        o.hidden_states = [None] * 40
        o.hidden_states[-2] = embeds.to(
            next(components["text_encoder"].parameters()).dtype)
        return o

    components["text_encoder"].forward = patched
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
        components["text_encoder"].forward = orig

    return imgs[0]


# ── grid builder ──────────────────────────────────────────────────────────────

def build_grid(all_images, configs, args):
    """
    all_images: list of (label, seed, pil_image) in order
    configs   : list of (label, ...) matching all_images order
    """
    n_cfg  = len(configs)
    n_seed = args.num_seeds
    cell   = 192
    pad    = 3
    lbl_w  = 260
    header = 60

    try:
        font_big   = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except Exception:
        font_big = font_small = ImageFont.load_default()

    W = lbl_w + n_seed * (cell + pad) + pad
    H = header + n_cfg * (cell + pad) + pad
    grid = Image.new("RGB", (W, H), (20, 20, 20))
    draw = ImageDraw.Draw(grid)

    title = (f"Decay fusion — '{args.prompt}'\n"
             f"count={args.count_words or '(none)'}  "
             f"nouns={args.nouns or '(none)'}")
    draw.text((pad, 6), title, fill=(220, 220, 220), font=font_big)

    for j in range(n_seed):
        x = lbl_w + pad + j * (cell + pad) + cell // 2 - 25
        draw.text((x, header - 16), f"seed {args.start_seed + j}",
                  fill=(160, 160, 160), font=font_small)

    img_idx = 0
    for i, cfg in enumerate(configs):
        label = cfg["label"]
        color = (100, 220, 100) if label == "baseline" else (180, 140, 220)
        y_top = header + pad + i * (cell + pad)
        draw.text((pad, y_top + cell // 2 - 8), label,
                  fill=color, font=font_small)
        for j in range(n_seed):
            _, _, img = all_images[img_idx]; img_idx += 1
            x = lbl_w + pad + j * (cell + pad)
            grid.paste(img.resize((cell, cell), Image.LANCZOS), (x, y_top))

    return grid


# ── main ──────────────────────────────────────────────────────────────────────

def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model_path = ensure_model_weights(args.model_dir, verify=False)
    components = load_from_local_dir(model_path, device=device, dtype=torch.bfloat16)

    print(f"\nEncoding: '{args.prompt}'")
    hs, tokens, ids, mask = encode_all_layers(
        args.prompt, components["text_encoder"], components["tokenizer"],
        device, args.max_seq_len)
    cs, ce = content_span(tokens)

    # Find count-word token indices
    count_words = args.count_words if args.count_words else []
    count_indices = find_token_indices(tokens, count_words, cs, ce)

    # Find noun token indices
    noun_words = args.nouns if args.nouns else []
    noun_indices = find_token_indices(tokens, noun_words, cs, ce)

    print(f"Count tokens : {[(i, t.strip()) for i, t in count_indices]}")
    print(f"Noun  tokens : {[(i, t.strip()) for i, t in noun_indices]}")
    print(f"Total layers : {len(hs)}")

    # ── build experiment configs ───────────────────────────────────────────
    configs = [{"label": "baseline"}]

    # Sweep: noun decay only
    for dr in args.noun_decay_rates:
        configs.append({
            "label":    f"noun_decay  [{args.noun_rs},{args.noun_re}) dr={dr}",
            "noun_dr":  dr,
            "count_dr": None,
        })

    # Sweep: count decay only
    for dr in args.count_decay_rates:
        configs.append({
            "label":    f"count_decay [{args.count_rs},{args.count_re}) dr={dr}",
            "noun_dr":  None,
            "count_dr": dr,
        })

    # Sweep: both
    for n_dr in args.noun_decay_rates:
        for c_dr in args.count_decay_rates:
            configs.append({
                "label":    (f"both  noun[{args.noun_rs},{args.noun_re})dr={n_dr}"
                             f" cnt[{args.count_rs},{args.count_re})dr={c_dr}"),
                "noun_dr":  n_dr,
                "count_dr": c_dr,
            })

    print(f"\n{len(configs)} configs × {args.num_seeds} seeds = "
          f"{len(configs) * args.num_seeds} images\n")

    # ── generate ──────────────────────────────────────────────────────────
    all_images = []

    for cfg in configs:
        label = cfg["label"]
        sub   = os.path.join(args.out_dir, label.replace(" ", "_").replace("/", "-"))
        os.makedirs(sub, exist_ok=True)

        if label == "baseline":
            embeds = hs[-2].clone()
        else:
            n_dr = cfg["noun_dr"]
            c_dr = cfg["count_dr"]
            embeds = build_decay_embeds(
                hs,
                count_indices if c_dr is not None else [],
                noun_indices  if n_dr is not None else [],
                args.count_rs, args.count_re, c_dr if c_dr is not None else 0.0,
                args.noun_rs,  args.noun_re,  n_dr if n_dr is not None else 0.0,
            )

        for seed in range(args.start_seed, args.start_seed + args.num_seeds):
            print(f"  [{label}] seed={seed} …", end=" ", flush=True)
            img = gen_image(components, args.prompt, embeds, seed, device)
            img.save(os.path.join(sub, f"seed{seed}.png"))
            all_images.append((label, seed, img))
            print("done")

    # ── summary grid ──────────────────────────────────────────────────────
    grid = build_grid(all_images, configs, args)
    grid_path = os.path.join(args.out_dir, "summary_grid.png")
    grid.save(grid_path)
    print(f"\n[saved] {grid_path}")
    print(f"All outputs in: {args.out_dir}/")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt",       default="a photo of four computer keyboards")
    p.add_argument("--nouns",        nargs="+", default=["computer", "keyboards"],
                   help="Noun tokens to apply decay to")
    p.add_argument("--count_words",  nargs="+", default=["four"],
                   help="Count-word tokens to apply decay to")
    p.add_argument("--model_dir",    default="ckpts/Z-Image-Turbo")
    p.add_argument("--out_dir",      default="noun_injection_results")

    # Noun decay range
    p.add_argument("--noun_rs",          type=int,   default=8,
                   help="Noun decay: first layer (inclusive)")
    p.add_argument("--noun_re",          type=int,   default=13,
                   help="Noun decay: last layer (exclusive)")
    p.add_argument("--noun_decay_rates", type=float, nargs="+", default=[0.1, 0.3, 0.5],
                   help="Decay rates to sweep for noun tokens")

    # Count-word decay range
    p.add_argument("--count_rs",          type=int,   default=8,
                   help="Count decay: first layer (inclusive)")
    p.add_argument("--count_re",          type=int,   default=13,
                   help="Count decay: last layer (exclusive)")
    p.add_argument("--count_decay_rates", type=float, nargs="+", default=[0.1, 0.3, 0.5],
                   help="Decay rates to sweep for count-word tokens")

    p.add_argument("--num_seeds",    type=int, default=4)
    p.add_argument("--start_seed",   type=int, default=42)
    p.add_argument("--max_seq_len",  type=int, default=512)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
