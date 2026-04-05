"""
transplant_count_token.py
=========================
"Token Transplant" experiment:
  Take the 'four' token embedding from a WORKING prompt (e.g. "a photo of four cups")
  and inject it into a FAILING prompt (e.g. "a photo of four computer keyboards")
  at the position of the counting word, then generate images.

This isolates whether the DiT's failure is:
  (A) The counting word's contextualised embedding is wrong → transplant helps
  (B) The DiT simply lacks visual knowledge of the object → transplant doesn't help

Usage
-----
  python transplant_count_token.py \
      --donor_prompt  "a photo of four cups" \
      --target_prompt "a photo of four computer keyboards" \
      --count_word four \
      --num_seeds 4 \
      --out_dir transplant_results
"""

import argparse
import os
import sys
import re

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils import ensure_model_weights, load_from_local_dir
from zimage.pipeline import generate


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _encode_full(text, text_encoder, tokenizer, device, max_seq_len=512):
    """Return (all_hidden_states, input_ids, attention_mask, decoded_tokens)."""
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


def _generate_with_embed(embed, ids, mask, components, device, seed):
    """Monkey-patch text_encoder to inject `embed` as hidden_states[-2]."""
    text_encoder = components["text_encoder"]
    original_fwd = text_encoder.forward

    def patched(*args, **kwargs):
        class O: pass
        o = O()
        o.hidden_states = [None] * 40
        o.hidden_states[-2] = embed
        return o

    text_encoder.forward = patched
    try:
        gen = torch.Generator(device=device).manual_seed(seed)
        imgs = generate(
            transformer=components["transformer"],
            vae=components["vae"],
            text_encoder=text_encoder,
            tokenizer=components["tokenizer"],
            scheduler=components["scheduler"],
            prompt=["__patched__"],
            height=1024, width=1024,
            num_inference_steps=8,
            guidance_scale=0.0,
            generator=gen,
        )
    finally:
        text_encoder.forward = original_fwd
    return imgs[0]


# ──────────────────────────────────────────────────────────────────────────────
# Main experiment
# ──────────────────────────────────────────────────────────────────────────────

def run(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading models …")
    model_path = ensure_model_weights(args.model_dir, verify=False)
    components = load_from_local_dir(model_path, device=device, dtype=torch.bfloat16)
    text_encoder = components["text_encoder"]
    tokenizer    = components["tokenizer"]

    # ── Encode both prompts ───────────────────────────────────────────────────
    print(f"\nEncoding donor:  \"{args.donor_prompt}\"")
    d_hs, d_ids, d_mask, d_tokens = _encode_full(
        args.donor_prompt, text_encoder, tokenizer, device, args.max_seq_len)

    print(f"Encoding target: \"{args.target_prompt}\"")
    t_hs, t_ids, t_mask, t_tokens = _encode_full(
        args.target_prompt, text_encoder, tokenizer, device, args.max_seq_len)

    # ── Locate count word in both sequences ──────────────────────────────────
    d_cs, d_ce = _content_span(d_tokens)
    t_cs, t_ce = _content_span(t_tokens)

    d_tidx_rel = _find_idx(d_tokens[d_cs:d_ce], args.count_word)
    t_tidx_rel = _find_idx(t_tokens[t_cs:t_ce], args.count_word)

    if d_tidx_rel == -1:
        raise ValueError(f"'{args.count_word}' not found in donor prompt tokens: "
                         f"{d_tokens[d_cs:d_ce]}")
    if t_tidx_rel == -1:
        raise ValueError(f"'{args.count_word}' not found in target prompt tokens: "
                         f"{t_tokens[t_cs:t_ce]}")

    d_full = d_cs + d_tidx_rel
    t_full = t_cs + t_tidx_rel

    print(f"\nDonor  '{args.count_word}' token idx: {d_full}  "
          f"('{d_tokens[d_full].strip()}')")
    print(f"Target '{args.count_word}' token idx: {t_full}  "
          f"('{t_tokens[t_full].strip()}')")

    # ── Build the three embedding variants ───────────────────────────────────
    dtype = d_hs[-2].dtype

    # 1. Baseline: target prompt deep embedding (unchanged)
    embed_baseline = t_hs[-2].clone().to(dtype)

    # 2. Transplant: replace target's count token with donor's
    embed_transplant = t_hs[-2].clone().to(dtype)
    donor_token_embed = d_hs[-2][0, d_full, :].to(dtype)
    embed_transplant[0, t_full, :] = donor_token_embed

    # 3. Donor baseline: full donor prompt deep embedding
    #    (upper bound: best case if donor context helps overall)
    embed_donor_full = d_hs[-2].clone().to(dtype)

    # Cosine similarity between donor and target count token embeddings
    a = t_hs[-2][0, t_full, :].float()
    b = d_hs[-2][0, d_full, :].float()
    cos_sim = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    print(f"\nCosine sim (target vs donor '{args.count_word}' embedding): {cos_sim:.4f}")
    if cos_sim > 0.98:
        print("  ⚠  Very similar embeddings — transplant may have negligible effect.")
    elif cos_sim < 0.90:
        print("  ✓  Meaningful embedding difference — transplant is a real intervention.")

    # ── Generate images ───────────────────────────────────────────────────────
    variants = [
        ("baseline_target",    embed_baseline,    args.target_prompt),
        ("transplant",         embed_transplant,  args.target_prompt + f" ['{args.count_word}'←donor]"),
        ("baseline_donor",     embed_donor_full,  args.donor_prompt),
    ]

    all_images = {k: [] for k, _, _ in variants}

    for seed_offset in range(args.num_seeds):
        seed = args.seed + seed_offset
        print(f"\n[Seed {seed}]")
        for key, embed, label in variants:
            print(f"  generating {key} …")
            img = _generate_with_embed(
                embed, t_ids if key != "baseline_donor" else d_ids,
                t_mask if key != "baseline_donor" else d_mask,
                components, device, seed,
            )
            all_images[key].append(img)
            img.save(os.path.join(args.out_dir, f"{key}_seed{seed}.png"))

    # ── Build comparison grid ─────────────────────────────────────────────────
    n        = args.num_seeds
    img_w, img_h = all_images["baseline_target"][0].size
    label_h  = 50
    pad      = 8
    n_rows   = len(variants)
    grid_w   = n * img_w + (n + 1) * pad
    grid_h   = n_rows * (img_h + label_h) + (n_rows + 1) * pad + label_h

    grid = Image.new("RGB", (grid_w, grid_h), color=(240, 240, 240))
    draw = ImageDraw.Draw(grid)
    try:
        font_lg = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_sm = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except Exception:
        font_lg = font_sm = ImageFont.load_default()

    title = (f"Token Transplant: '{args.count_word}' embedding from  "
             f"\"{args.donor_prompt}\"  →  \"{args.target_prompt}\"")
    draw.text((pad, pad), title, fill=(20, 20, 20), font=font_lg)

    row_colors = [(200, 220, 255), (200, 255, 210), (255, 230, 200)]
    row_desc = [
        f"(A) Baseline target  — \"{args.target_prompt}\"",
        f"(B) Transplant       — target + donor's '{args.count_word}' embedding",
        f"(C) Baseline donor   — \"{args.donor_prompt}\"  (upper bound)",
    ]

    for row_idx, (key, _, _) in enumerate(variants):
        y_label = label_h + pad + row_idx * (img_h + label_h + pad)
        y_img   = y_label + label_h
        draw.rectangle([pad, y_label, grid_w - pad, y_label + label_h - 4],
                       fill=row_colors[row_idx])
        draw.text((pad * 2, y_label + 8), row_desc[row_idx],
                  fill=(20, 20, 20), font=font_sm)
        for col_idx, img in enumerate(all_images[key]):
            x = pad + col_idx * (img_w + pad)
            grid.paste(img, (x, y_img))
            draw.text((x + 6, y_img + 6),
                      f"seed={args.seed + col_idx}", fill=(255, 255, 255), font=font_sm)

    grid_path = os.path.join(args.out_dir, "transplant_comparison.png")
    grid.save(grid_path)
    print(f"\n[Saved] Comparison grid → {grid_path}")

    # ── Text report ──────────────────────────────────────────────────────────
    report = [
        "Token Transplant Experiment",
        "===========================",
        f"Donor  prompt : \"{args.donor_prompt}\"",
        f"Target prompt : \"{args.target_prompt}\"",
        f"Count word    : \"{args.count_word}\"",
        f"Cosine sim (donor vs target '{args.count_word}' embedding): {cos_sim:.4f}",
        "",
        "How to read the comparison grid:",
        "  Row A  Baseline target  — standard generation for the failing prompt.",
        "  Row B  Transplant       — same prompt but the counting token's",
        "                            embedding is replaced with the one from",
        "                            the donor (working) prompt.",
        "  Row C  Baseline donor   — standard generation for the working prompt.",
        "",
        "Interpretation:",
        "  B ≈ A  → transplant has no effect → problem is in DiT visual knowledge,",
        "           not in text encoding. Modifying text encoder won't help.",
        "  B ≈ C  → transplant fully fixes counting → problem IS in how the",
        "           counting word is contextualised by the text encoder.",
        "  A < B < C → partial improvement → both factors contribute.",
    ]
    rpt_path = os.path.join(args.out_dir, "report.txt")
    with open(rpt_path, "w") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"\n[Saved] Report → {rpt_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--donor_prompt",
                   default="a photo of four cups",
                   help="Prompt that generates CORRECTLY (source of count embedding)")
    p.add_argument("--target_prompt",
                   default="a photo of four computer keyboards",
                   help="Prompt that FAILS to count correctly (target to fix)")
    p.add_argument("--count_word", default="four")
    p.add_argument("--num_seeds", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default="transplant_results")
    p.add_argument("--model_dir", default="ckpts/Z-Image-Turbo")
    p.add_argument("--max_seq_len", type=int, default=512)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
