"""
zimage_generate_noun_count_decay.py
====================================
GenEval generation with decay fusion applied to BOTH count-word tokens AND
noun tokens simultaneously.

Count tokens  → identified by QUANTITY_BANK
Noun tokens   → identified as content tokens that are NOT in STOPWORDS
                (catches nouns/adjectives, skips "a", "photo", "of", etc.)

Each group can use an independent [route_start, route_end) range and decay_rate.
"""

import argparse
import json
import os
import re
import sys

import torch
import numpy as np
from PIL import Image
from tqdm import trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from pytorch_lightning import seed_everything

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from utils import ensure_model_weights, load_from_local_dir, set_attention_backend
from zimage import generate

torch.set_grad_enabled(False)


QUANTITY_BANK = [
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
]

# Function words to skip when detecting noun tokens
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "is", "are", "and",
    "or", "with", "photo", "picture", "image", "photograph", "shot",
    "showing", "depicts", "there", "some", "many", "several",
}


# ── token detection ───────────────────────────────────────────────────────────

def content_span(tokens):
    cs, ce = 0, len(tokens)
    for i, t in enumerate(tokens):
        if "user" in t.lower():
            cs = i + 1
        elif "<|im_end|>" in t and i > cs:
            ce = i; break
    return cs, ce


def find_count_indices(content_tokens):
    """Return indices (in content) of count-word tokens."""
    hits = []
    for i, t in enumerate(content_tokens):
        clean = t.lower().strip().replace(" ", "")
        for w in QUANTITY_BANK:
            if clean == w or re.search(r"\b" + re.escape(w) + r"\b", clean):
                hits.append(i); break
    return hits


def find_noun_indices(content_tokens, count_indices_set):
    """
    Return indices (in content) of noun/adjective tokens:
    content tokens that are NOT count words and NOT stopwords.
    """
    hits = []
    for i, t in enumerate(content_tokens):
        if i in count_indices_set:
            continue
        clean = t.lower().strip().replace(" ", "").replace("▁", "")
        if clean and clean not in STOPWORDS and len(clean) > 1:
            hits.append(i)
    return hits


# ── decay fusion ──────────────────────────────────────────────────────────────

def decay_weights(n, decay_rate, device, dtype):
    w = torch.exp(-decay_rate * torch.arange(n, device=device, dtype=torch.float32))
    return (w / w.sum()).to(dtype)


def build_noun_count_decay_embeds(
    prompts, text_encoder, tokenizer, device,
    count_rs, count_re, count_dr,
    noun_rs,  noun_re,  noun_dr,
    max_sequence_length=512,
):
    messages_batch = [[{"role": "user", "content": p}] for p in prompts]
    formatted = [
        tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        for m in messages_batch
    ]

    enc = tokenizer(formatted, padding="max_length", max_length=max_sequence_length,
                    truncation=True, return_tensors="pt")
    ids  = enc.input_ids.to(device)
    mask = enc.attention_mask.to(device).bool()

    with torch.no_grad():
        out = text_encoder(input_ids=ids, attention_mask=mask,
                           output_hidden_states=True)
    hs = out.hidden_states           # tuple of [B, S, D]
    total = len(hs)

    mixed = hs[-2].clone()           # start from deep embedding

    # Pre-compute weight tensors once
    c_rs = max(1, count_rs);  c_re = max(c_rs+1, min(count_re, total-1))
    n_rs = max(1, noun_rs);   n_re = max(n_rs+1, min(noun_re,  total-1))

    c_layers = hs[c_rs:c_re];  c_n = len(c_layers)
    n_layers = hs[n_rs:n_re];  n_n = len(n_layers)

    c_w = decay_weights(c_n, count_dr, device, mixed.dtype)
    n_w = decay_weights(n_n, noun_dr,  device, mixed.dtype)

    print(f"  [count decay] layers [{c_rs},{c_re})  n={c_n}  "
          f"w[0]={c_w[0]:.4f}  w[-1]={c_w[-1]:.4f}")
    print(f"  [noun  decay] layers [{n_rs},{n_re})  n={n_n}  "
          f"w[0]={n_w[0]:.4f}  w[-1]={n_w[-1]:.4f}")

    for b in range(ids.shape[0]):
        valid_ids = ids[b][mask[b]]
        tokens    = [tokenizer.decode([t]) for t in valid_ids.tolist()]
        cs, ce    = content_span(tokens)
        content   = tokens[cs:ce]

        count_ci = find_count_indices(content)
        noun_ci  = find_noun_indices(content, set(count_ci))

        for ci in count_ci:
            fi = cs + ci
            fused = torch.zeros_like(mixed[b, fi, :], dtype=torch.float32)
            for i, lyr in enumerate(c_layers):
                fused += c_w[i].float() * lyr[b, fi, :].float()
            mixed[b, fi, :] = fused.to(mixed.dtype)

        for ni in noun_ci:
            fi = cs + ni
            fused = torch.zeros_like(mixed[b, fi, :], dtype=torch.float32)
            for i, lyr in enumerate(n_layers):
                fused += n_w[i].float() * lyr[b, fi, :].float()
            mixed[b, fi, :] = fused.to(mixed.dtype)

    return mixed, ids, mask


# ── generation wrapper ────────────────────────────────────────────────────────

def generate_with_noun_count_decay(components, prompts, opt, device, generator):
    text_encoder = components["text_encoder"]

    mixed_embeds, expected_ids, _ = build_noun_count_decay_embeds(
        prompts, text_encoder, components["tokenizer"], device,
        count_rs=opt.count_rs, count_re=opt.count_re, count_dr=opt.count_dr,
        noun_rs=opt.noun_rs,   noun_re=opt.noun_re,   noun_dr=opt.noun_dr,
        max_sequence_length=opt.max_sequence_length,
    )

    original_fwd = text_encoder.forward

    def patched_fwd(input_ids, attention_mask, **kwargs):
        class O: pass
        o = O()
        if input_ids.shape == expected_ids.shape and torch.equal(input_ids, expected_ids):
            o.hidden_states = [None] * 40
            o.hidden_states[-2] = mixed_embeds
            return o
        return original_fwd(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    text_encoder.forward = patched_fwd
    try:
        images = generate(
            prompt=prompts,
            **components,
            height=opt.H,
            width=opt.W,
            num_inference_steps=opt.steps,
            guidance_scale=opt.scale,
            generator=generator,
            max_sequence_length=opt.max_sequence_length,
        )
    finally:
        text_encoder.forward = original_fwd

    return images


# ── main ──────────────────────────────────────────────────────────────────────

def main(opt):
    with open(opt.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    if opt.tags is not None:
        metadatas = [m for m in metadatas if m.get("tag") in opt.tags]
        print(f"Filtered to {len(metadatas)} prompts with tags: {opt.tags}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model_path = ensure_model_weights("ckpts/Z-Image-Turbo", verify=False)
    components = load_from_local_dir(model_path, device=device,
                                     dtype=torch.bfloat16, compile=False)
    attn_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")
    set_attention_backend(attn_backend)

    print(f"count decay: layers [{opt.count_rs},{opt.count_re})  rate={opt.count_dr}")
    print(f"noun  decay: layers [{opt.noun_rs},{opt.noun_re})   rate={opt.noun_dr}")

    for index, metadata in enumerate(metadatas):
        seed_everything(opt.seed)

        outpath = os.path.join(opt.outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        prompt = metadata["prompt"]
        batch_size = opt.batch_size
        print(f"[{index:>3}/{len(metadatas)}] '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        sample_count = 0
        with torch.no_grad():
            all_samples = []
            for _ in trange((opt.n_samples + batch_size - 1) // batch_size,
                            desc="Sampling", leave=False):
                cur_bs   = min(batch_size, opt.n_samples - sample_count)
                prompts  = [prompt] * cur_bs
                generator = torch.Generator(device).manual_seed(opt.seed + sample_count)

                images = generate_with_noun_count_decay(
                    components, prompts, opt, device, generator)

                for img in images:
                    img.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                    sample_count += 1
                    if not opt.skip_grid:
                        all_samples.append(ToTensor()(img))

            if not opt.skip_grid and all_samples:
                grid = make_grid(torch.stack(all_samples), nrow=opt.batch_size)
                grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
                Image.fromarray(grid.astype(np.uint8)).save(
                    os.path.join(outpath, "grid.png"))

    print("Done.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("metadata_file", type=str)
    p.add_argument("--outdir",      type=str, default="outputs")
    p.add_argument("--n_samples",   type=int, default=4)
    p.add_argument("--steps",       type=int, default=8)
    p.add_argument("--H",           type=int, default=1024)
    p.add_argument("--W",           type=int, default=1024)
    p.add_argument("--scale",       type=float, default=0.0)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--batch_size",  type=int, default=1)
    p.add_argument("--skip_grid",   action="store_true")
    p.add_argument("--tags",        type=str, nargs="+", default=None)
    p.add_argument("--max_sequence_length", type=int, default=512)

    # Count-word decay
    p.add_argument("--count_rs",  type=int,   default=8,
                   help="Count decay: first layer (inclusive)")
    p.add_argument("--count_re",  type=int,   default=13,
                   help="Count decay: last layer (exclusive)")
    p.add_argument("--count_dr",  type=float, default=0.3,
                   help="Count decay rate")

    # Noun decay
    p.add_argument("--noun_rs",   type=int,   default=8,
                   help="Noun decay: first layer (inclusive)")
    p.add_argument("--noun_re",   type=int,   default=13,
                   help="Noun decay: last layer (exclusive)")
    p.add_argument("--noun_dr",   type=float, default=0.3,
                   help="Noun decay rate")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
