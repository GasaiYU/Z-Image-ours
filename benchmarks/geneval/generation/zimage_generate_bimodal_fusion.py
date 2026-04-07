"""
zimage_generate_bimodal_fusion.py
==================================
GenEval generation with **Bimodal Fusion** for noun tokens.

Motivation（来自实验 A/B/C 的发现）
---------------------------------------
实验 C 的 U 型曲线告诉我们：
  - 浅层（Layer 0-4）  ：名词特征和数量词完全无关（相似度 = 1.0，差异为零）
  - 中间层（Layer ~12）：名词最大程度地吸收了数量词信息（U 型谷底）
  - 深层（Layer 35）   ：名词和修饰形容词完美融合，但数量信息被坍缩掉了

  → 所以名词理想的特征 = "中间层的数量感知" + "深层的物体语义"

策略：两个 Token 组，各用独立的融合策略
    count tokens  →  Decay Fusion（不变，从浅层提取精确数字）
    noun tokens   →  Bimodal Fusion = alpha * hidden[valley_layer]
                                    + (1 - alpha) * hidden[deep_layer]

对于 "four computer keyboards"：
    - valley_layer ≈ 12（keyboards 此时最大程度地看了 four）
    - deep_layer   ≈ 35（keyboards 此时完美融合了 computer）
    - alpha = 0.5  （可调）

用法
----
python benchmarks/geneval/generation/zimage_generate_bimodal_fusion.py \\
    benchmarks/geneval/prompts/evaluation_metadata.jsonl \\
    --outdir outputs/geneval_bimodal \\
    --count_rs 8 --count_re 13 --count_dr 0.3 \\
    --noun_valley 12 --noun_deep -2 --noun_alpha 0.5 \\
    --tags two_objects counting
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
    hits = []
    for i, t in enumerate(content_tokens):
        clean = t.lower().strip().replace(" ", "")
        for w in QUANTITY_BANK:
            if clean == w or re.search(r"\b" + re.escape(w) + r"\b", clean):
                hits.append(i); break
    return hits


def find_noun_indices(content_tokens, count_indices_set):
    hits = []
    for i, t in enumerate(content_tokens):
        if i in count_indices_set:
            continue
        clean = t.lower().strip().replace(" ", "").replace("▁", "")
        if clean and clean not in STOPWORDS and len(clean) > 1:
            hits.append(i)
    return hits


# ── decay fusion (for count tokens，不变) ─────────────────────────────────────

def decay_weights(n, decay_rate, device, dtype):
    w = torch.exp(-decay_rate * torch.arange(n, device=device, dtype=torch.float32))
    return (w / w.sum()).to(dtype)


# ── bimodal fusion (for noun tokens，核心新策略) ─────────────────────────────

def bimodal_fusion(hs_valley, hs_deep, alpha):
    """
    将 valley_layer 的特征（含数量感知）与 deep_layer 的特征（含形容词语义）
    按 alpha 比例融合。

    Args:
        hs_valley: Tensor [D]，中间层（数量吸收峰值层）的隐状态
        hs_deep:   Tensor [D]，深层（语义完备层）的隐状态
        alpha:     float，valley 层的权重（deep 层权重 = 1 - alpha）

    Returns:
        Tensor [D]，融合后的特征
    """
    return alpha * hs_valley.float() + (1.0 - alpha) * hs_deep.float()


# ── main embedding builder ────────────────────────────────────────────────────

def build_bimodal_embeds(
    prompts, text_encoder, tokenizer, device,
    # count token: decay fusion 参数
    count_rs, count_re, count_dr,
    # noun token: bimodal fusion 参数
    noun_valley_layer,   # 中间层 index（U 型谷底，数量感知最强）
    noun_deep_layer,     # 深层 index（-2 表示 last hidden，一般是倒数第二层）
    noun_alpha,          # valley 层权重
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

    hs    = out.hidden_states    # tuple of [B, S, D]，长度 = n_layers + 1
    total = len(hs)

    mixed = hs[-2].clone()       # 以深层 embedding 为基础

    # ── count token: decay fusion ──────────────────────────────────────────────
    c_rs = max(1, count_rs)
    c_re = max(c_rs + 1, min(count_re, total - 1))
    c_layers = hs[c_rs:c_re]
    c_w = decay_weights(len(c_layers), count_dr, device, mixed.dtype)

    # ── noun token: bimodal fusion ─────────────────────────────────────────────
    # 支持负数索引（如 -2 表示倒数第二层）
    valley_idx = noun_valley_layer if noun_valley_layer >= 0 \
                 else total + noun_valley_layer
    deep_idx   = noun_deep_layer   if noun_deep_layer   >= 0 \
                 else total + noun_deep_layer

    valley_idx = max(0, min(valley_idx, total - 1))
    deep_idx   = max(0, min(deep_idx,   total - 1))

    print(f"  [count  decay]   layers [{c_rs},{c_re})  n={len(c_layers)}  "
          f"rate={count_dr}  w[0]={c_w[0]:.4f}")
    print(f"  [noun   bimodal] valley=L{valley_idx}  deep=L{deep_idx}  "
          f"alpha={noun_alpha:.2f} (valley) + {1-noun_alpha:.2f} (deep)")

    for b in range(ids.shape[0]):
        valid_ids = ids[b][mask[b]]
        tokens    = [tokenizer.decode([t]) for t in valid_ids.tolist()]
        cs, ce    = content_span(tokens)
        content   = tokens[cs:ce]

        count_ci = find_count_indices(content)
        noun_ci  = find_noun_indices(content, set(count_ci))

        # count token: decay fusion（提取浅/中层的精确数字特征）
        for ci in count_ci:
            fi    = cs + ci
            fused = torch.zeros_like(mixed[b, fi, :], dtype=torch.float32)
            for i, lyr in enumerate(c_layers):
                fused += c_w[i].float() * lyr[b, fi, :].float()
            mixed[b, fi, :] = fused.to(mixed.dtype)

        # noun token: bimodal fusion（缝合"数量感知" + "形容词语义"）
        for ni in noun_ci:
            fi         = cs + ni
            h_valley   = hs[valley_idx][b, fi, :]
            h_deep     = hs[deep_idx  ][b, fi, :]
            fused      = bimodal_fusion(h_valley, h_deep, noun_alpha)
            mixed[b, fi, :] = fused.to(mixed.dtype)

    return mixed, ids, mask


# ── generation wrapper ────────────────────────────────────────────────────────

def generate_with_bimodal(components, prompts, opt, device, generator):
    text_encoder = components["text_encoder"]

    mixed_embeds, expected_ids, _ = build_bimodal_embeds(
        prompts, text_encoder, components["tokenizer"], device,
        count_rs=opt.count_rs, count_re=opt.count_re, count_dr=opt.count_dr,
        noun_valley_layer=opt.noun_valley,
        noun_deep_layer=opt.noun_deep,
        noun_alpha=opt.noun_alpha,
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

    print(f"count decay:   layers [{opt.count_rs},{opt.count_re})  rate={opt.count_dr}")
    print(f"noun bimodal:  valley=L{opt.noun_valley}  "
          f"deep=L{opt.noun_deep}  alpha={opt.noun_alpha}")

    for index, metadata in enumerate(metadatas):
        seed_everything(opt.seed)

        outpath = os.path.join(opt.outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        prompt = metadata["prompt"]
        print(f"[{index:>3}/{len(metadatas)}] '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        sample_count = 0
        with torch.no_grad():
            all_samples = []
            for _ in trange((opt.n_samples + opt.batch_size - 1) // opt.batch_size,
                            desc="Sampling", leave=False):
                cur_bs    = min(opt.batch_size, opt.n_samples - sample_count)
                prompts   = [prompt] * cur_bs
                generator = torch.Generator(device).manual_seed(opt.seed + sample_count)

                images = generate_with_bimodal(
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
    p = argparse.ArgumentParser(
        description="GenEval generation with Bimodal Fusion for noun tokens."
    )
    p.add_argument("metadata_file", type=str)
    p.add_argument("--outdir",     type=str, default="outputs/geneval_bimodal")
    p.add_argument("--n_samples",  type=int, default=4)
    p.add_argument("--steps",      type=int, default=8)
    p.add_argument("--H",          type=int, default=1024)
    p.add_argument("--W",          type=int, default=1024)
    p.add_argument("--scale",      type=float, default=0.0)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--skip_grid",  action="store_true")
    p.add_argument("--tags",       type=str, nargs="+", default=None)
    p.add_argument("--max_sequence_length", type=int, default=512)

    # Count token: Decay Fusion（与 noun_count_decay 保持一致）
    p.add_argument("--count_rs",  type=int,   default=8,
                   help="Count decay: 起始层 (inclusive)")
    p.add_argument("--count_re",  type=int,   default=13,
                   help="Count decay: 结束层 (exclusive)")
    p.add_argument("--count_dr",  type=float, default=0.3,
                   help="Count decay rate")

    # Noun token: Bimodal Fusion（新策略核心参数）
    p.add_argument("--noun_valley", type=int,   default=12,
                   help="名词的 valley 层 index（U 型谷底，数量感知最强）。"
                        "支持负数，如 -24 表示倒数第 24 层。"
                        "建议先跑实验 C 确认谷底位置，keyboards 约为 12，cups 约为 8。")
    p.add_argument("--noun_deep",   type=int,   default=-2,
                   help="名词的 deep 层 index（语义完备层，默认 -2 即倒数第二层）")
    p.add_argument("--noun_alpha",  type=float, default=0.5,
                   help="valley 层的融合权重（0~1）。"
                        "alpha=1.0 → 只用 valley 层；alpha=0.0 → 只用深层（等同无干预）。"
                        "建议从 0.3~0.6 开始调参。")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
