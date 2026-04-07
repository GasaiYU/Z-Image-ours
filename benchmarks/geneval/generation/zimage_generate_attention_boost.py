"""
zimage_generate_attention_boost.py
===================================
GenEval generation with **Attention Boosting** for count-noun binding.

核心逻辑：
拦截 Qwen3 深层（如 Layer 20-35）的 Qwen2Attention.forward，
在 Softmax 之前，人为给 `[noun_index, count_index]` 的 Attention Score 加上一个极大的 Bias。
强迫深层名词 Token 重新关注数量词，解决“距离阻断”导致的特征坍缩。
"""

import argparse
import json
import os
import re
import sys
import math

import torch
import torch.nn.functional as F
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


# ── Attention Patching ────────────────────────────────────────────────────────

def _get_decoder_layers(model):
    """
    通用方法：从任意 HuggingFace 模型中找到 Transformer Layer 列表。
    兼容 Qwen3 / Qwen2 / LLaMA 等 Decoder-only 模型。
    """
    for attr_path in ["layers", "model.layers", "encoder.layers", "transformer.h"]:
        obj = model
        found = True
        for part in attr_path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                found = False
                break
        if found and obj is not None:
            return obj
    raise AttributeError(
        f"Cannot find transformer layers in {type(model).__name__}. "
        f"Tried: layers, model.layers, encoder.layers, transformer.h"
    )


def apply_attention_boost(text_encoder, count_indices, noun_indices, boost_value, start_layer):
    """
    通过修改传递给每一层的 4D Attention Mask 来注入 Boost。
    这是模型无关（Model-Agnostic）的方法：
      - 不依赖任何具体的 Attention 实现（Qwen2/Qwen3/LLaMA）
      - Transformer 内部: attn_scores = QK^T / sqrt(d) + attention_mask
      - 因此，在 attention_mask 的 [b, :, noun_idx, count_idx] 位置加上 boost_value，
        等价于在 Softmax 前直接放大该位置的 attention score。
    """
    layers = _get_decoder_layers(text_encoder)
    original_forwards = {}

    for layer_idx, layer in enumerate(layers):
        original_forwards[layer_idx] = layer.forward
        if layer_idx < start_layer:
            continue

        def make_patched(orig_fwd, _c_idxs, _n_idxs):
            def patched(hidden_states, attention_mask=None, **kwargs):
                if attention_mask is not None and attention_mask.dim() == 4:
                    # attention_mask shape: [B, 1, seq_len, seq_len]
                    # 值是加性 mask：0 表示可以 attend，-inf 表示被遮蔽
                    # 我们在 noun->count 的位置加上 boost_value，强迫注意力集中
                    bsz = attention_mask.shape[0]
                    q_len = attention_mask.shape[2]
                    boosted = attention_mask.clone()
                    for b in range(min(bsz, len(_c_idxs))):
                        for ni in _n_idxs[b]:
                            for ci in _c_idxs[b]:
                                if ni < q_len and ci < q_len and ni >= ci:
                                    boosted[b, :, ni, ci] = boosted[b, :, ni, ci] + boost_value
                    return orig_fwd(hidden_states, attention_mask=boosted, **kwargs)
                return orig_fwd(hidden_states, attention_mask=attention_mask, **kwargs)
            return patched

        layer.forward = make_patched(
            original_forwards[layer_idx],
            count_indices,
            noun_indices,
        )

    return original_forwards, layers


def remove_attention_boost(layers, original_forwards):
    for layer_idx, orig_fwd in original_forwards.items():
        layers[layer_idx].forward = orig_fwd


# ── Count Token Decay Fusion（与 noun_count_decay.py 保持一致）────────────────

def decay_weights(n, decay_rate, device, dtype):
    w = torch.exp(-decay_rate * torch.arange(n, device=device, dtype=torch.float32))
    return (w / w.sum()).to(dtype)


def build_count_decay_embeds(prompts, text_encoder, tokenizer, device,
                             count_rs, count_re, count_dr,
                             max_sequence_length=512):
    """对 count token 做 Decay Fusion，同时返回 token indices 供 Attention Boost 使用。"""
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

    hs    = out.hidden_states   # tuple of [B, S, D]
    total = len(hs)
    mixed = hs[-2].clone()

    c_rs = max(1, count_rs)
    c_re = max(c_rs + 1, min(count_re, total - 1))
    c_layers = hs[c_rs:c_re]
    c_w = decay_weights(len(c_layers), count_dr, device, mixed.dtype)

    batch_count_indices = []
    batch_noun_indices  = []

    for b in range(ids.shape[0]):
        valid_ids = ids[b][mask[b]]
        tokens    = [tokenizer.decode([t]) for t in valid_ids.tolist()]
        cs, ce    = content_span(tokens)
        content   = tokens[cs:ce]

        c_idx = find_count_indices(content)
        n_idx = find_noun_indices(content, set(c_idx))

        batch_count_indices.append([cs + i for i in c_idx])
        batch_noun_indices.append( [cs + i for i in n_idx])

        # count token: Decay Fusion
        for ci in c_idx:
            fi    = cs + ci
            fused = torch.zeros_like(mixed[b, fi, :], dtype=torch.float32)
            for i, lyr in enumerate(c_layers):
                fused += c_w[i].float() * lyr[b, fi, :].float()
            mixed[b, fi, :] = fused.to(mixed.dtype)

    print(f"  [count decay] layers [{c_rs},{c_re})  n={len(c_layers)}  "
          f"rate={count_dr}  w[0]={c_w[0]:.4f}")
    return mixed, ids, mask, batch_count_indices, batch_noun_indices


# ── Generation Wrapper ────────────────────────────────────────────────────────

def generate_with_attention_boost(components, prompts, opt, device, generator):
    text_encoder = components["text_encoder"]
    tokenizer    = components["tokenizer"]

    # Step 1: Count token Decay Fusion（提取浅层精确数字特征）
    mixed_embeds, expected_ids, _, batch_count_indices, batch_noun_indices = \
        build_count_decay_embeds(
            prompts, text_encoder, tokenizer, device,
            count_rs=opt.count_rs, count_re=opt.count_re, count_dr=opt.count_dr,
            max_sequence_length=opt.max_sequence_length,
        )

    # Step 2: 注入 count token 的 mixed_embeds（Monkey patch text_encoder.forward）
    original_te_fwd = text_encoder.forward

    def patched_te_fwd(input_ids, attention_mask, **kwargs):
        class O: pass
        o = O()
        if input_ids.shape == expected_ids.shape and torch.equal(input_ids, expected_ids):
            o.hidden_states = [None] * 40
            o.hidden_states[-2] = mixed_embeds
            return o
        return original_te_fwd(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    text_encoder.forward = patched_te_fwd

    # Step 3: Noun token Attention Boosting（强迫深层名词重新关注数量词）
    print(f"  [attn boost]  start_layer={opt.start_layer}  boost_value={opt.boost_value}")
    orig_fwds, layers = apply_attention_boost(
        text_encoder, batch_count_indices, batch_noun_indices,
        boost_value=opt.boost_value, start_layer=opt.start_layer,
    )

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
        remove_attention_boost(layers, orig_fwds)
        text_encoder.forward = original_te_fwd

    return images


# ── Main ──────────────────────────────────────────────────────────────────────

def main(opt):
    with open(opt.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    if opt.tags is not None:
        metadatas = [m for m in metadatas if m.get("tag") in opt.tags]
        print(f"Filtered to {len(metadatas)} prompts with tags: {opt.tags}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_path = ensure_model_weights("ckpts/Z-Image-Turbo", verify=False)
    components = load_from_local_dir(model_path, device=device, dtype=torch.bfloat16, compile=False)
    set_attention_backend(os.environ.get("ZIMAGE_ATTENTION", "_native_flash"))

    print(f"count decay:  layers [{opt.count_rs},{opt.count_re})  rate={opt.count_dr}")
    print(f"attn boost:   +{opt.boost_value} from layer {opt.start_layer}")

    for index, metadata in enumerate(metadatas):
        seed_everything(opt.seed)
        outpath = os.path.join(opt.outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)
        prompt = metadata["prompt"]
        print(f"[{index:>3}/{len(metadatas)}] '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        
        sample_count = 0
        with torch.no_grad():
            all_samples = []
            for _ in trange((opt.n_samples + opt.batch_size - 1) // opt.batch_size, desc="Sampling", leave=False):
                cur_bs = min(opt.batch_size, opt.n_samples - sample_count)
                prompts = [prompt] * cur_bs
                generator = torch.Generator(device).manual_seed(opt.seed + sample_count)

                images = generate_with_attention_boost(components, prompts, opt, device, generator)

                for img in images:
                    img.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                    sample_count += 1
                    if not opt.skip_grid:
                        all_samples.append(ToTensor()(img))

            if not opt.skip_grid and all_samples:
                grid = make_grid(torch.stack(all_samples), nrow=opt.batch_size)
                grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
                Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, "grid.png"))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("metadata_file", type=str)
    p.add_argument("--outdir", type=str, default="outputs/attention_boost")
    p.add_argument("--n_samples", type=int, default=4)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--H", type=int, default=1024)
    p.add_argument("--W", type=int, default=1024)
    p.add_argument("--scale", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--skip_grid", action="store_true")
    p.add_argument("--tags", type=str, nargs="+", default=None)
    p.add_argument("--max_sequence_length", type=int, default=512)

    # Count token: Decay Fusion（浅层精确数字特征）
    p.add_argument("--count_rs",  type=int,   default=8,
                   help="Count decay 起始层 (inclusive)")
    p.add_argument("--count_re",  type=int,   default=13,
                   help="Count decay 结束层 (exclusive)")
    p.add_argument("--count_dr",  type=float, default=0.3,
                   help="Count decay rate")

    # Noun token: Attention Boosting（强迫深层名词关注数量词）
    p.add_argument("--boost_value", type=float, default=10.0,
                   help="加在 Attention Score 上的 Bias 值（Softmax 之前）。"
                        "10.0 几乎能保证注意力权重接近 1.0。")
    p.add_argument("--start_layer", type=int, default=20,
                   help="从哪一层开始 Attention Boost 干预（建议从特征坍缩处开始，如 20）。")

    return p.parse_args()

if __name__ == "__main__":
    main(parse_args())
