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
    直接 patch 每层 self_attn.forward，在 Softmax 前注入 boost_value。
    同时兼容两种 attention mask 格式：
      - 4D 加性 mask（SDPA/Eager）: [B, 1, seq, seq]，0=可attend，-inf=遮蔽
      - None（Flash Attention）: 此时手动构造 bias tensor 并加到 attn_weights
    """
    layers = _get_decoder_layers(text_encoder)
    original_forwards = {}
    boost_hit_counter = [0]   # mutable counter to verify boost is triggered

    for layer_idx, layer in enumerate(layers):
        attn_module = layer.self_attn
        original_forwards[layer_idx] = attn_module.forward
        if layer_idx < start_layer:
            continue

        def make_patched(orig_attn_fwd, _c_idxs, _n_idxs, _bv):
            def patched(hidden_states, attention_mask=None, **kwargs):
                bsz, q_len, _ = hidden_states.shape

                if attention_mask is not None and attention_mask.dim() == 4:
                    # 标准路径：直接在加性 mask 上加 bias
                    boosted = attention_mask.clone()
                    for b in range(min(bsz, len(_c_idxs))):
                        for ni in _n_idxs[b]:
                            for ci in _c_idxs[b]:
                                if ni < q_len and ci < q_len and ni >= ci:
                                    boosted[b, :, ni, ci] = boosted[b, :, ni, ci] + _bv
                                    boost_hit_counter[0] += 1
                    return orig_attn_fwd(hidden_states, attention_mask=boosted, **kwargs)

                else:
                    # Fallback：mask 为 None（Flash Attention）或 2D
                    # 构造一个显式的 4D additive bias tensor，传给 attention_mask
                    device = hidden_states.device
                    dtype  = hidden_states.dtype
                    bias   = torch.zeros(bsz, 1, q_len, q_len, device=device, dtype=dtype)
                    for b in range(min(bsz, len(_c_idxs))):
                        for ni in _n_idxs[b]:
                            for ci in _c_idxs[b]:
                                if ni < q_len and ci < q_len and ni >= ci:
                                    bias[b, :, ni, ci] += _bv
                                    boost_hit_counter[0] += 1

                    # 只有在有实际 boost 位置时才注入，否则传回原始 mask
                    if boost_hit_counter[0] > 0:
                        # 如果原始 mask 存在（2D），先展开
                        if attention_mask is not None and attention_mask.dim() == 2:
                            # 2D bool mask → 4D additive，再叠加 bias
                            am4d = (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * torch.finfo(dtype).min
                            am4d = am4d.expand(bsz, 1, q_len, q_len)
                            bias = bias + am4d
                        return orig_attn_fwd(hidden_states, attention_mask=bias, **kwargs)

                    return orig_attn_fwd(hidden_states, attention_mask=attention_mask, **kwargs)

            return patched

        attn_module.forward = make_patched(
            original_forwards[layer_idx],
            count_indices,
            noun_indices,
            boost_value,
        )

    return original_forwards, layers, boost_hit_counter


def remove_attention_boost(layers, original_forwards):
    for layer_idx, orig_fwd in original_forwards.items():
        layers[layer_idx].self_attn.forward = orig_fwd


# ── Count Token Decay Fusion（与 noun_count_decay.py 保持一致）────────────────

def decay_weights(n, decay_rate, device, dtype):
    w = torch.exp(-decay_rate * torch.arange(n, device=device, dtype=torch.float32))
    return (w / w.sum()).to(dtype)


# ── Generation Wrapper ────────────────────────────────────────────────────────

def generate_with_attention_boost(components, prompts, opt, device, generator):
    text_encoder = components["text_encoder"]
    tokenizer    = components["tokenizer"]

    # 1. 准备 Token 和 Indices
    messages_batch = [[{"role": "user", "content": p}] for p in prompts]
    formatted = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        for m in messages_batch
    ]
    enc = tokenizer(formatted, padding="max_length", max_length=opt.max_sequence_length, truncation=True, return_tensors="pt")
    ids  = enc.input_ids.to(device)
    mask = enc.attention_mask.to(device).bool()

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

    # 2. 应用 Attention Boost（patch self_attn.forward，兼容 SDPA / Flash Attn / Eager）
    print(f"  [attn boost]  start_layer={opt.start_layer}  boost_value={opt.boost_value}")
    orig_fwds, layers, boost_counter = apply_attention_boost(
        text_encoder, batch_count_indices, batch_noun_indices,
        boost_value=opt.boost_value, start_layer=opt.start_layer,
    )

    # 3. 前向传播：此时深层名词会因为 Boost 强行关注数量词！
    with torch.no_grad():
        out = text_encoder(input_ids=ids, attention_mask=mask, output_hidden_states=True)

    # 4. 验证 Boost 是否真的被触发了（如果是 0 说明 mask 路径有问题）
    print(f"  [attn boost]  total boost injections this forward: {boost_counter[0]}")
    if boost_counter[0] == 0:
        print("  [WARNING] Attention Boost was NOT applied! Check attention_mask format.")

    # 5. 移除 Attention Boost (保持环境干净)
    remove_attention_boost(layers, orig_fwds)

    # 5. 提取特征并应用 Count Decay Fusion
    hs    = out.hidden_states
    total = len(hs)
    mixed = hs[-2].clone()

    c_rs = max(1, opt.count_rs)
    c_re = max(c_rs + 1, min(opt.count_re, total - 1))
    c_layers = hs[c_rs:c_re]
    c_w = decay_weights(len(c_layers), opt.count_dr, device, mixed.dtype)

    for b in range(ids.shape[0]):
        for ci in batch_count_indices[b]:
            fused = torch.zeros_like(mixed[b, ci, :], dtype=torch.float32)
            for i, lyr in enumerate(c_layers):
                fused += c_w[i].float() * lyr[b, ci, :].float()
            mixed[b, ci, :] = fused.to(mixed.dtype)

    print(f"  [count decay] layers [{c_rs},{c_re})  n={len(c_layers)}  rate={opt.count_dr}")

    # 6. 拦截 Text Encoder 的 forward，直接返回我们精心调配的 mixed_embeds 给 DiT
    original_te_fwd = text_encoder.forward

    def patched_te_fwd(input_ids, attention_mask, **kwargs):
        class O: pass
        o = O()
        if input_ids.shape == ids.shape and torch.equal(input_ids, ids):
            o.hidden_states = [None] * 40
            o.hidden_states[-2] = mixed
            return o
        return original_te_fwd(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    text_encoder.forward = patched_te_fwd

    # 7. 运行 DiT 生成
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
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

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
    p.add_argument("--boost_value", type=float, default=50.0,
                   help="加在 Attention Score 上的 Bias 值（Softmax 之前）。"
                        "10.0 几乎能保证注意力权重接近 1.0。")
    p.add_argument("--start_layer", type=int, default=12,
                   help="从哪一层开始 Attention Boost 干预（建议从特征坍缩处开始，如 20）。")

    return p.parse_args()

    print("Done.")

if __name__ == "__main__":
    main(parse_args())
