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
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

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

def apply_attention_boost(text_encoder, expected_ids, count_indices, noun_indices, boost_value, start_layer):
    """
    Monkey patch Qwen2Attention to boost specific attention scores.
    """
    original_forwards = {}

    for layer_idx, layer in enumerate(text_encoder.encoder.layers):
        attn_module = layer.self_attn
        original_forwards[layer_idx] = attn_module.forward

        if layer_idx < start_layer:
            continue

        def make_patched_forward(orig_fwd, l_idx):
            def patched_forward(
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=None,
                **kwargs
            ):
                # 1. 检查是否是我们关心的 input_ids（避免影响其他无关的 forward）
                # 由于在 transformers 内部 forward 时拿不到 input_ids，我们通过 seq_len 简单过滤
                # 这是一个 hack，但在单次推理中足够安全
                seq_len = hidden_states.shape[1]
                is_target = (seq_len == expected_ids.shape[1])

                if not is_target:
                    return orig_fwd(
                        hidden_states=hidden_states, attention_mask=attention_mask,
                        position_ids=position_ids, past_key_value=past_key_value,
                        output_attentions=output_attentions, use_cache=use_cache,
                        cache_position=cache_position, **kwargs
                    )

                # 2. 手动实现 Qwen2 的 Attention 计算，以便在 Softmax 前注入 Bias
                bsz, q_len, _ = hidden_states.size()
                
                # 获取 Q, K, V
                query_states = attn_module.q_proj(hidden_states)
                key_states = attn_module.k_proj(hidden_states)
                value_states = attn_module.v_proj(hidden_states)

                query_states = query_states.view(bsz, q_len, attn_module.num_heads, attn_module.head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim).transpose(1, 2)

                # RoPE
                cos, sin = attn_module.rotary_emb(value_states, position_ids)
                from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                # Repeat KV if needed (GQA)
                from transformers.models.qwen2.modeling_qwen2 import repeat_kv
                key_states = repeat_kv(key_states, attn_module.num_key_value_groups)
                value_states = repeat_kv(value_states, attn_module.num_key_value_groups)

                # QK^T
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(attn_module.head_dim)

                # Apply standard attention mask
                if attention_mask is not None:
                    attn_weights = attn_weights + attention_mask

                # ── 核心干预逻辑：Attention Boosting ──
                # 给所有 Head 的 [noun_idx, count_idx] 加上 boost_value
                for b in range(bsz):
                    for ni in noun_indices[b]:
                        for ci in count_indices[b]:
                            # 确保不会越界
                            if ni < q_len and ci < q_len:
                                # 只在因果允许的范围内加（ni >= ci）
                                if ni >= ci:
                                    attn_weights[b, :, ni, ci] += boost_value

                # Softmax
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights = F.dropout(attn_weights, p=attn_module.attention_dropout, training=attn_module.training)

                # V
                attn_output = torch.matmul(attn_weights, value_states)
                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(bsz, q_len, -1)
                attn_output = attn_module.o_proj(attn_output)

                return attn_output, None, past_key_value

            return patched_forward

        layer.self_attn.forward = make_patched_forward(original_forwards[layer_idx], layer_idx)

    return original_forwards


def remove_attention_boost(text_encoder, original_forwards):
    for layer_idx, orig_fwd in original_forwards.items():
        text_encoder.encoder.layers[layer_idx].self_attn.forward = orig_fwd


# ── Generation Wrapper ────────────────────────────────────────────────────────

def generate_with_attention_boost(components, prompts, opt, device, generator):
    text_encoder = components["text_encoder"]
    tokenizer = components["tokenizer"]

    messages_batch = [[{"role": "user", "content": p}] for p in prompts]
    formatted = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        for m in messages_batch
    ]
    enc = tokenizer(formatted, padding="max_length", max_length=opt.max_sequence_length, truncation=True, return_tensors="pt")
    ids = enc.input_ids.to(device)
    mask = enc.attention_mask.to(device).bool()

    # 找出每个 batch 的 token indices
    batch_count_indices = []
    batch_noun_indices = []
    for b in range(ids.shape[0]):
        valid_ids = ids[b][mask[b]]
        tokens = [tokenizer.decode([t]) for t in valid_ids.tolist()]
        cs, ce = content_span(tokens)
        content = tokens[cs:ce]
        c_idx = find_count_indices(content)
        n_idx = find_noun_indices(content, set(c_idx))
        batch_count_indices.append([cs + i for i in c_idx])
        batch_noun_indices.append([cs + i for i in n_idx])

    # 必须强制使用 eager 模式才能拦截 attention
    text_encoder.config._attn_implementation = "eager"

    # 应用 Monkey Patch
    orig_fwds = apply_attention_boost(
        text_encoder, ids, batch_count_indices, batch_noun_indices, 
        boost_value=opt.boost_value, start_layer=opt.start_layer
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
        remove_attention_boost(text_encoder, orig_fwds)

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

    print(f"Attention Boost: +{opt.boost_value} starting from layer {opt.start_layer}")

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

    # Attention Boost 参数
    p.add_argument("--boost_value", type=float, default=10.0, 
                   help="加在 Attention Score 上的 Bias 值（Softmax之前）。10.0 几乎能保证注意力权重接近 1.0。")
    p.add_argument("--start_layer", type=int, default=20, 
                   help="从哪一层开始干预。建议从深层（特征开始坍缩的地方）开始，比如 20。")

    return p.parse_args()

if __name__ == "__main__":
    main(parse_args())
