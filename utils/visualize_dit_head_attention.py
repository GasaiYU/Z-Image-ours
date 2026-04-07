"""
visualize_dit_head_attention.py
================================
分析 DiT 各注意力头（Attention Head）是否有专门负责
"数量词-名词绑定（Count-Noun Binding）"的特化头。

核心思路
--------
DiT 的 Full Self-Attention 矩阵可以切分为 4 个象限：
    ┌─────────────────┬─────────────────┐
    │  Image→Image    │  Image→Text     │
    ├─────────────────┼─────────────────┤
    │  Text→Image     │  Text→Text      │
    └─────────────────┴─────────────────┘

"Count-Noun Binding" 的核心信号在于 Text→Text 象限：
    attn[count_token_idx → noun_token_idx]
这个权重越高，说明这个 Head 越"关心"数量词和名词的绑定关系。

输出
----
outputs/dit_head_attention/<prompt>/
    binding_score_heatmap.png        - 每个 (layer, head) 的绑定分数热力图
    binding_score_avg_timestep.png   - 跨 timestep 平均的绑定分数
    top_heads_spatial_<word>.png     - 绑定分数最高的 Top-K Head 的空间注意力图
    text_attn_layer<L>_head<H>.png   - 某个头的完整文本内部注意力矩阵

用法
----
    cd /path/to/Z-Image-ours
    python utils/visualize_dit_head_attention.py \\
        --prompt "a photo of four computer keyboards" \\
        --count_token four \\
        --noun_token keyboards \\
        --vis_layers 0 5 10 15 20 25 29 \\
        --vis_timestep_indices 0 2 5 7 \\
        --top_k_heads 6 \\
        --outdir outputs/dit_head_attention
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
import matplotlib.gridspec as gridspec
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
    cs, ce = 0, len(tokens)
    for i, t in enumerate(tokens):
        if "user" in t.lower():
            cs = i + 1
        elif "<|im_end|>" in t and i > cs:
            ce = i
            break
    return cs, ce


def find_token_index(valid_ids: torch.Tensor, tokenizer, word: str) -> List[int]:
    """返回 word 在 cap_feats 序列中的绝对 token 索引（支持子词分片）。"""
    tokens  = [tokenizer.decode([t]) for t in valid_ids.tolist()]
    cs, ce  = content_span(tokens)
    w_clean = word.lower().replace(" ", "")
    hits    = []
    for i, t in enumerate(tokens[cs:ce]):
        clean = t.lower().strip().replace(" ", "").replace("▁", "").replace("Ġ", "")
        if clean == w_clean or re.search(r"\b" + re.escape(w_clean) + r"\b", clean):
            hits.append(cs + i)
    return hits


# ── per-head capture ───────────────────────────────────────────────────────────

class HeadCapture:
    """
    存储每个 (timestep, layer, head) 的注意力信息。

    binding_scores[t][l][h]  = scalar float
        = mean attn[count→noun] + mean attn[noun→count]   (Text→Text 象限)

    img_text_maps[t][l][h]   = FloatTensor [x_len, cap_len]
        = 每个图像 Patch 对每个文本 Token 的注意力（该 head）

    text_text_maps[t][l][h]  = FloatTensor [cap_len, cap_len]
        = 文本内部注意力矩阵（该 head），用于可视化 count→noun 权重
    """

    def __init__(self, capture_layers: List[int], capture_timestep_indices: Set[int]):
        self.capture_layers              = set(capture_layers)
        self.capture_timestep_indices    = set(capture_timestep_indices)
        self.timestep_idx: int           = 0
        self.x_lens: List[int]           = []
        self.cap_lens: List[int]         = []

        # count/noun token indices (cap_feats 中的绝对位置)
        self.count_indices: List[int]    = []
        self.noun_indices:  List[int]    = []

        # storage
        self.binding_scores: Dict[int, Dict[int, Dict[int, float]]]               = {}
        self.img_text_maps:  Dict[int, Dict[int, Dict[int, torch.Tensor]]]        = {}
        self.text_text_maps: Dict[int, Dict[int, Dict[int, torch.Tensor]]]        = {}

    def should_capture(self, layer_idx: int) -> bool:
        return (
            self.timestep_idx in self.capture_timestep_indices
            and layer_idx in self.capture_layers
            and bool(self.x_lens)
        )

    def store(
        self,
        layer_idx: int,
        binding_score_per_head: List[float],     # [n_heads]
        img_text_per_head: torch.Tensor,          # [n_heads, x_len, cap_len]
        text_text_per_head: torch.Tensor,         # [n_heads, cap_len, cap_len]
    ) -> None:
        t = self.timestep_idx
        for d in (self.binding_scores, self.img_text_maps, self.text_text_maps):
            if t not in d:
                d[t] = {}
            if layer_idx not in d[t]:
                d[t][layer_idx] = {}

        for h, score in enumerate(binding_score_per_head):
            self.binding_scores[t][layer_idx][h]  = score
            self.img_text_maps [t][layer_idx][h]  = img_text_per_head[h].cpu()
            self.text_text_maps[t][layer_idx][h]  = text_text_per_head[h].cpu()


def _compute_per_head_attn(
    query: torch.Tensor,    # [B, H, S, D_h]  after RoPE, float32
    key:   torch.Tensor,    # [B, H, S, D_h]
    x_len: int,
    cap_len: int,
    count_indices: List[int],
    noun_indices:  List[int],
) -> Tuple[List[float], torch.Tensor, torch.Tensor]:
    """
    返回：
        binding_scores   [H]            float list
        img_text_maps    [H, x_len, cap_len]   cpu float32
        text_text_maps   [H, cap_len, cap_len] cpu float32
    """
    scale   = 1.0 / math.sqrt(query.shape[-1])
    H       = query.shape[1]
    S_all   = x_len + cap_len

    # 只取 batch item 0
    q = query[0]   # [H, S, D_h]
    k = key[0]     # [H, S, D_h]

    # ── 文本内部注意力（Text→Text）────────────────────────────────────────────
    q_text = q[:, x_len: x_len + cap_len, :]   # [H, cap_len, D_h]
    k_text = k[:, x_len: x_len + cap_len, :]   # [H, cap_len, D_h]

    # 完整 row-softmax：每个文本 token 的 query 对全序列 key 做 softmax
    # 行 = 文本 Query，列 = 文本 Key（只截文本部分列）
    # 注意：为了节省内存，这里只计算文本 query 对文本 key 的局部 softmax
    # （全序列 softmax 精确但过于耗内存；对 binding score 分析，局部 softmax 已足够）
    logits_tt  = torch.matmul(q_text, k_text.transpose(-2, -1)) * scale   # [H, cap, cap]
    attn_tt    = torch.softmax(logits_tt.float(), dim=-1)                  # [H, cap, cap]

    # Binding Score：count→noun + noun→count 的平均注意力权重
    binding_scores: List[float] = []
    for h in range(H):
        score = 0.0
        cnt   = 0
        # count → noun
        for ci in count_indices:
            for ni in noun_indices:
                if ci < attn_tt.shape[1] and ni < attn_tt.shape[2]:
                    score += attn_tt[h, ci, ni].item()
                    cnt   += 1
        # noun → count
        for ni in noun_indices:
            for ci in count_indices:
                if ni < attn_tt.shape[1] and ci < attn_tt.shape[2]:
                    score += attn_tt[h, ni, ci].item()
                    cnt   += 1
        binding_scores.append(score / max(cnt, 1))

    # ── 图像→文本注意力（Image→Text），分块计算 ────────────────────────────────
    q_img   = q[:, :x_len, :]                     # [H, x_len, D_h]
    k_all   = k[:, :S_all, :]                     # [H, S_all, D_h]

    chunk   = 512
    chunks  = []
    for ci in range(math.ceil(x_len / chunk)):
        s, e     = ci * chunk, min((ci + 1) * chunk, x_len)
        logits   = torch.matmul(q_img[:, s:e, :], k_all.transpose(-2, -1)) * scale
        attn_w   = torch.softmax(logits.float(), dim=-1)
        chunks.append(attn_w[:, :, x_len: x_len + cap_len].cpu())   # [H, chunk, cap]

    img_text = torch.cat(chunks, dim=1)            # [H, x_len, cap_len]

    return binding_scores, img_text, attn_tt.cpu()


# ── hooks ──────────────────────────────────────────────────────────────────────

def install_hooks(transformer, capture: HeadCapture):
    original_pae = transformer.patchify_and_embed

    def patched_pae(all_image, all_cap_feats, patch_size, f_patch_size):
        result = original_pae(all_image, all_cap_feats, patch_size, f_patch_size)
        capture.x_lens  = [len(v) for v in result[0]]
        capture.cap_lens = [len(v) for v in result[1]]
        return result

    transformer.patchify_and_embed = patched_pae

    patched_attns: Dict[int, Tuple] = {}

    for layer_idx, layer in enumerate(transformer.layers):
        if layer_idx not in capture.capture_layers:
            continue

        attn = layer.attention
        orig_fwd = attn.forward

        def make_fwd(lid: int, orig, mod: ZImageAttention):
            def patched_fwd(hidden_states, attention_mask=None, freqs_cis=None):
                output = orig(hidden_states, attention_mask, freqs_cis)

                if capture.should_capture(lid):
                    with torch.no_grad():
                        wdtype = mod.to_q.weight.dtype
                        hs = hidden_states.to(wdtype)
                        q  = mod.to_q(hs).float()
                        k  = mod.to_k(hs).float()

                        q = q.unflatten(-1, (mod.n_heads,    mod.head_dim))
                        k = k.unflatten(-1, (mod.n_kv_heads, mod.head_dim))

                        if mod.norm_q is not None: q = mod.norm_q(q)
                        if mod.norm_k is not None: k = mod.norm_k(k)

                        if freqs_cis is not None:
                            q = apply_rotary_emb(q, freqs_cis)
                            k = apply_rotary_emb(k, freqs_cis)

                        q = q.permute(0, 2, 1, 3)   # [B, H, S, D_h]
                        k = k.permute(0, 2, 1, 3)

                        x_len   = capture.x_lens[0]
                        cap_len = capture.cap_lens[0]

                        scores, img_text, text_text = _compute_per_head_attn(
                            q, k, x_len, cap_len,
                            capture.count_indices,
                            capture.noun_indices,
                        )
                        capture.store(lid, scores, img_text, text_text)

                return output
            return patched_fwd

        patched_attns[layer_idx] = (attn, orig_fwd)
        attn.forward = make_fwd(layer_idx, orig_fwd, attn)

    return patched_attns, original_pae


def remove_hooks(transformer, patched_attns, original_pae):
    transformer.patchify_and_embed = original_pae
    for _, (attn, orig) in patched_attns.items():
        attn.forward = orig


# ── visualisation ──────────────────────────────────────────────────────────────

def _binding_score_array(capture: HeadCapture) -> np.ndarray:
    """
    返回 binding score 数组，形状 [n_timesteps, n_layers, n_heads]。
    """
    t_idxs     = sorted(capture.binding_scores.keys())
    layer_idxs = sorted({l for t in capture.binding_scores.values() for l in t.keys()})
    # n_heads 从第一个非空 entry 获取
    n_heads = max(max(h + 1 for h in lv.keys())
                  for tv in capture.binding_scores.values()
                  for lv in tv.values())

    arr = np.zeros((len(t_idxs), len(layer_idxs), n_heads), dtype=np.float32)
    for ti, t in enumerate(t_idxs):
        for li, l in enumerate(layer_idxs):
            for h, score in capture.binding_scores[t][l].items():
                arr[ti, li, h] = score

    return arr, t_idxs, layer_idxs


def visualize(
    capture:   HeadCapture,
    img_h: int,
    img_w: int,
    patch_size: int,
    vae_scale: int,
    count_token: str,
    noun_token:  str,
    top_k: int,
    outdir: str,
    gen_image: Optional[Image.Image] = None,
) -> None:
    os.makedirs(outdir, exist_ok=True)
    if gen_image is not None:
        gen_image.save(os.path.join(outdir, "generated.png"))

    grid_h = (img_h // vae_scale) // patch_size
    grid_w = (img_w // vae_scale) // patch_size
    print(f"  Patch grid: {grid_h}×{grid_w}")

    arr, t_idxs, layer_idxs = _binding_score_array(capture)
    # arr: [T, L, H]
    n_heads = arr.shape[2]

    # ── 1. 每个 timestep 的绑定分数热力图 (layers × heads) ────────────────────
    for ti, t_idx in enumerate(t_idxs):
        fig, ax = plt.subplots(figsize=(max(n_heads * 0.5, 8), max(len(layer_idxs) * 0.5, 5)))
        im = ax.imshow(arr[ti], aspect="auto", cmap="hot",
                       vmin=arr.min(), vmax=arr.max())
        ax.set_xlabel("Head index")
        ax.set_ylabel("Layer index")
        ax.set_xticks(range(n_heads))
        ax.set_yticks(range(len(layer_idxs)))
        ax.set_yticklabels(layer_idxs)
        ax.set_title(f'Binding Score ("{count_token}"→"{noun_token}") | Step {t_idx}',
                     fontsize=11)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"binding_score_step{t_idx:02d}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: binding_score_step{t_idx:02d}.png")

    # ── 2. 跨 timestep 平均的绑定分数 ─────────────────────────────────────────
    avg_arr = arr.mean(axis=0)   # [L, H]
    fig, ax = plt.subplots(figsize=(max(n_heads * 0.5, 8), max(len(layer_idxs) * 0.5, 5)))
    im = ax.imshow(avg_arr, aspect="auto", cmap="hot")
    ax.set_xlabel("Head index")
    ax.set_ylabel("Layer index")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(len(layer_idxs)))
    ax.set_yticklabels(layer_idxs)
    ax.set_title(f'Avg Binding Score (all timesteps)\n"{count_token}" ↔ "{noun_token}"',
                 fontsize=11)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "binding_score_avg.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: binding_score_avg.png")

    # ── 3. Top-K 绑定头的空间注意力图 ─────────────────────────────────────────
    # 把全局平均分数展平，找 top-k (layer, head) 组合
    flat = avg_arr.flatten()
    top_k_actual = min(top_k, len(flat))
    top_flat_idx = np.argsort(flat)[::-1][:top_k_actual]

    print(f"\n  Top-{top_k_actual} binding heads:")
    top_entries = []   # [(layer_actual, head, score)]
    for fi in top_flat_idx:
        li, h  = divmod(int(fi), n_heads)
        layer  = layer_idxs[li]
        score  = avg_arr[li, h]
        top_entries.append((layer, h, score))
        print(f"    Layer {layer:>2d}  Head {h:>2d}  score={score:.5f}")

    # 对每个 target word（count 和 noun），画 top-k 头的空间热力图
    for word_label, word_indices in [
        (count_token, capture.count_indices),
        (noun_token,  capture.noun_indices),
    ]:
        if not word_indices:
            print(f"  Warning: no token for '{word_label}', skipping spatial vis.")
            continue

        ncols = top_k_actual
        nrows = len(t_idxs)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(2.5 * ncols, 2.5 * nrows + 0.8),
            squeeze=False,
        )
        fig.suptitle(
            f'Top-{top_k_actual} Binding Heads | Spatial Attention to "{word_label}"\n'
            f'(rows=timestep, cols=head)',
            fontsize=11,
        )

        for ri, t_idx in enumerate(t_idxs):
            for ci, (layer, h, score) in enumerate(top_entries):
                ax  = axes[ri][ci]
                img_text = capture.img_text_maps.get(t_idx, {}).get(layer, {}).get(h)
                if img_text is None:
                    ax.set_visible(False)
                    continue

                # Average over all sub-token pieces of the word
                cols = [img_text[:, wi] for wi in word_indices
                        if wi < img_text.shape[1]]
                if not cols:
                    ax.set_visible(False)
                    continue

                attn_1d = torch.stack(cols).mean(0).numpy()[:grid_h * grid_w]
                spatial = attn_1d.reshape(grid_h, grid_w)
                lo, hi  = spatial.min(), spatial.max()
                spatial = (spatial - lo) / (hi - lo + 1e-8)

                ax.imshow(spatial, cmap="hot", interpolation="bilinear",
                          vmin=0, vmax=1)
                if ri == 0:
                    ax.set_title(f"L{layer}/H{h}\n{score:.4f}", fontsize=7)
                if ci == 0:
                    ax.set_ylabel(f"t={t_idx}", fontsize=7)
                ax.set_xticks([]); ax.set_yticks([])

        plt.tight_layout()
        fname = f"top_heads_spatial_{word_label.replace(' ', '_')}.png"
        plt.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {fname}")

    # ── 4. Top-1 头的完整 Text→Text 注意力矩阵 ────────────────────────────────
    top_layer, top_h, _ = top_entries[0]
    # 用所有 timestep 平均
    tt_maps = []
    for t_idx in t_idxs:
        m = capture.text_text_maps.get(t_idx, {}).get(top_layer, {}).get(top_h)
        if m is not None:
            tt_maps.append(m)
    if tt_maps:
        avg_tt = torch.stack(tt_maps).mean(0).numpy()   # [cap_len, cap_len]
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(avg_tt, cmap="Blues", vmin=0)
        ax.set_title(f"Text→Text Attn (avg timesteps)\nTop Binding Head: Layer {top_layer}, Head {top_h}",
                     fontsize=10)
        ax.set_xlabel("Key token index (cap_feats)")
        ax.set_ylabel("Query token index (cap_feats)")
        # 标记 count 和 noun 索引
        for ci in capture.count_indices:
            ax.axvline(ci, color="red",  lw=1.2, alpha=0.8, linestyle="--")
            ax.axhline(ci, color="red",  lw=1.2, alpha=0.8, linestyle="--")
        for ni in capture.noun_indices:
            ax.axvline(ni, color="lime", lw=1.2, alpha=0.8, linestyle="--")
            ax.axhline(ni, color="lime", lw=1.2, alpha=0.8, linestyle="--")
        ax.legend(handles=[
            plt.Line2D([0], [0], color="red",  lw=1.5, label=f'"{count_token}"'),
            plt.Line2D([0], [0], color="lime", lw=1.5, label=f'"{noun_token}"'),
        ], loc="upper right", fontsize=8)
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "top_head_text_attn.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved: top_head_text_attn.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main(opt: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    seed_everything(opt.seed)

    model_path = ensure_model_weights("ckpts/Z-Image-Turbo", verify=False)
    components  = load_from_local_dir(model_path, device=device,
                                      dtype=torch.bfloat16, compile=False)
    set_attention_backend(os.environ.get("ZIMAGE_ATTENTION", "_native_flash"))

    tokenizer   = components["tokenizer"]
    transformer = components["transformer"]

    # ── 找 count/noun token 在 cap_feats 中的位置 ─────────────────────────────
    messages  = [{"role": "user", "content": opt.prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    enc = tokenizer([formatted], padding="max_length",
                    max_length=opt.max_sequence_length,
                    truncation=True, return_tensors="pt")
    valid_ids = enc.input_ids[0][enc.attention_mask[0].bool()]

    count_indices = find_token_index(valid_ids, tokenizer, opt.count_token)
    noun_indices  = find_token_index(valid_ids, tokenizer, opt.noun_token)

    decoded = [tokenizer.decode([t]) for t in valid_ids.tolist()]
    print(f"\nToken positions:")
    print(f"  '{opt.count_token}' → {count_indices} = {[decoded[i] for i in count_indices]}")
    print(f"  '{opt.noun_token}'  → {noun_indices}  = {[decoded[i] for i in noun_indices]}")

    if not count_indices or not noun_indices:
        print("ERROR: 找不到 count 或 noun token，请检查 --count_token / --noun_token 参数。")
        return

    # ── 配置 capture ────────────────────────────────────────────────────────────
    n_layers   = len(transformer.layers)
    vis_layers = opt.vis_layers if opt.vis_layers else [
        l for l in [0, 5, 10, 15, 20, 25, n_layers - 1] if l < n_layers
    ]
    vis_layers = sorted(set(l for l in vis_layers if 0 <= l < n_layers))
    vis_steps  = set(opt.vis_timestep_indices)

    print(f"\nCapturing layers: {vis_layers}")
    print(f"Capturing steps:  {sorted(vis_steps)}")

    capture = HeadCapture(vis_layers, vis_steps)
    capture.count_indices = count_indices
    capture.noun_indices  = noun_indices

    # timestep 计数
    original_fwd = transformer.forward
    call_count   = [0]

    def counting_fwd(x, t, cap_feats, patch_size=2, f_patch_size=1):
        capture.timestep_idx = call_count[0]
        call_count[0] += 1
        return original_fwd(x, t, cap_feats, patch_size, f_patch_size)

    transformer.forward = counting_fwd
    patched_attns, original_pae = install_hooks(transformer, capture)

    print(f"\nGenerating: '{opt.prompt}'")
    try:
        generator = torch.Generator(device).manual_seed(opt.seed)
        images = generate(
            prompt=[opt.prompt], **components,
            height=opt.H, width=opt.W,
            num_inference_steps=opt.steps,
            guidance_scale=opt.scale,
            generator=generator,
            max_sequence_length=opt.max_sequence_length,
        )
        gen_image = images[0]
    finally:
        remove_hooks(transformer, patched_attns, original_pae)
        transformer.forward = original_fwd

    # ── 可视化 ─────────────────────────────────────────────────────────────────
    slug   = re.sub(r"[^a-z0-9]+", "_", opt.prompt.lower())[:50].strip("_")
    outdir = os.path.join(opt.outdir, slug)

    visualize(
        capture,
        img_h=opt.H, img_w=opt.W,
        patch_size=2, vae_scale=16,
        count_token=opt.count_token,
        noun_token=opt.noun_token,
        top_k=opt.top_k_heads,
        outdir=outdir,
        gen_image=gen_image,
    )
    print(f"\nDone. Results saved to: {outdir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize DiT per-head attention & binding scores.")
    p.add_argument("--prompt",       type=str, default="a photo of four computer keyboards")
    p.add_argument("--count_token",  type=str, default="four",
                   help="数量词（Count token），用于计算绑定分数")
    p.add_argument("--noun_token",   type=str, default="keyboards",
                   help="名词（Noun token），用于计算绑定分数")
    p.add_argument("--outdir",       type=str, default="outputs/dit_head_attention")
    p.add_argument("--steps",        type=int, default=8)
    p.add_argument("--H",            type=int, default=1024)
    p.add_argument("--W",            type=int, default=1024)
    p.add_argument("--scale",        type=float, default=0.0)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--vis_layers",   type=int, nargs="+", default=None,
                   help="要捕获的层索引，默认: 0 5 10 15 20 25 29")
    p.add_argument("--vis_timestep_indices", type=int, nargs="+", default=[0, 2, 5, 7],
                   help="要捕获的去噪步索引（0=第一步）")
    p.add_argument("--top_k_heads",  type=int, default=6,
                   help="展示绑定分数最高的 Top-K 个 (layer, head) 组合")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
