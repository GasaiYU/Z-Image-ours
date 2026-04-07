"""
visualize_text_encoder_attention.py
=====================================
可视化 Qwen3 (Text Encoder) 内部的 Self-Attention，揭示
"数量词特征坍缩（Count Feature Collapse）"的凶案现场。

核心 Story
----------
Qwen3 是 Causal LLM，注意力是单向的（从左到右）：
    "four" ──✗──> "keyboards"   (future token，看不到)
    "keyboards" ──✓──> "four"   (past token，能看到)

这意味着：当 Qwen3 处理 "keyboards" 时，它"选择性地"从
"four" 处吸收信息。但吸收的是"绝对数字 4"还是"模糊复数"？
这个脚本通过三组实验来揭示真相。

三组实验
--------
实验 A - 凶案现场（Full Attention Matrix）：
    对于 "a photo of four computer keyboards"，画出 Qwen3 
    每一层每一个 Head 的完整 Token×Token Causal Attention 矩阵。
    高亮 "four"（count）和 "keyboards"（noun）的行列交叉处，
    直接看到 keyboards → four 的注意力权重。

实验 B - 吸收曲线（Absorption Curve）：
    追踪 "keyboards" 对 "four" 的注意力权重，从第 1 层到最后一层的变化。
    展示"在哪一层发生了最强的特征吸收"。

实验 C - 数字坍缩曲线（Count Collapse Curve）：
    同时编码 "one/two/.../ten keyboards"，追踪各层的：
    1. "keyboards" Token 的跨数字相似度（2→10 vs 1→2）
    2. "count word" Token 的跨数字相似度
    展示 keyboards 深层特征如何从"可区分"坍缩为"不可区分"。

用法
----
    cd /path/to/Z-Image-ours
    python utils/visualize_text_encoder_attention.py \\
        --prompt "a photo of four computer keyboards" \\
        --count_token four \\
        --noun_token keyboards \\
        --outdir outputs/text_encoder_attention

    # 只跑某一个实验
    python utils/visualize_text_encoder_attention.py \\
        --experiments A B C \\
        --prompt "a photo of four computer keyboards"
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from transformers import AutoModel, AutoTokenizer

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC  = os.path.join(_ROOT, "src")
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils import ensure_model_weights

torch.set_grad_enabled(False)

NUMBERS = ["one", "two", "three", "four", "five",
           "six", "seven", "eight", "nine", "ten"]


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


def find_token_indices(valid_ids: torch.Tensor, tokenizer, word: str) -> List[int]:
    tokens  = [tokenizer.decode([t]) for t in valid_ids.tolist()]
    cs, ce  = content_span(tokens)
    w_clean = word.lower().replace(" ", "")
    hits    = []
    for i, t in enumerate(tokens[cs:ce]):
        clean = t.lower().strip().replace(" ", "").replace("▁", "").replace("Ġ", "")
        if clean == w_clean or re.search(r"\b" + re.escape(w_clean) + r"\b", clean):
            hits.append(cs + i)
    return hits


def encode_prompt(prompt: str, tokenizer, text_encoder, device: str,
                  max_len: int = 512):
    """前向传播，返回 (valid_ids, hidden_states_tuple, attentions_tuple)。"""
    messages  = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    enc = tokenizer([formatted], padding="max_length", max_length=max_len,
                    truncation=True, return_tensors="pt")
    ids  = enc.input_ids.to(device)
    mask = enc.attention_mask.to(device).bool()

    with torch.no_grad():
        out = text_encoder(
            input_ids=ids,
            attention_mask=mask,
            output_hidden_states=True,
            output_attentions=True,
        )

    valid_ids = ids[0][mask[0]]
    return valid_ids, out.hidden_states, out.attentions


# ── 实验 A：Full Attention Matrix（凶案现场）────────────────────────────────────

def experiment_A(
    valid_ids,
    attentions,          # tuple of [1, n_heads, seq, seq], len = n_layers
    tokenizer,
    count_indices: List[int],
    noun_indices:  List[int],
    count_token: str,
    noun_token:  str,
    vis_layers: List[int],
    outdir: str,
) -> None:
    """
    对每个指定层，画出：
      左图：所有头平均的完整 causal attention 矩阵（Token×Token）
      右图：noun → count 每个头的注意力权重（条形图）
    """
    os.makedirs(outdir, exist_ok=True)
    tokens = [tokenizer.decode([t]) for t in valid_ids.tolist()]
    n_tokens = len(tokens)

    # 截短 token 标签，避免太长
    labels = [t.replace("▁", "").replace("Ġ", "")[:8] for t in tokens]

    for layer_idx in vis_layers:
        if layer_idx >= len(attentions):
            continue

        attn = attentions[layer_idx][0].float().cpu()   # [n_heads, seq, seq]
        n_heads = attn.shape[0]

        avg_attn = attn.mean(dim=0).numpy()              # [seq, seq]

        # 截取有效 token 范围
        avg_attn = avg_attn[:n_tokens, :n_tokens]

        # ── 左图：完整注意力矩阵 ──────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                                 gridspec_kw={"width_ratios": [3, 1]})

        ax = axes[0]
        im = ax.imshow(avg_attn, cmap="Blues", vmin=0, aspect="auto")
        ax.set_title(f"Layer {layer_idx} | Avg Attention Matrix (all heads)\n"
                     f"Causal: token only attends to LEFT tokens",
                     fontsize=10)
        ax.set_xlabel("Key token (source)")
        ax.set_ylabel("Query token (target)")

        # Token 标签（太多时只显示内容词）
        if n_tokens <= 40:
            ax.set_xticks(range(n_tokens))
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_yticks(range(n_tokens))
            ax.set_yticklabels(labels, fontsize=6)
        else:
            # 只标注 count 和 noun 的位置
            key_pos = sorted(set(count_indices + noun_indices))
            ax.set_xticks(key_pos)
            ax.set_xticklabels([labels[p] for p in key_pos], rotation=90, fontsize=7)
            ax.set_yticks(key_pos)
            ax.set_yticklabels([labels[p] for p in key_pos], fontsize=7)

        # 高亮 count 列（红色竖线）和 noun 行（绿色横线）
        for ci in count_indices:
            if ci < n_tokens:
                ax.axvline(ci, color="red",  lw=1.5, alpha=0.9, linestyle="--")
                ax.axhline(ci, color="red",  lw=1.5, alpha=0.9, linestyle="--")
        for ni in noun_indices:
            if ni < n_tokens:
                ax.axvline(ni, color="lime", lw=1.5, alpha=0.9, linestyle="--")
                ax.axhline(ni, color="lime", lw=1.5, alpha=0.9, linestyle="--")

        ax.legend(handles=[
            plt.Line2D([0], [0], color="red",  lw=1.5, linestyle="--",
                       label=f'"{count_token}" index'),
            plt.Line2D([0], [0], color="lime", lw=1.5, linestyle="--",
                       label=f'"{noun_token}" index'),
        ], loc="upper left", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.03)

        # ── 右图：noun→count 每个头的注意力值（条形图）─────────────────────────
        ax2 = axes[1]
        # 对 count_indices 和 noun_indices 各取均值
        scores = []
        for h in range(n_heads):
            s = 0.0
            cnt = 0
            for ni in noun_indices:
                for ci in count_indices:
                    if ni < attn.shape[1] and ci < attn.shape[2]:
                        s += attn[h, ni, ci].item()
                        cnt += 1
            scores.append(s / max(cnt, 1))

        y_pos = range(n_heads)
        colors = ["#d62728" if s > np.percentile(scores, 75) else "#aec7e8"
                  for s in scores]
        ax2.barh(y_pos, scores, color=colors)
        ax2.set_yticks(range(n_heads))
        ax2.set_yticklabels([f"H{h}" for h in range(n_heads)], fontsize=6)
        ax2.set_xlabel("Attn weight")
        ax2.set_title(f'"{noun_token}" → "{count_token}"\nper head', fontsize=9)
        ax2.axvline(np.mean(scores), color="black", lw=1.2, linestyle=":",
                    label=f"mean={np.mean(scores):.4f}")
        ax2.legend(fontsize=7)

        # 标注最高分的头
        top_h = int(np.argmax(scores))
        ax2.annotate(f"★ H{top_h}\n{scores[top_h]:.4f}",
                     xy=(scores[top_h], top_h),
                     xytext=(scores[top_h] + 0.005, top_h),
                     fontsize=7, color="#d62728")

        plt.tight_layout()
        fname = f"A_layer{layer_idx:02d}_attn_matrix.png"
        plt.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [A] Saved: {fname}")


# ── 实验 B：Absorption Curve（吸收曲线）──────────────────────────────────────────

def experiment_B(
    attentions,
    valid_ids,
    tokenizer,
    count_indices: List[int],
    noun_indices:  List[int],
    count_token: str,
    noun_token:  str,
    outdir: str,
) -> None:
    """
    逐层追踪：
      - noun → count 的平均注意力权重（吸收强度）
      - 同时画出每个头的散点（展示 head 间的分工）
    """
    os.makedirs(outdir, exist_ok=True)
    n_layers = len(attentions)
    n_heads  = attentions[0].shape[1]

    avg_curve  = []   # [n_layers]  mean over heads
    head_curves = [[] for _ in range(n_heads)]   # [n_heads][n_layers]

    for layer_idx in range(n_layers):
        attn = attentions[layer_idx][0].float().cpu()   # [n_heads, seq, seq]
        for h in range(n_heads):
            s = 0.0; cnt = 0
            for ni in noun_indices:
                for ci in count_indices:
                    if ni < attn.shape[1] and ci < attn.shape[2]:
                        s += attn[h, ni, ci].item()
                        cnt += 1
            head_curves[h].append(s / max(cnt, 1))
        avg_curve.append(np.mean([head_curves[h][-1] for h in range(n_heads)]))

    layers = list(range(n_layers))

    fig, ax = plt.subplots(figsize=(12, 5))

    # 每个 head 的细曲线（半透明）
    for h in range(n_heads):
        ax.plot(layers, head_curves[h], color="steelblue", alpha=0.15, lw=0.8)

    # 均值粗曲线
    ax.plot(layers, avg_curve, color="#d62728", lw=2.5, zorder=5,
            label=f'Mean over {n_heads} heads')

    # 标出最大值层
    peak_layer = int(np.argmax(avg_curve))
    ax.axvline(peak_layer, color="#d62728", lw=1.5, linestyle=":",
               alpha=0.8, label=f"Peak at Layer {peak_layer}")
    ax.annotate(f"Peak\nL{peak_layer}\n{avg_curve[peak_layer]:.4f}",
                xy=(peak_layer, avg_curve[peak_layer]),
                xytext=(peak_layer + 1, avg_curve[peak_layer] + 0.001),
                fontsize=9, color="#d62728")

    ax.set_xlabel("Layer index", fontsize=11)
    ax.set_ylabel(f'Attention weight: "{noun_token}" → "{count_token}"', fontsize=11)
    ax.set_title(f'Absorption Curve: How much does "{noun_token}" attend to "{count_token}" across layers?\n'
                 f'(blue thin lines = individual heads, red = mean)',
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(0, n_layers - 1)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fname = "B_absorption_curve.png"
    plt.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [B] Saved: {fname}")


# ── 实验 C：Count Collapse Curve（数字坍缩曲线）──────────────────────────────────

def experiment_C(
    tokenizer,
    text_encoder,
    device: str,
    noun_template: str,   # e.g. "a photo of {num} computer keyboards"
    noun_token: str,      # e.g. "keyboards"
    max_len: int,
    outdir: str,
) -> None:
    """
    编码 "one/two/.../ten {noun_template}"，追踪每一层：
      - noun Token 的跨数字 Cosine Similarity（2→10 vs 1）
        → 展示 noun 在深层从"可区分"坍缩为"不可区分"
      - count Token 本身的跨数字相似度
        → 展示"one"和"ten"在深层也变得相似（都被同化为复数语法）
    """
    os.makedirs(outdir, exist_ok=True)

    all_hidden: Dict[str, List[torch.Tensor]] = {}   # num_word -> [layer_tensor]
    num_indices: Dict[str, int]               = {}   # num_word -> token idx in valid

    for num_word in NUMBERS:
        prompt = noun_template.replace("{num}", num_word)
        valid_ids, hidden_states, _ = encode_prompt(
            prompt, tokenizer, text_encoder, device, max_len
        )
        # 找 noun token 和 count token 的位置
        noun_idxs  = find_token_indices(valid_ids, tokenizer, noun_token)
        count_idxs = find_token_indices(valid_ids, tokenizer, num_word)

        if not noun_idxs or not count_idxs:
            print(f"  [C] Warning: token not found for '{num_word}' or '{noun_token}', skipping.")
            continue

        ni = noun_idxs[0]
        ci = count_idxs[0]

        # hidden_states: tuple of [1, seq, D]，取 valid token 位置的特征
        noun_hs  = [hs[0, ni, :].float().cpu() for hs in hidden_states]   # [n_layers+1, D]
        count_hs = [hs[0, ci, :].float().cpu() for hs in hidden_states]

        all_hidden[num_word] = {"noun": noun_hs, "count": count_hs}

    if len(all_hidden) < 2:
        print("  [C] Not enough data, skipping.")
        return

    words = [w for w in NUMBERS if w in all_hidden]
    n_layers = len(all_hidden[words[0]]["noun"])
    layers = list(range(n_layers))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, token_role in enumerate(["noun", "count"]):
        ax = axes[ax_idx]

        # 用 "one" 作为参考，计算其他词与它的逐层相似度
        ref_word = "one"
        if ref_word not in all_hidden:
            ref_word = words[0]

        cmap_colors = plt.cm.tab10(np.linspace(0, 1, len(words)))

        for wi, word in enumerate(words):
            if word == ref_word:
                continue
            sims = []
            for l in layers:
                v1 = all_hidden[ref_word][token_role][l]
                v2 = all_hidden[word][token_role][l]
                sim = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                sims.append(sim)
            ax.plot(layers, sims, color=cmap_colors[wi], lw=1.5,
                    label=f'"{ref_word}" vs "{word}"')

        title_token = noun_token if token_role == "noun" else "count word"
        ax.set_title(f'"{title_token}" Token: Cross-Number Cosine Similarity\n'
                     f'(reference = "{ref_word}", others compared to it)',
                     fontsize=10)
        ax.set_xlabel("Layer index", fontsize=10)
        ax.set_ylabel("Cosine Similarity", fontsize=10)
        ax.set_ylim(0.3, 1.05)
        ax.axhline(0.95, color="gray", lw=1.0, linestyle=":", alpha=0.7,
                   label="sim=0.95 (near-collapse)")
        ax.legend(fontsize=7, ncol=2, loc="lower right")
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
        ax.grid(axis="y", alpha=0.3)

        # 标注坍缩起始层（相似度首次超过 0.95）
        ref_sims = []
        for l in layers:
            all_sim = []
            for word in words:
                if word == ref_word:
                    continue
                v1 = all_hidden[ref_word][token_role][l]
                v2 = all_hidden[word][token_role][l]
                all_sim.append(F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item())
            ref_sims.append(np.mean(all_sim))

        collapse_layer = next((l for l, s in enumerate(ref_sims) if s > 0.95), None)
        if collapse_layer is not None:
            ax.axvline(collapse_layer, color="red", lw=2, linestyle="--", alpha=0.8)
            ax.annotate(f"Collapse\n≥ Layer {collapse_layer}",
                        xy=(collapse_layer, 0.95),
                        xytext=(collapse_layer + 1, 0.88),
                        fontsize=8, color="red",
                        arrowprops=dict(arrowstyle="->", color="red"))

    plt.suptitle(f'Count Feature Collapse in Qwen3 (Text Encoder)\n'
                 f'Template: "{noun_template.replace("{num}", "<NUM>")}"',
                 fontsize=12)
    plt.tight_layout()
    fname = "C_count_collapse_curve.png"
    plt.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [C] Saved: {fname}")


# ── 主函数 ────────────────────────────────────────────────────────────────────

def load_text_encoder(opt: argparse.Namespace, device: str):
    """
    只加载 Text Encoder 和 Tokenizer（跳过 DiT/VAE，节省内存），
    并强制使用 eager attention，以支持 output_attentions=True。
    """
    model_path = ensure_model_weights("ckpts/Z-Image-Turbo", verify=False)
    model_path = Path(model_path)

    text_encoder_dir = model_path / "text_encoder"
    tokenizer_dir    = model_path / "tokenizer"

    print(f"Loading text encoder from: {text_encoder_dir}")
    print("  (attn_implementation=eager to enable output_attentions)")

    text_encoder = AutoModel.from_pretrained(
        str(text_encoder_dir),
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="eager",   # ← 必须！sdpa 不支持 output_attentions
    ).to(device)
    text_encoder.eval()

    tokenizer_path = str(tokenizer_dir) if tokenizer_dir.exists() \
                     else str(text_encoder_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True
    )
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("  Text encoder loaded.")
    return text_encoder, tokenizer


def main(opt: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    seed_everything(opt.seed)

    text_encoder, tokenizer = load_text_encoder(opt, device)

    slug   = re.sub(r"[^a-z0-9]+", "_", opt.prompt.lower())[:50].strip("_")
    outdir = os.path.join(opt.outdir, slug)
    os.makedirs(outdir, exist_ok=True)

    experiments = set(opt.experiments)
    print(f"\nRunning experiments: {sorted(experiments)}")
    print(f"Prompt: '{opt.prompt}'")

    # ── 共用：编码 prompt，获取 hidden states 和 attentions ──────────────────
    if "A" in experiments or "B" in experiments:
        print("\nEncoding prompt (with attentions)...")
        valid_ids, hidden_states, attentions = encode_prompt(
            opt.prompt, tokenizer, text_encoder, device, opt.max_sequence_length
        )

        count_indices = find_token_indices(valid_ids, tokenizer, opt.count_token)
        noun_indices  = find_token_indices(valid_ids, tokenizer, opt.noun_token)
        decoded = [tokenizer.decode([t]) for t in valid_ids.tolist()]

        print(f"  '{opt.count_token}' → indices {count_indices} = "
              f"{[decoded[i] for i in count_indices]}")
        print(f"  '{opt.noun_token}'  → indices {noun_indices}  = "
              f"{[decoded[i] for i in noun_indices]}")
        print(f"  Total layers with attention: {len(attentions)}")

        if not count_indices or not noun_indices:
            print("ERROR: 找不到 count 或 noun token，请检查参数。")
            return

        n_layers   = len(attentions)
        vis_layers = opt.vis_layers if opt.vis_layers else list(
            range(0, n_layers, max(1, n_layers // 8))
        ) + [n_layers - 1]
        vis_layers = sorted(set(l for l in vis_layers if 0 <= l < n_layers))
        print(f"  Visualizing layers: {vis_layers}")

    # ── 实验 A ────────────────────────────────────────────────────────────────
    if "A" in experiments:
        print("\n[Experiment A] Full Attention Matrix...")
        experiment_A(
            valid_ids, attentions, tokenizer,
            count_indices, noun_indices,
            opt.count_token, opt.noun_token,
            vis_layers, outdir,
        )

    # ── 实验 B ────────────────────────────────────────────────────────────────
    if "B" in experiments:
        print("\n[Experiment B] Absorption Curve...")
        experiment_B(
            attentions, valid_ids, tokenizer,
            count_indices, noun_indices,
            opt.count_token, opt.noun_token,
            outdir,
        )

    # ── 实验 C ────────────────────────────────────────────────────────────────
    if "C" in experiments:
        print("\n[Experiment C] Count Collapse Curve...")
        # 从 prompt 自动推断 template（把数量词替换成 {num}）
        template = opt.prompt.replace(opt.count_token, "{num}")
        print(f"  Template: '{template}'")
        experiment_C(
            tokenizer, text_encoder, device,
            noun_template=template,
            noun_token=opt.noun_token,
            max_len=opt.max_sequence_length,
            outdir=outdir,
        )

    print(f"\nDone. Results saved to: {outdir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize Qwen3 Text Encoder internal attention (crime scene)."
    )
    p.add_argument("--prompt",       type=str,
                   default="a photo of four computer keyboards")
    p.add_argument("--count_token",  type=str, default="four")
    p.add_argument("--noun_token",   type=str, default="keyboards")
    p.add_argument("--outdir",       type=str,
                   default="outputs/text_encoder_attention")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--vis_layers",   type=int, nargs="+", default=None,
                   help="实验 A 显示哪些层（默认均匀采样 8 层）")
    p.add_argument("--experiments",  type=str, nargs="+", default=["A", "B", "C"],
                   choices=["A", "B", "C"],
                   help="运行哪些实验，默认全部")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
