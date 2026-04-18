"""
generate_dpo_pairs.py
=====================
为 DPO 训练生成"同名词、不同数量"的图像对，完整三阶段流水线。

Stage 1 – 收集种子图像
  对每个名词，扫描 generated_images 下所有对应 anchor 目录，
  读取 verdict.json，选出 score 最高的单张图像作为编辑基底（seed）。
  若找不到满足 --min_seed_score 的图，则用 QwenImagePipeline 现场生成。

Stage 2 – 数量编辑
  以种子图像为基底，用 QwenImageEditPipeline 将数量逐一改为 1~max_count，
  每个目标数量可生成 --n_edits 张（不同 edit seed）。
  输出目录结构：
      {outdir}/{sanitized_noun}/{count_word}/
          seed{i}.png
          meta.json   ← 记录编辑参数、原始种子路径、使用的 prompt 等

Stage 3 – VLM 验证（可选，--skip_vlm 跳过）
  用 Qwen2-VL 对 Stage 2 生成的每张图做数量 VQA，写入 verdict.json。
  verdict.json 格式与 filter_triplet_images.py 保持一致，
  可直接被 train_counting_dpo_diffusion.py 的 _sample_loser 读取。

多卡并行（按名词列表切片）：
    bash data/gen_dpo_pairs.sh

单卡调试：
    python data/generate_dpo_pairs.py \\
        --jsonl data/train_triplets/counting_triplets_minimal_origin.jsonl \\
        --image_dir data/generated_images \\
        --outdir data/dpo_edit_images \\
        --rank 0 --world_size 1
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import torch
from pathlib import Path
from typing import Optional
from PIL import Image
from tqdm import tqdm

# 复用 filter_triplet_images 中的 VLM 加载与推理逻辑，避免重复实现
# 同在 data/ 目录，支持两种运行方式：
#   python data/generate_dpo_pairs.py  （从项目根目录）
#   python generate_dpo_pairs.py       （从 data/ 目录内）
import importlib, sys as _sys
_here = Path(__file__).resolve().parent
for _candidate in [_here, _here.parent]:
    if str(_candidate) not in _sys.path:
        _sys.path.insert(0, str(_candidate))
try:
    from filter_triplet_images import load_vlm, vlm_answer, make_vqa
except ModuleNotFoundError:
    from data.filter_triplet_images import load_vlm, vlm_answer, make_vqa


# ── 常量 ──────────────────────────────────────────────────────────────────────

INT_TO_WORD = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}
WORD_TO_INT = {v: k for k, v in INT_TO_WORD.items()}

DEFAULT_GEN_MODEL  = "Qwen/Qwen-Image-2512"
DEFAULT_EDIT_MODEL = "Qwen/Qwen-Image-Edit"
DEFAULT_VLM_MODEL  = "Qwen/Qwen3-VL-8B-Instruct"

GEN_NEG_PROMPT = (
    "low resolution, low quality, deformed limbs, deformed fingers, "
    "oversaturated, waxy skin, no facial details, over-smooth, AI-looking. "
    "messy composition. blurry or distorted text."
)

EDIT_NEG_PROMPT = " "   # QwenImageEdit 推荐空字符串


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def sanitize(text: str, maxlen: int = 80) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:maxlen]


def parse_int_from_str(s: str) -> Optional[int]:
    m = re.search(r"\b(\d+)\b", s)
    if m:
        return int(m.group(1))
    for word, val in WORD_TO_INT.items():
        if re.search(rf"\b{word}\b", s.lower()):
            return val
    return None


def make_edit_prompt(target_count: int, noun: str) -> str:
    """构造数量编辑 prompt，尽量简洁且明确。"""
    count_word = INT_TO_WORD[target_count]
    # 尝试做简单的单复数判断
    if target_count == 1:
        # 粗略单数化：去掉末尾的 s / es（若有）
        singular = re.sub(r"(?<=\w)(s|es)$", "", noun.strip(), flags=re.I).strip()
        obj_phrase = f"one {singular}"
    else:
        obj_phrase = f"{count_word} {noun}"
    return (
        f"Edit the image so that it contains exactly {target_count} {noun}. "
        f"Keep all other visual elements (background, style, lighting, colors) "
        f"identical. The final image must show exactly {obj_phrase} and nothing else changed."
    )


# ── JSONL / 名词列表加载 ──────────────────────────────────────────────────────

def load_nouns_from_txt(txt_path: str) -> list[str]:
    """从纯文本文件加载名词列表（每行一个名词）。"""
    nouns = []
    with open(txt_path) as f:
        for line in f:
            noun = line.strip().lower()
            if noun:
                nouns.append(noun)
    return nouns


def load_noun_anchors(jsonl_path: str,
                      filter_nouns: Optional[set[str]] = None) -> dict[str, list[str]]:
    """
    读取 JSONL，按 noun 分组，返回 {noun: [anchor, ...]} 映射。
    若提供 filter_nouns，则只保留其中的名词。
    """
    noun_to_anchors: dict[str, list[str]] = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("task") != "counting":
                continue
            noun   = obj.get("noun", "").strip().lower()
            anchor = obj.get("anchor", "").strip()
            if filter_nouns is not None and noun not in filter_nouns:
                continue
            if noun and anchor:
                noun_to_anchors.setdefault(noun, [])
                if anchor not in noun_to_anchors[noun]:
                    noun_to_anchors[noun].append(anchor)
    return noun_to_anchors


# ── Stage 1: 收集/生成种子图 ──────────────────────────────────────────────────

def find_best_seed(
    noun: str,
    anchors: list[str],
    image_dir: Path,
) -> Optional[tuple[Path, float, str]]:
    """
    在 image_dir/counting/{sanitized_anchor}/ 下查找 verdict.json，
    返回 (image_path, score, source_anchor)，score 最高的那张。
    以下情况均返回 None（调用方会触发现场生成）：
      - anchors 为空
      - 对应目录不存在
      - 目录存在但没有任何图像
    """
    best_path: Optional[Path] = None
    best_score = -1.0
    best_anchor = ""

    for anchor in anchors:
        anchor_dir = image_dir / "counting" / sanitize(anchor)

        # 目录不存在 → 跳过
        if not anchor_dir.exists():
            continue

        verdict_file = anchor_dir / "verdict.json"
        if not verdict_file.exists():
            # 有目录但没有 verdict：取已有图像，给默认低分 0.0
            pngs = sorted(anchor_dir.glob("seed*.png"))
            for png in pngs:
                if best_score < 0.0:   # 只在完全没有候选时才用这张
                    best_path   = png
                    best_score  = 0.0
                    best_anchor = anchor
            continue

        try:
            verdict = json.loads(verdict_file.read_text())
        except Exception:
            continue

        for result in verdict.get("results", []):
            score    = result.get("score", 0.0)
            img_name = result.get("image", "")
            img_path = anchor_dir / img_name
            if img_path.exists() and score > best_score:
                best_score  = score
                best_path   = img_path
                best_anchor = anchor

    return (best_path, best_score, best_anchor) if best_path is not None else None


def anchor_to_count(anchor: str) -> Optional[int]:
    """从 anchor 字符串（如 'five apples'）解析出数量整数。"""
    first_word = anchor.strip().lower().split()[0] if anchor.strip() else ""
    return WORD_TO_INT.get(first_word)


SEED_GEN_COUNT = 3   # generate_seed_image 固定生成的数量词对应的整数


def generate_seed_image(
    noun: str,
    gen_pipe,
    device: str,
    outdir: Path,
    opt,
) -> Optional[Path]:
    """
    用 QwenImagePipeline 临时生成一批种子图（--seed_gen_attempts 张），
    保存到 {outdir}/_seeds/{sanitized_noun}/seed{i}.png。
    返回目录路径（留给调用方做 VLM 筛选），若生成失败则返回 None。
    """
    seed_dir = outdir / "_seeds" / sanitize(noun)
    seed_dir.mkdir(parents=True, exist_ok=True)

    # 断点续传：已有足够图像则直接返回目录
    existing = sorted(seed_dir.glob("seed*.png"))
    if len(existing) >= opt.seed_gen_attempts:
        return seed_dir

    prompt = f"a photo of {INT_TO_WORD[SEED_GEN_COUNT]} {noun}"
    try:
        for i in range(len(existing), opt.seed_gen_attempts):
            gen = torch.Generator(device=device).manual_seed(opt.seed + i)
            with torch.inference_mode():
                out = gen_pipe(
                    prompt=[prompt],
                    negative_prompt=[GEN_NEG_PROMPT],
                    width=opt.gen_width,
                    height=opt.gen_height,
                    num_inference_steps=opt.gen_steps,
                    true_cfg_scale=opt.gen_cfg,
                    generator=[gen],
                )
            img = out.images[0]
            img.save(seed_dir / f"seed{i}.png")

        meta = {
            "noun": noun, "prompt": prompt,
            "seed_count": SEED_GEN_COUNT,
            "width": opt.gen_width, "height": opt.gen_height,
            "source": "generated",
        }
        (seed_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
        return seed_dir
    except Exception as e:
        print(f"\n  [GEN ERROR] noun='{noun}': {e}")
        return None


def verify_seed_dir(
    noun: str,
    seed_dir: Path,
    expected_count: int,
    vlm_model,
    vlm_processor,
    device: str,
) -> Optional[Path]:
    """
    用 VLM 对 seed_dir 下所有图像做数量 VQA，
    返回其中 score 最高且 score > 0 的图像路径，找不到则返回 None。
    同时将结果写入 seed_dir/verdict.json 供断点续传。
    """
    verdict_file = seed_dir / "verdict.json"

    # 断点续传：已有 verdict 则直接读取
    if verdict_file.exists():
        try:
            cached = json.loads(verdict_file.read_text())
            best = cached.get("best_seed")
            if best:
                p = seed_dir / best
                if p.exists():
                    return p
        except Exception:
            pass

    pngs = sorted(seed_dir.glob("seed*.png"))
    if not pngs:
        return None

    results = []
    for png in pngs:
        try:
            img = Image.open(png).convert("RGB")
            pred, score = vlm_count_score(
                vlm_model, vlm_processor, img, noun, expected_count, device
            )
            results.append({"image": png.name, "predicted": pred, "score": score})
        except Exception as e:
            results.append({"image": png.name, "error": str(e), "score": 0.0})

    best_entry = max(results, key=lambda r: r["score"])
    best_seed  = best_entry["image"] if best_entry["score"] > 0 else None

    verdict_file.write_text(json.dumps(
        {"noun": noun, "expected_count": expected_count,
         "best_seed": best_seed, "results": results},
        ensure_ascii=False, indent=2
    ))

    if best_seed:
        return seed_dir / best_seed
    return None


# ── Stage 2: 数量编辑 ─────────────────────────────────────────────────────────

def edit_count(
    seed_img: Image.Image,
    target_count: int,
    noun: str,
    edit_pipe,
    device: str,
    opt,
    edit_idx: int = 0,
) -> Optional[Image.Image]:
    """调用 QwenImageEditPipeline 改变图中的数量。"""
    prompt = make_edit_prompt(target_count, noun)
    gen = torch.Generator(device=device).manual_seed(opt.seed + edit_idx * 100 + target_count)
    try:
        torch.cuda.empty_cache()
        with torch.inference_mode():
            out = edit_pipe(
                image=seed_img,
                prompt=prompt,
                negative_prompt=EDIT_NEG_PROMPT,
                generator=gen,
                true_cfg_scale=opt.edit_cfg,
                num_inference_steps=opt.edit_steps,
            )
        return out.images[0]
    except Exception as e:
        print(f"\n  [EDIT ERROR] noun='{noun}' count={target_count} idx={edit_idx}: {e}")
        return None


def run_stage2(
    noun: str,
    seed_img_path: Path,
    seed_score: float,
    seed_count: Optional[int],   # 种子图像本身的数量，此 count 无需 edit
    edit_pipe,
    device: str,
    outdir: Path,
    opt,
):
    """对一个名词的种子图像做全量数量编辑（min_count ~ max_count）。
    seed_count 对应的 count 直接复制种子图，无需调用 edit 模型。
    """
    import shutil
    seed_img = Image.open(seed_img_path).convert("RGB")
    counts = list(range(opt.min_count, opt.max_count + 1))

    for target_count in counts:
        count_word = INT_TO_WORD.get(target_count, str(target_count))
        count_dir  = outdir / sanitize(noun) / count_word
        count_dir.mkdir(parents=True, exist_ok=True)

        # 断点续传
        existing = sorted(count_dir.glob("seed*.png"))
        if len(existing) >= opt.n_edits:
            continue

        # 种子图本身的数量：直接复制，不走 edit 模型
        if target_count == seed_count:
            dst = count_dir / "seed0.png"
            if not dst.exists():
                shutil.copy2(seed_img_path, dst)
            # meta.json 在下方统一写
        else:
            already_done = len(existing)
            for edit_idx in range(already_done, opt.n_edits):
                result_img = edit_count(
                    seed_img, target_count, noun,
                    edit_pipe, device, opt, edit_idx=edit_idx,
                )
                if result_img is None:
                    continue
                save_path = count_dir / f"seed{edit_idx}.png"
                result_img.save(save_path)

        # 写 meta.json（只写一次即可，若已存在则跳过）
        meta_file = count_dir / "meta.json"
        if not meta_file.exists():
            is_copy = (target_count == seed_count)
            meta = {
                "noun":          noun,
                "target_count":  target_count,
                "count_word":    count_word,
                "source":        "copy" if is_copy else "edit",
                "edit_prompt":   None if is_copy else make_edit_prompt(target_count, noun),
                "seed_source":   str(seed_img_path),
                "seed_score":    seed_score,
                "n_edits":       1 if is_copy else opt.n_edits,
                "edit_steps":    None if is_copy else opt.edit_steps,
                "edit_cfg":      None if is_copy else opt.edit_cfg,
            }
            meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2))


# ── Stage 3: VLM 验证 ─────────────────────────────────────────────────────────
# load_vlm / vlm_answer / make_vqa 均从 filter_triplet_images 导入，不再重复定义。

def vlm_count_score(model, processor, image: Image.Image,
                    noun: str, expected_count: int, device: str) -> tuple[str, float]:
    """
    用 filter_triplet_images 的 vlm_answer + make_vqa 做数量 VQA，
    返回 (predicted_answer, score)。
    """
    anchor = f"{INT_TO_WORD.get(expected_count, str(expected_count))} {noun}"
    question, score_fn = make_vqa("counting", anchor, INT_TO_WORD.get(expected_count, str(expected_count)))
    pred  = vlm_answer(model, processor, image, question, device)
    score = score_fn(pred)
    return pred, score


def run_stage3(noun: str, outdir: Path, opt, vlm_model, vlm_processor, device: str):
    """对 Stage 2 输出目录中所有编辑图像运行 VLM 验证，写 verdict.json。"""
    noun_dir = outdir / sanitize(noun)
    if not noun_dir.exists():
        return

    counts = list(range(opt.min_count, opt.max_count + 1))
    for target_count in counts:
        count_word = INT_TO_WORD.get(target_count, str(target_count))
        count_dir  = noun_dir / count_word
        verdict_file = count_dir / "verdict.json"

        if verdict_file.exists() and not opt.overwrite_verdict:
            continue

        pngs = sorted(count_dir.glob("seed*.png"))
        if not pngs:
            continue

        per_image = []
        for png in pngs:
            try:
                img = Image.open(png).convert("RGB")
                pred, score = vlm_count_score(
                    vlm_model, vlm_processor, img, noun, target_count, device
                )
                per_image.append({
                    "image":     png.name,
                    "predicted": pred,
                    "score":     score,
                    "pass":      score >= opt.vlm_threshold,
                })
            except Exception as e:
                print(f"\n  [VLM ERROR] {png}: {e}")
                per_image.append({"image": png.name, "error": str(e), "score": 0.0, "pass": False})

        avg_score = sum(r["score"] for r in per_image) / len(per_image) if per_image else 0.0
        verdict = {
            "noun":         noun,
            "target_count": target_count,
            "count_word":   count_word,
            "avg_score":    round(avg_score, 4),
            "threshold":    opt.vlm_threshold,
            "pass":         avg_score >= opt.vlm_threshold,
            "results":      per_image,
        }
        verdict_file.write_text(json.dumps(verdict, ensure_ascii=False, indent=2))


# ── 主流程 ────────────────────────────────────────────────────────────────────

def run(opt):
    rank       = opt.rank
    world_size = opt.world_size
    device     = "cuda:0"   # CUDA_VISIBLE_DEVICES 已由 bash 脚本设置

    image_dir = Path(opt.image_dir)
    outdir    = Path(opt.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 加载名词列表（优先用 --nouns_file，否则从 JSONL 中全量提取）
    if opt.nouns_file:
        txt_nouns    = load_nouns_from_txt(opt.nouns_file)
        filter_set   = set(txt_nouns)
        noun_anchors = load_noun_anchors(opt.jsonl, filter_nouns=filter_set)
        # 保持 txt 文件中的顺序，过滤掉 JSONL 中不存在的名词
        all_nouns = [n for n in txt_nouns if n in noun_anchors]
        missing   = [n for n in txt_nouns if n not in noun_anchors]
        if missing:
            print(f"[rank={rank}] WARNING: {len(missing)} noun(s) in nouns_file "
                  f"not found in JSONL (will still attempt edit): {missing[:10]}")
            # 对 JSONL 中找不到的名词，用空 anchor 列表（依赖现场生成）
            for n in missing:
                noun_anchors[n] = []
            all_nouns = txt_nouns  # 保留全量，Stage 1 会触发现场生成
    else:
        noun_anchors = load_noun_anchors(opt.jsonl)
        all_nouns    = sorted(noun_anchors.keys())

    shard_nouns = all_nouns[rank::world_size]
    print(f"[rank={rank}/{world_size}] Total nouns: {len(all_nouns)}, "
          f"This shard: {len(shard_nouns)}")

    # ── 加载生成模型（仅在 Stage 1 需要现场生成时才用）──
    gen_pipe  = None
    edit_pipe = None
    vlm_model = None
    vlm_proc  = None

    # 预先扫描哪些名词需要现场生成
    # noun_seeds: {noun: (path, score, seed_count)}
    nouns_need_gen = []
    noun_seeds: dict[str, tuple[Path, float, Optional[int]]] = {}
    for noun in shard_nouns:
        anchors = noun_anchors[noun]
        result  = find_best_seed(noun, anchors, image_dir)
        if result is None or result[1] < opt.min_seed_score:
            nouns_need_gen.append(noun)
        else:
            seed_path, seed_score, seed_anchor = result
            noun_seeds[noun] = (seed_path, seed_score, anchor_to_count(seed_anchor))

    print(f"[rank={rank}] Seeds found: {len(noun_seeds)}, "
          f"Need generation: {len(nouns_need_gen)}")

    # ── Stage 1a: 对需要生成的名词，加载生成模型并生成 ──
    if nouns_need_gen:
        print(f"[rank={rank}] Loading QwenImagePipeline for seed generation ...")
        from diffusers import QwenImagePipeline
        gen_pipe = QwenImagePipeline.from_pretrained(
            opt.gen_model, torch_dtype=torch.bfloat16
        ).to(device)
        try:
            gen_pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
        gen_pipe.set_progress_bar_config(disable=True)

        generated_seed_dirs: dict[str, Path] = {}
        for noun in tqdm(nouns_need_gen, desc=f"[rank={rank}] Stage1-gen", position=rank, leave=True):
            seed_dir = generate_seed_image(noun, gen_pipe, device, outdir, opt)
            if seed_dir is not None:
                generated_seed_dirs[noun] = seed_dir
            else:
                print(f"  [WARN] Generation failed for '{noun}', skipping.")

        # 释放生成模型显存
        del gen_pipe
        gen_pipe = None
        torch.cuda.empty_cache()

        # ── Stage 1b: VLM 验证现场生成的种子图 ──
        if generated_seed_dirs:
            print(f"[rank={rank}] Loading VLM to verify {len(generated_seed_dirs)} generated seeds ...")
            vlm_model, vlm_proc = load_vlm(opt.vlm_model, device)

            for noun, seed_dir in tqdm(
                generated_seed_dirs.items(),
                desc=f"[rank={rank}] Stage1b-verify",
                position=rank, leave=True,
            ):
                best = verify_seed_dir(
                    noun, seed_dir, SEED_GEN_COUNT, vlm_model, vlm_proc, device
                )
                if best is not None:
                    noun_seeds[noun] = (best, 0.0, SEED_GEN_COUNT)
                else:
                    print(f"  [WARN] No valid seed found after VLM check for '{noun}', skipping.")

            del vlm_model, vlm_proc
            vlm_model = vlm_proc = None
            torch.cuda.empty_cache()

    # ── Stage 2: 数量编辑 ──
    print(f"[rank={rank}] Loading QwenImageEditPipeline ...")
    from diffusers import QwenImageEditPipeline
    edit_pipe = QwenImageEditPipeline.from_pretrained(
        opt.edit_model, torch_dtype=torch.bfloat16
    ).to(device)
    try:
        edit_pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    edit_pipe.set_progress_bar_config(disable=True)

    for noun in tqdm(shard_nouns, desc=f"[rank={rank}] Stage2-edit", position=rank, leave=True):
        if noun not in noun_seeds:
            print(f"  [SKIP] No seed for '{noun}'")
            continue
        seed_path, seed_score, seed_count = noun_seeds[noun]
        run_stage2(noun, seed_path, seed_score, seed_count, edit_pipe, device, outdir, opt)

    # 释放 edit 模型
    del edit_pipe
    edit_pipe = None
    torch.cuda.empty_cache()

    # ── Stage 3: VLM 验证 ──
    if not opt.skip_vlm:
        print(f"[rank={rank}] Loading VLM for verification ...")
        vlm_model, vlm_proc = load_vlm(opt.vlm_model, device)

        for noun in tqdm(shard_nouns, desc=f"[rank={rank}] Stage3-vlm", position=rank, leave=True):
            run_stage3(noun, outdir, opt, vlm_model, vlm_proc, device)

    print(f"\n[rank={rank}] All done. Output: {outdir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate count-edited image pairs for DPO training"
    )

    # 输入输出
    p.add_argument("--jsonl",       type=str,
                   default="data/train_triplets/counting_triplets_minimal_origin.jsonl",
                   help="counting JSONL 文件路径（用于 noun→anchor 映射，查找已有种子图）")
    p.add_argument("--nouns_file",  type=str,
                   default="data/train_triplets/counting_nouns.txt",
                   help="名词列表 txt 文件，每行一个名词；指定后以此为准过滤 JSONL")
    p.add_argument("--image_dir",   type=str, default="data/generated_images",
                   help="generate_triplet_images.py 的输出目录（用于找种子图）")
    p.add_argument("--outdir",     type=str, default="data/dpo_edit_images",
                   help="本脚本的输出目录")

    # 模型
    p.add_argument("--gen_model",  type=str, default=DEFAULT_GEN_MODEL,
                   help="QwenImagePipeline 模型名（用于现场生成种子图）")
    p.add_argument("--edit_model", type=str, default=DEFAULT_EDIT_MODEL,
                   help="QwenImageEditPipeline 模型名")
    p.add_argument("--vlm_model",  type=str, default=DEFAULT_VLM_MODEL,
                   help="VLM 验证模型名")

    # Stage 1 参数
    p.add_argument("--min_seed_score", type=float, default=0.8,
                   help="种子图像的最低可接受分数（低于此值则现场生成）")

    # Stage 2 参数
    p.add_argument("--min_count",  type=int, default=1,   help="编辑数量范围下界（含）")
    p.add_argument("--max_count",  type=int, default=10,  help="编辑数量范围上界（含）")
    p.add_argument("--n_edits",    type=int, default=3,
                   help="每个 (名词, 数量) 组合生成几张编辑图")
    p.add_argument("--edit_steps", type=int, default=50,  help="QwenImageEdit 推理步数")
    p.add_argument("--edit_cfg",   type=float, default=4.0, help="QwenImageEdit CFG scale")

    # Stage 1 现场生成参数
    p.add_argument("--gen_steps",         type=int,   default=28)
    p.add_argument("--gen_cfg",           type=float, default=4.0)
    p.add_argument("--gen_width",         type=int,   default=1024)
    p.add_argument("--gen_height",        type=int,   default=1024)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--seed_gen_attempts", type=int,   default=4,
                   help="现场生成种子图时，每个名词生成几张备选（VLM 选最好的）")

    # Stage 3 参数
    p.add_argument("--skip_vlm",        action="store_true",
                   help="跳过 VLM 验证阶段（Stage 3）")
    p.add_argument("--vlm_threshold",   type=float, default=0.5,
                   help="VLM 平均分 >= 此值才算 pass")
    p.add_argument("--overwrite_verdict", action="store_true",
                   help="重新跑 VLM，覆盖已有 verdict.json")

    # 多卡分片
    p.add_argument("--rank",       type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)

    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
