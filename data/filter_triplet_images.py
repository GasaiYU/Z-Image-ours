"""
filter_triplet_images.py
========================
用本地 VLM（默认 Qwen2-VL-7B-Instruct）对 generate_triplet_images.py 生成的
图像进行 VQA 质量筛选，为每张图打 0~1 对齐分，写入 verdict.json，
最后汇总为 filtered_rank{rank}.jsonl。

输出目录结构（在已有 generated_images 基础上追加）：
    data/generated_images/
        <task>/
            <sanitized_anchor>/
                seed*.png
                meta.json
                verdict.json       ← 本脚本写入

筛选逻辑（参考 EvoGen pipeline.py）：
    counting   : VLM 数数，exact→1.0，差1→0.5，否则0.0
    color      : VLM 与 target_word 精确匹配→1.0
    non-spatial: VLM 判断 yes/no，yes→1.0
    scene      : VLM 判断 yes/no（关键词在场景里），yes→1.0

单卡调试：
    python data/filter_triplet_images.py

多卡并行：
    bash data/filter_image.sh
"""

import argparse
import json
import re
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# ── 数字词映射 ────────────────────────────────────────────────────────────────

WORD_TO_INT = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12,
}

def _parse_int(s: str):
    """从字符串中提取整数，优先识别阿拉伯数字，其次识别英文数字词。"""
    m = re.search(r"\b(\d+)\b", s)
    if m:
        return int(m.group(1))
    for word, val in WORD_TO_INT.items():
        if re.search(rf"\b{word}\b", s.lower()):
            return val
    return None


# ── Task-specific 问题与打分 ──────────────────────────────────────────────────

def make_vqa(task: str, anchor: str, target_word: str):
    """
    返回 (question: str, score_fn: Callable[[str], float])
    score_fn 接受 VLM 的原始回复，返回 0~1 浮点分数。
    """
    anchor_clean = anchor.strip().rstrip(".")

    if task == "counting":
        tokens = anchor_clean.lower().split()
        number_word = tokens[0] if tokens else ""
        noun = " ".join(tokens[1:]) if len(tokens) > 1 else anchor_clean
        expected_int = WORD_TO_INT.get(number_word)

        question = (
            f"How many {noun} are visible in this image? "
            f"Answer with a single number only, e.g. '3'."
        )

        def score_fn(pred: str) -> float:
            pred_int = _parse_int(pred)
            if pred_int is None or expected_int is None:
                return 0.0
            if pred_int == expected_int:
                return 1.0
            if abs(pred_int - expected_int) == 1:
                return 0.5
            return 0.0

        return question, score_fn

    elif task == "color":
        # anchor: "a black apple."  target_word: "black"
        question = (
            f"What is the color of the main object in this image? "
            f"Answer with a single color word only."
        )
        expected_color = (target_word or "").strip().lower()

        def score_fn(pred: str) -> float:
            return 1.0 if expected_color and expected_color in pred.lower() else 0.0

        return question, score_fn

    elif task == "non-spatial":
        # anchor: "a cat is sleeping"  target_word: "sleeping"
        question = (
            f"Does this image depict the following: '{anchor_clean}'? "
            f"Answer with yes or no only."
        )

        def score_fn(pred: str) -> float:
            p = pred.strip().lower()
            if p.startswith("yes"):
                return 1.0
            if p.startswith("no"):
                return 0.0
            # 含 yes 但不以 yes 开头（如 "Yes, it does"）
            return 1.0 if "yes" in p else 0.0

        return question, score_fn

    else:  # scene
        # anchor: "on a sunny beach"  target_word: "beach"
        keyword = (target_word or anchor_clean).strip().lower()
        question = (
            f"Does this image show a scene that can be described as '{anchor_clean}'? "
            f"Answer with yes or no only."
        )

        def score_fn(pred: str) -> float:
            p = pred.strip().lower()
            return 1.0 if p.startswith("yes") or ("yes" in p and "no" not in p) else 0.0

        return question, score_fn


# ── 构建 (task,anchor) -> target_word 查找表 ──────────────────────────────────

def build_target_lookup(triplet_dir: str) -> dict:
    """从原始 JSONL 里读取 target_word，key=(task,anchor)。"""
    lookup = {}
    for jsonl_file in Path(triplet_dir).glob("*.jsonl"):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                key = (obj.get("task", ""), obj.get("anchor", "").strip())
                lookup[key] = obj.get("target_word", "")
    return lookup


# ── VLM 加载与推理 ────────────────────────────────────────────────────────────

def load_vlm(model_name: str, device: str):
    """
    自动适配 Qwen 各代 VLM：
      Qwen3-VL-*   → Qwen3VLForConditionalGeneration
      Qwen2.5-VL-* / Qwen2-VL-* → Qwen2VLForConditionalGeneration
    """
    from transformers import AutoProcessor

    name_lower = model_name.lower()
    print(f"  Loading {model_name} ...")

    if "qwen3" in name_lower:
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)
    else:
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)

    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def vlm_answer(model, processor, image: Image.Image,
               question: str, device: str, max_new_tokens: int = 32) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text":  question},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs,
            videos=video_inputs, padding=True, return_tensors="pt",
        ).to(device)
    except ImportError:
        inputs = processor(
            text=[text], images=[image], padding=True, return_tensors="pt",
        ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated = [
        out[len(inp):]
        for out, inp in zip(output_ids, inputs.input_ids)
    ]
    return processor.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


# ── 主流程 ────────────────────────────────────────────────────────────────────

def collect_jobs(image_dir: str, tasks: list, rank: int, world_size: int):
    """
    遍历 image_dir 下所有 meta.json，返回该 rank 负责的子集。
    每项：(img_dir: Path, meta: dict, png_files: List[Path])
    """
    all_jobs = []
    for meta_file in sorted(Path(image_dir).rglob("meta.json")):
        img_dir = meta_file.parent
        with open(meta_file) as f:
            meta = json.load(f)
        task = meta.get("task", "")
        if tasks and task not in tasks:
            continue
        png_files = sorted(img_dir.glob("seed*.png"))
        if not png_files:
            continue
        all_jobs.append((img_dir, meta, png_files))
    return all_jobs[rank::world_size]


def run_filter(opt):
    rank   = opt.rank
    device = f"cuda:{opt.gpu_id}"

    print(f"[rank={rank}] Loading VLM: {opt.vlm_model} on {device}")
    model, processor = load_vlm(opt.vlm_model, device)

    target_lookup = build_target_lookup(opt.triplet_dir)

    jobs = collect_jobs(opt.image_dir, opt.tasks, rank, opt.world_size)
    print(f"[rank={rank}] Jobs: {len(jobs)}")

    passed = failed = skipped = errors = 0
    summary_rows = []

    for img_dir, meta, png_files in tqdm(jobs, desc=f"rank={rank}", position=rank, leave=True):
        verdict_file = img_dir / "verdict.json"

        if verdict_file.exists() and not opt.overwrite:
            skipped += 1
            continue

        task   = meta["task"]
        anchor = meta["anchor"]
        target = target_lookup.get((task, anchor), "")
        question, score_fn = make_vqa(task, anchor, target)

        per_image = []
        for png in png_files:
            try:
                img  = Image.open(png).convert("RGB")
                pred = vlm_answer(model, processor, img, question, device, opt.max_new_tokens)
                score = score_fn(pred)
                per_image.append({
                    "image":     png.name,
                    "question":  question,
                    "predicted": pred,
                    "score":     score,
                    "pass":      score >= opt.threshold,
                })
            except Exception as e:
                print(f"\n[rank={rank} ERR] {png}: {e}")
                errors += 1
                per_image.append({"image": png.name, "error": str(e), "score": 0.0, "pass": False})

        avg_score = sum(r["score"] for r in per_image) / len(per_image) if per_image else 0.0
        verdict = {
            "task":      task,
            "anchor":    anchor,
            "target":    target,
            "avg_score": round(avg_score, 4),
            "threshold": opt.threshold,
            "pass":      avg_score >= opt.threshold,
            "results":   per_image,
        }
        with open(verdict_file, "w") as f:
            json.dump(verdict, f, ensure_ascii=False, indent=2)

        if verdict["pass"]:
            passed += 1
            summary_rows.append({
                "img_dir":   str(img_dir),
                "task":      task,
                "anchor":    anchor,
                "avg_score": verdict["avg_score"],
            })
        else:
            failed += 1

    print(f"\n[rank={rank}] passed={passed}  failed={failed}  skipped={skipped}  errors={errors}")

    summary_path = Path(opt.image_dir) / f"filtered_rank{rank}.jsonl"
    with open(summary_path, "w") as f:
        for row in summary_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[rank={rank}] Summary → {summary_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image_dir",      type=str,   default="data/generated_images",
                   help="generate_triplet_images.py 的输出目录")
    p.add_argument("--triplet_dir",    type=str,   default="data/train_triplets",
                   help="用于查找 target_word 的原始 JSONL 目录")
    p.add_argument("--tasks",          type=str,   nargs="+",
                   default=["counting", "color", "scene", "non-spatial"])
    p.add_argument("--vlm_model",      type=str,   default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--threshold",      type=float, default=0.5,
                   help="avg_score >= threshold 则判定为 pass")
    p.add_argument("--max_new_tokens", type=int,   default=32)
    p.add_argument("--gpu_id",         type=int,   default=0,
                   help="当 CUDA_VISIBLE_DEVICES 已设置时统一用 0")
    p.add_argument("--rank",           type=int,   default=0)
    p.add_argument("--world_size",     type=int,   default=1)
    p.add_argument("--overwrite",      action="store_true",
                   help="重新评估已有 verdict.json 的样本")
    return p.parse_args()


if __name__ == "__main__":
    run_filter(parse_args())
