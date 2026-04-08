"""
generate_triplet_images.py
===========================
从 data/train_triplets/ 下的所有 JSONL 文件中提取 anchor prompt，
去重后用 QwenImage 批量生成训练图像。

输出目录结构：
    data/generated_images/
        <task>/
            <sanitized_anchor>/
                seed{seed}.png
                meta.json

用法：
    python data/generate_triplet_images.py \
        --triplet_dir data/train_triplets \
        --outdir     data/generated_images \
        --n_samples  4 \
        --batch_size 4 \
        --tasks counting color scene non-spatial

"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


# ── 默认生成参数 ──────────────────────────────────────────────────────────────
DEFAULT_WIDTH  = 1024
DEFAULT_HEIGHT = 1024
INFERENCE_STEPS = 50
CFG_SCALE = 4.0
NEGATIVE_PROMPT = (
    "low resolution, low quality, deformed limbs, deformed fingers, "
    "oversaturated, waxy skin, no facial details, over-smooth, AI-looking. "
    "messy composition. blurry or distorted text."
)


def sanitize(text: str, maxlen: int = 80) -> str:
    """将 prompt 转为合法目录名。"""
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"[\s]+", "_", s).strip("_")
    return s[:maxlen]


def load_all_anchors(triplet_dir: str, tasks: list[str]) -> dict[str, list[str]]:
    """
    从 triplet_dir 下读取所有 JSONL，按 task 分类并去重。
    返回 {task: [anchor, ...]}
    """
    triplet_dir = Path(triplet_dir)
    task_anchors: dict[str, set] = {}

    for jsonl_file in sorted(triplet_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                task   = obj.get("task", jsonl_file.stem)
                anchor = obj.get("anchor", "").strip()
                if not anchor:
                    continue
                if tasks and task not in tasks:
                    continue
                task_anchors.setdefault(task, set()).add(anchor)

    return {task: sorted(anchors) for task, anchors in task_anchors.items()}


def to_prompt(anchor: str) -> str:
    """将短 anchor 包装成完整 prompt。"""
    a = anchor.strip().rstrip(".")
    if not a[0].isupper():
        # 短名词 → "a photo of X"
        return f"a photo of {a}"
    # 已经是完整句子（non-spatial 类型）
    return a


def generate_images(pipe, prompt: str, n_samples: int, batch_size: int,
                    width: int, height: int, device: str, seed_offset: int = 0):
    """生成 n_samples 张图片，返回 PIL Image 列表。"""
    images = []
    generated = 0
    while generated < n_samples:
        cur_bs = min(batch_size, n_samples - generated)
        generators = [
            torch.Generator(device=device).manual_seed(seed_offset + generated + i)
            for i in range(cur_bs)
        ]
        outs = pipe(
            prompt=[prompt] * cur_bs,
            negative_prompt=[NEGATIVE_PROMPT] * cur_bs,
            width=width,
            height=height,
            num_inference_steps=INFERENCE_STEPS,
            true_cfg_scale=CFG_SCALE,
            generator=generators,
        ).images
        images.extend(outs)
        generated += cur_bs
    return images


def main(opt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading QwenImagePipeline on {device} ...")
    from diffusers import QwenImagePipeline
    pipe = QwenImagePipeline.from_pretrained(
        "Qwen/Qwen-Image-2512", torch_dtype=dtype
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    # 收集所有 anchor
    task_anchors = load_all_anchors(opt.triplet_dir, opt.tasks)
    total = sum(len(v) for v in task_anchors.values())
    print(f"Tasks: {list(task_anchors.keys())}")
    print(f"Total unique anchors: {total}")

    outdir = Path(opt.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    done = skipped = 0
    for task, anchors in task_anchors.items():
        task_dir = outdir / task
        task_dir.mkdir(exist_ok=True)

        for anchor in tqdm(anchors, desc=task, leave=True):
            slug    = sanitize(anchor)
            img_dir = task_dir / slug
            img_dir.mkdir(exist_ok=True)

            # 跳过已生成的（支持断点续传）
            existing = list(img_dir.glob("seed*.png"))
            if len(existing) >= opt.n_samples:
                skipped += 1
                continue

            prompt = to_prompt(anchor)

            try:
                imgs = generate_images(
                    pipe, prompt,
                    n_samples=opt.n_samples,
                    batch_size=opt.batch_size,
                    width=opt.width,
                    height=opt.height,
                    device=device,
                    seed_offset=opt.seed,
                )
            except Exception as e:
                print(f"  [ERROR] '{prompt}': {e}")
                continue

            for i, img in enumerate(imgs):
                img.save(img_dir / f"seed{opt.seed + i}.png")

            # 保存 meta
            meta = {"task": task, "anchor": anchor, "prompt": prompt,
                    "n_samples": opt.n_samples, "width": opt.width, "height": opt.height}
            with open(img_dir / "meta.json", "w") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            done += 1

    print(f"\nDone. Generated: {done}  |  Skipped (already exists): {skipped}")
    print(f"Images saved to: {outdir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--triplet_dir", type=str, default="data/train_triplets",
                   help="包含 *.jsonl 的目录")
    p.add_argument("--outdir",      type=str, default="data/generated_images",
                   help="图像输出根目录")
    p.add_argument("--tasks",       type=str, nargs="+",
                   default=["counting", "color", "scene", "non-spatial"],
                   help="只处理哪些 task；默认全部")
    p.add_argument("--n_samples",   type=int, default=4,
                   help="每个 anchor 生成几张图")
    p.add_argument("--batch_size",  type=int, default=4,
                   help="单次推理的 batch size")
    p.add_argument("--width",       type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height",      type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
