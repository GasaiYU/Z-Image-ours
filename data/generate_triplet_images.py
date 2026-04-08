"""
generate_triplet_images.py
===========================
从 data/train_triplets/ 下的所有 JSONL 文件中提取 anchor prompt，
去重后用 QwenImage 多卡并行批量生成训练图像。

输出目录结构：
    data/generated_images/
        <task>/
            <sanitized_anchor>/
                seed{seed}.png
                meta.json

单卡用法：
    python data/generate_triplet_images.py \
        --triplet_dir data/train_triplets \
        --outdir      data/generated_images \
        --n_samples   4 --batch_size 4

多卡用法（自动检测所有 GPU）：
    python data/generate_triplet_images.py --n_gpus 8

指定 GPU：
    python data/generate_triplet_images.py --gpu_ids 0 1 2 3
"""

import argparse
import json
import re
import torch
import torch.multiprocessing as mp
from pathlib import Path
from tqdm import tqdm


# ── 默认生成参数 ──────────────────────────────────────────────────────────────
DEFAULT_WIDTH   = 1024
DEFAULT_HEIGHT  = 1024
INFERENCE_STEPS = 50
CFG_SCALE       = 4.0
NEGATIVE_PROMPT = (
    "low resolution, low quality, deformed limbs, deformed fingers, "
    "oversaturated, waxy skin, no facial details, over-smooth, AI-looking. "
    "messy composition. blurry or distorted text."
)


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def sanitize(text: str, maxlen: int = 80) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"[\s]+", "_", s).strip("_")
    return s[:maxlen]


def to_prompt(anchor: str) -> str:
    a = anchor.strip().rstrip(".")
    if not a[0].isupper():
        return f"a photo of {a}"
    return a


def load_all_anchors(triplet_dir: str, tasks: list) -> list:
    """
    读取所有 JSONL，全局去重，返回 [(task, anchor), ...] 列表。
    """
    triplet_dir = Path(triplet_dir)
    seen   = set()
    items  = []
    for jsonl_file in sorted(triplet_dir.glob("*.jsonl")):
        with open(jsonl_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj    = json.loads(line)
                task   = obj.get("task", jsonl_file.stem)
                anchor = obj.get("anchor", "").strip()
                if not anchor or (tasks and task not in tasks):
                    continue
                key = (task, anchor)
                if key not in seen:
                    seen.add(key)
                    items.append(key)
    return items


# ── 单卡 Worker ───────────────────────────────────────────────────────────────

def worker(rank: int, gpu_id: int, shard: list, opt):
    """每个进程负责处理 shard 中的 anchor 列表，独占 gpu_id 这张卡。"""
    device = f"cuda:{gpu_id}"
    dtype  = torch.bfloat16

    # 只在 rank==0 的进程打印总体信息
    is_main = (rank == 0)
    if is_main:
        print(f"[GPU {gpu_id}] Loading QwenImagePipeline ...")

    # 解决多卡加载模型时的 IO 拥堵：错开加载时间
    if not is_main:
        import time
        import random
        # 睡 0~10 秒，避免 8 个进程同时读硬盘、解压 safetensors 导致 IO 瓶颈
        time.sleep(random.uniform(0, 10))

    from diffusers import QwenImagePipeline
    pipe = QwenImagePipeline.from_pretrained(
        "Qwen/Qwen-Image-2512", torch_dtype=dtype
    ).to(device)
    
    # 开启内存/速度优化：
    # 1. 禁用 cudnn_benchmark，在多进程变长序列下它会导致每次都重新寻找最优算法，反而极慢
    torch.backends.cudnn.benchmark = False
    
    # 2. VAE Tiling 和 Slicing 会大幅降低推理速度（用时间换空间）。
    # 如果你不爆显存（比如 A100 80G），千万不要开！
    # pipe.vae.enable_tiling()
    # pipe.vae.enable_slicing()
    
    # 3. 开启 xformers 或 sdpa 加速（如果环境支持）
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass # fallback to default SDPA

    pipe.set_progress_bar_config(disable=True)

    outdir = Path(opt.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    done = skipped = errors = 0
    desc = f"GPU:{gpu_id}"
    for task, anchor in tqdm(shard, desc=desc, position=rank, leave=True):
        img_dir = outdir / task / sanitize(anchor)
        img_dir.mkdir(parents=True, exist_ok=True)

        # 断点续传：已生成足够数量则跳过
        if len(list(img_dir.glob("seed*.png"))) >= opt.n_samples:
            skipped += 1
            continue

        prompt = to_prompt(anchor)
        try:
            images = []
            generated = 0
            while generated < opt.n_samples:
                cur_bs = min(opt.batch_size, opt.n_samples - generated)
                gens   = [
                    torch.Generator(device=device).manual_seed(opt.seed + generated + i)
                    for i in range(cur_bs)
                ]
                outs = pipe(
                    prompt=[prompt] * cur_bs,
                    negative_prompt=[NEGATIVE_PROMPT] * cur_bs,
                    width=opt.width,
                    height=opt.height,
                    num_inference_steps=INFERENCE_STEPS,
                    true_cfg_scale=CFG_SCALE,
                    generator=gens,
                ).images
                images.extend(outs)
                generated += cur_bs
        except Exception as e:
            print(f"\n  [GPU {gpu_id} ERROR] '{prompt}': {e}")
            errors += 1
            continue

        for i, img in enumerate(images):
            img.save(img_dir / f"seed{opt.seed + i}.png")

        meta = {"task": task, "anchor": anchor, "prompt": prompt,
                "n_samples": opt.n_samples, "width": opt.width, "height": opt.height}
        with open(img_dir / "meta.json", "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        done += 1

    print(f"\n[GPU {gpu_id}] done={done}  skipped={skipped}  errors={errors}")


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main(opt):
    # 确定使用哪几张 GPU
    n_available = torch.cuda.device_count()
    if opt.gpu_ids:
        gpu_ids = opt.gpu_ids
    elif opt.n_gpus > 0:
        gpu_ids = list(range(min(opt.n_gpus, n_available)))
    else:
        gpu_ids = list(range(n_available)) if n_available > 0 else []

    if not gpu_ids:
        print("No CUDA GPU found, falling back to CPU (slow!).")
        gpu_ids = [None]   # None → CPU

    # 收集所有 anchor，全局去重
    all_items = load_all_anchors(opt.triplet_dir, opt.tasks)
    total     = len(all_items)
    n_gpus    = len(gpu_ids)
    print(f"Total unique (task, anchor) pairs: {total}")
    print(f"Using {n_gpus} GPU(s): {gpu_ids}")

    if n_gpus == 1:
        # 单卡：直接在主进程里跑，方便调试
        worker(0, gpu_ids[0] if gpu_ids[0] is not None else "cpu",
               all_items, opt)
    else:
        # 多卡：均匀切分，每张卡启动一个独立进程
        shards = [all_items[i::n_gpus] for i in range(n_gpus)]
        processes = []
        mp.set_start_method("spawn", force=True)
        for rank, (gpu_id, shard) in enumerate(zip(gpu_ids, shards)):
            p = mp.Process(target=worker, args=(rank, gpu_id, shard, opt))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    print(f"\nAll done. Images saved to: {opt.outdir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--triplet_dir", type=str, default="data/train_triplets")
    p.add_argument("--outdir",      type=str, default="data/generated_images")
    p.add_argument("--tasks",       type=str, nargs="+",
                   default=["counting", "color", "scene", "non-spatial"])
    p.add_argument("--n_samples",   type=int, default=4,
                   help="每个 anchor 生成几张图")
    p.add_argument("--batch_size",  type=int, default=1,
                   help="单次推理 batch size (建议设为1，QwenImage 1024x1024 很容易 OOM)")
    p.add_argument("--width",       type=int, default=DEFAULT_WIDTH)
    p.add_argument("--height",      type=int, default=DEFAULT_HEIGHT)
    p.add_argument("--seed",        type=int, default=42)

    # 多卡控制（二选一）
    p.add_argument("--n_gpus",   type=int, default=0,
                   help="使用前 N 张 GPU（0 = 自动检测所有）")
    p.add_argument("--gpu_ids",  type=int, nargs="+", default=None,
                   help="手动指定 GPU ID，如 --gpu_ids 0 1 4 5")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
