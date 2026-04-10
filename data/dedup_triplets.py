"""
dedup_triplets.py
=================
对 data/train_triplets/ 下的 JSONL triplet 文件去重。

支持两种去重策略：
  1. exact   —— 整行完全相同才视为重复（默认）
  2. anchor  —— 对 counting 行先做数词-名词单复数修正，再去重
                 规则：单数无 s，复数有 s（兼容常见 es/ies）

用法：
  # 对单个文件，原地去重（会备份原文件为 .bak）
  python data/dedup_triplets.py --input data/train_triplets/counting_triplets.jsonl

  # 指定输出路径（不覆盖原文件）
  python data/dedup_triplets.py --input data/train_triplets/counting_triplets.jsonl \
                                --output data/train_triplets/counting_triplets_dedup.jsonl

  # 对 train_triplets/ 下所有文件批量去重
  python data/dedup_triplets.py --all

  # 只处理 counting / color 两个任务
  python data/dedup_triplets.py --all --tasks counting color

  # 使用 anchor 归一化去重（单复数合并）
  python data/dedup_triplets.py --input data/train_triplets/counting_triplets.jsonl \
                                --mode anchor
"""

import argparse
import json
import shutil
from pathlib import Path


# ── 归一化函数 ────────────────────────────────────────────────────────────────

NUMBER_WORD_TO_INT = {
    "a": 1, "an": 1,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}


def parse_count_token(token: str) -> int | None:
    t = token.strip().lower()
    if t in NUMBER_WORD_TO_INT:
        return NUMBER_WORD_TO_INT[t]
    if t.isdigit():
        return int(t)
    return None


def to_singular(noun: str) -> str:
    n = noun.strip()
    lower = n.lower()
    if lower.endswith("ies") and len(lower) > 3:
        return n[:-3] + "y"
    if lower.endswith("oes") and len(lower) > 3:
        return n[:-2]
    if lower.endswith("es") and len(lower) > 2:
        if lower.endswith(("ches", "shes", "xes", "zes", "ses")):
            return n[:-2]
    if lower.endswith("s") and not lower.endswith("ss") and len(lower) > 1:
        return n[:-1]
    return n


def to_plural(noun: str) -> str:
    n = noun.strip()
    lower = n.lower()
    if lower.endswith(("s", "x", "z", "ch", "sh")):
        return n + "es"
    if len(lower) >= 2 and lower.endswith("o") and lower[-2] not in "aeiou":
        return n + "es"
    if len(lower) >= 2 and lower.endswith("y") and lower[-2] not in "aeiou":
        return n[:-1] + "ies"
    return n + "s"


def normalize_count_phrase(text: str) -> str:
    """
    将 counting 短语规范为：
      - 数量为 1：末尾名词单数
      - 数量不为 1：末尾名词复数
    """
    phrase = " ".join(text.strip().split())
    if not phrase:
        return phrase

    words = phrase.split(" ")
    if len(words) < 2:
        return phrase

    count = parse_count_token(words[0])
    if count is None:
        return phrase

    noun = words[-1]
    noun_singular = to_singular(noun)
    words[-1] = noun_singular if count == 1 else to_plural(noun_singular)
    return " ".join(words)


def normalize_counting_row(row: dict) -> dict:
    fixed = dict(row)
    if fixed.get("task") != "counting":
        return fixed
    for field in ("anchor", "positive", "negative"):
        val = fixed.get(field)
        if isinstance(val, str):
            fixed[field] = normalize_count_phrase(val)
    return fixed


# ── 核心去重逻辑 ──────────────────────────────────────────────────────────────

def dedup_file(input_path: Path, output_path: Path, mode: str) -> tuple[int, int]:
    """
    对单个 JSONL 文件去重。

    Returns:
        (原始行数, 去重后行数)
    """
    rows: list[dict] = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    original_count = len(rows)
    seen: set = set()
    deduped: list[dict] = []

    for row in rows:
        if mode == "exact":
            # 用整行 JSON 字符串作为 key（字段顺序一致时可靠）
            key = json.dumps(row, ensure_ascii=False, sort_keys=True)
            row_to_save = row
        elif mode == "anchor":
            # 对 counting 行修正单复数，再按规范化后的整行去重
            row_to_save = normalize_counting_row(row)
            key = json.dumps(row_to_save, ensure_ascii=False, sort_keys=True)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if key not in seen:
            seen.add(key)
            deduped.append(row_to_save)

    with open(output_path, "w", encoding="utf-8") as f:
        for row in deduped:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return original_count, len(deduped)


# ── 主入口 ────────────────────────────────────────────────────────────────────

def main(opt):
    triplet_dir = Path(opt.triplet_dir)

    if opt.all:
        # 批量处理
        if opt.tasks:
            files = [triplet_dir / f"{t}_triplets.jsonl" for t in opt.tasks]
        else:
            files = sorted(triplet_dir.glob("*_triplets.jsonl"))
    else:
        if not opt.input:
            raise ValueError("必须指定 --input 或 --all")
        files = [Path(opt.input)]

    for input_path in files:
        if not input_path.exists():
            print(f"[SKIP] 文件不存在: {input_path}")
            continue

        if opt.output and not opt.all:
            output_path = Path(opt.output)
        else:
            # 原地覆盖，先备份
            backup_path = input_path.with_suffix(".jsonl.bak")
            shutil.copy2(input_path, backup_path)
            print(f"  备份原文件 → {backup_path}")
            output_path = input_path

        original, deduped = dedup_file(input_path, output_path, mode=opt.mode)
        removed = original - deduped
        print(
            f"[{input_path.name}]  "
            f"原始: {original}  去重后: {deduped}  "
            f"移除: {removed}  ({removed / original * 100:.1f}%)"
        )


def parse_args():
    p = argparse.ArgumentParser(description="Triplet JSONL 去重工具")
    p.add_argument("--input",       type=str, default=None,
                   help="单文件输入路径")
    p.add_argument("--output",      type=str, default=None,
                   help="输出路径（不指定则原地覆盖并备份）")
    p.add_argument("--all",         action="store_true",
                   help="批量处理 triplet_dir 下所有文件")
    p.add_argument("--tasks",       type=str, nargs="+", default=None,
                   help="配合 --all，只处理指定 task（如 counting color）")
    p.add_argument("--triplet_dir", type=str, default="data/train_triplets",
                   help="triplet 文件目录")
    p.add_argument("--mode",        type=str, default="exact",
                   choices=["exact", "anchor"],
                   help="去重模式：exact=整行去重，anchor=anchor归一化去重")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
