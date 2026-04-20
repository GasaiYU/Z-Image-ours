"""
build_dpo_training_index.py
===========================
扫描 data/dpo_edit_images 下的 noun/count 目录，读取 meta.json 和 verdict.json，
生成按 noun 聚合的 DPO 训练索引 JSON。

输出为一个带缩进的 JSON 数组，每个元素对应一个 noun，包含该 noun 下不同数量词
的 anchor pools，以及满足阈值的图片路径。

示例：
  python data/utils/build_dpo_training_index.py \
      --dpo_root data/dpo_edit_images \
      --output_json data/train_triplets/DPO/counting_dpo_index.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional


VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_THRESHOLD = 0.8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build noun-grouped DPO training index from dpo_edit_images.")
    p.add_argument("--dpo_root", type=str, default="data/dpo_edit_images", help="DPO 编辑图目录根路径")
    p.add_argument(
        "--output_json",
        type=str,
        default="data/train_triplets/DPO/counting_dpo_index.json",
        help="输出索引 JSON 文件",
    )
    p.add_argument(
        "--output_jsonl",
        type=str,
        default="",
        help="兼容旧参数名；若提供则等价于 --output_json",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="单图筛选阈值；不指定则优先用 verdict.threshold，否则回退到脚本默认值",
    )
    p.add_argument(
        "--min_count_pools",
        type=int,
        default=2,
        help="每个 noun 至少保留多少个 count pool；默认 2，便于同 noun 不同数量采样",
    )
    p.add_argument(
        "--include_empty_pools",
        action="store_true",
        help="若设置，则 count pool 即使没有 valid_images 也保留到输出中",
    )
    return p.parse_args()


def maybe_load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def to_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def resolve_anchor(noun: str, count_word: str) -> str:
    return f"{count_word} {noun}".strip()


def choose_effective_threshold(
    verdict: Optional[dict[str, Any]],
    cli_threshold: Optional[float],
) -> float:
    if cli_threshold is not None:
        return cli_threshold
    threshold = None if verdict is None else verdict.get("threshold")
    if isinstance(threshold, (int, float)):
        return float(threshold)
    return DEFAULT_THRESHOLD


def is_valid_item(
    item: dict[str, Any],
    verdict: Optional[dict[str, Any]],
    cli_threshold: Optional[float],
) -> bool:
    score = item.get("score")
    if not isinstance(score, (int, float)):
        return False
    return float(score) >= choose_effective_threshold(verdict, cli_threshold)


def build_pool_entry(
    count_dir: Path,
    project_root: Path,
    cli_threshold: Optional[float],
) -> tuple[Optional[dict[str, Any]], dict[str, int]]:
    stats = {
        "missing_meta": 0,
        "bad_meta": 0,
        "missing_verdict": 0,
        "bad_verdict": 0,
        "empty_all_images": 0,
        "empty_valid_images": 0,
    }

    meta_path = count_dir / "meta.json"
    verdict_path = count_dir / "verdict.json"

    meta = maybe_load_json(meta_path)
    if meta is None:
        if meta_path.exists():
            stats["bad_meta"] += 1
        else:
            stats["missing_meta"] += 1
        return None, stats

    verdict = maybe_load_json(verdict_path)
    if verdict is None:
        if verdict_path.exists():
            stats["bad_verdict"] += 1
        else:
            stats["missing_verdict"] += 1
        return None, stats

    all_images = sorted(
        p.name for p in count_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    )
    if not all_images:
        stats["empty_all_images"] += 1

    results = verdict.get("results", [])
    result_by_name: dict[str, dict[str, Any]] = {}
    if isinstance(results, list):
        for item in results:
            if not isinstance(item, dict):
                continue
            name = item.get("image")
            if isinstance(name, str) and name:
                result_by_name[name] = item

    valid_images = [
        name for name in all_images
        if name in result_by_name and is_valid_item(result_by_name[name], verdict, cli_threshold)
    ]
    if not valid_images:
        stats["empty_valid_images"] += 1

    noun = str(meta.get("noun", "")).strip() or str(verdict.get("noun", "")).strip()
    target_count = meta.get("target_count", verdict.get("target_count"))
    count_word = str(meta.get("count_word", verdict.get("count_word", count_dir.name))).strip()
    anchor = resolve_anchor(noun, count_word)

    pool = {
        "noun": noun,
        "anchor": anchor,
        "target_count": target_count,
        "image_paths": [
            to_rel(count_dir / image_name, project_root) for image_name in valid_images
        ],
    }
    return pool, stats


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    dpo_root = Path(args.dpo_root)
    output_arg = args.output_jsonl or args.output_json
    output_path = Path(output_arg)

    if not dpo_root.exists():
        print(f"[ERROR] dpo_root not found: {dpo_root}", file=sys.stderr)
        sys.exit(1)

    noun_to_pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    stats = Counter()

    noun_dirs = sorted(p for p in dpo_root.iterdir() if p.is_dir() and p.name != "_seeds")
    for noun_dir in noun_dirs:
        stats["noun_dirs_scanned"] += 1
        count_dirs = sorted(p for p in noun_dir.iterdir() if p.is_dir())
        for count_dir in count_dirs:
            stats["count_dirs_scanned"] += 1
            pool, pool_stats = build_pool_entry(count_dir, project_root, args.threshold)
            stats.update(pool_stats)
            if pool is None:
                continue
            if not args.include_empty_pools and not pool["image_paths"]:
                continue

            noun = str(pool.get("noun", "")).strip() or noun_dir.name.replace("_", " ")
            noun_to_pools[noun].append(pool)

    kept_rows: list[dict[str, Any]] = []
    pool_hist = Counter()
    dropped_too_few_pools = 0

    for noun, pools in sorted(noun_to_pools.items()):
        pools = sorted(
            pools,
            key=lambda x: (
                x.get("target_count") if isinstance(x.get("target_count"), int) else 1_000_000,
                str(x.get("anchor", "")),
            ),
        )
        if len(pools) < args.min_count_pools:
            dropped_too_few_pools += 1
            continue
        anchor_pools = [
            {
                "anchor": pool["anchor"],
                "image_paths": pool["image_paths"],
            }
            for pool in pools
        ]
        kept_rows.append(
            {
                "task": "counting",
                "noun": noun,
                "anchor_pools": anchor_pools,
            }
        )
        pool_hist[len(anchor_pools)] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kept_rows, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print("\n[SCAN]")
    print(f"  noun_dirs_scanned     : {stats['noun_dirs_scanned']}")
    print(f"  count_dirs_scanned    : {stats['count_dirs_scanned']}")

    print("\n[ISSUES]")
    print(f"  missing_meta          : {stats['missing_meta']}")
    print(f"  bad_meta              : {stats['bad_meta']}")
    print(f"  missing_verdict       : {stats['missing_verdict']}")
    print(f"  bad_verdict           : {stats['bad_verdict']}")
    print(f"  empty_all_images      : {stats['empty_all_images']}")
    print(f"  empty_valid_images    : {stats['empty_valid_images']}")

    print("\n[KEEP]")
    print(f"  nouns_kept            : {len(kept_rows)}")
    print(f"  nouns_dropped_pools   : {dropped_too_few_pools}")
    print(f"  min_count_pools       : {args.min_count_pools}")

    print("\n[POOL DISTRIBUTION]")
    if pool_hist:
        for n_pools in sorted(pool_hist):
            print(f"  nouns_with_{n_pools}_pools : {pool_hist[n_pools]}")
    else:
        print("  (empty)")

    print(f"\n[OUTPUT] {output_path}")


if __name__ == "__main__":
    main()
