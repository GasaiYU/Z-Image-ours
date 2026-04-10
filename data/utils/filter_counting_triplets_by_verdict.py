"""
filter_counting_triplets_by_verdict.py
=====================================
根据 generated_images/counting/<sanitized_anchor>/verdict.json 的评分过滤
data/train_triplets/counting_triplets.jsonl。

默认规则：
  - 保留存在 verdict.json 且至少有一张图 score > threshold 的样本
  - threshold 默认 0.8

示例：
  python data/filter_counting_triplets_by_verdict.py
  python data/filter_counting_triplets_by_verdict.py --threshold 0.8
  python data/filter_counting_triplets_by_verdict.py --mode avg
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any


def sanitize(text: str, maxlen: int = 80) -> str:
    """与生成脚本保持一致的 anchor -> 目录名映射。"""
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"[\s]+", "_", s).strip("_")
    return s[:maxlen]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def score_from_verdict(verdict: dict[str, Any], mode: str) -> float | None:
    """从 verdict.json 提取用于过滤的分数。"""
    if mode == "avg":
        val = verdict.get("avg_score")
        return float(val) if isinstance(val, (int, float)) else None

    # mode == "max"
    scores: list[float] = []
    for item in verdict.get("results", []):
        s = item.get("score")
        if isinstance(s, (int, float)):
            scores.append(float(s))
    if scores:
        return max(scores)

    # 兼容没有 results 的旧格式
    val = verdict.get("avg_score")
    return float(val) if isinstance(val, (int, float)) else None


def passes_threshold(anchor: str, generated_root: Path, threshold: float, mode: str) -> tuple[bool, str]:
    verdict_path = generated_root / "counting" / sanitize(anchor) / "verdict.json"
    if not verdict_path.exists():
        return False, "missing_verdict"

    try:
        with open(verdict_path, "r", encoding="utf-8") as f:
            verdict = json.load(f)
    except Exception:
        return False, "bad_verdict"

    score = score_from_verdict(verdict, mode)
    if score is None:
        return False, "no_score"

    return score > threshold, "kept" if score > threshold else "low_score"


def main(opt: argparse.Namespace) -> None:
    input_path = Path(opt.input_jsonl)
    generated_root = Path(opt.generated_root)
    output_path = Path(opt.output_jsonl) if opt.output_jsonl else input_path.with_name(
        f"{input_path.stem}.filtered_{str(opt.threshold).replace('.', 'p')}.jsonl"
    )

    rows = read_jsonl(input_path)
    kept_rows: list[dict[str, Any]] = []
    stats = {
        "total": 0,
        "kept": 0,
        "missing_verdict": 0,
        "bad_verdict": 0,
        "no_score": 0,
        "low_score": 0,
        "no_anchor": 0,
    }

    for row in rows:
        stats["total"] += 1
        anchor = str(row.get("anchor", "")).strip()
        if not anchor:
            stats["no_anchor"] += 1
            continue

        ok, reason = passes_threshold(anchor, generated_root, opt.threshold, opt.mode)
        if ok:
            kept_rows.append(row)
            stats["kept"] += 1
        else:
            stats[reason] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for row in kept_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("========================================")
    print(f"Input JSONL     : {input_path}")
    print(f"Generated root  : {generated_root}")
    print(f"Filter mode     : {opt.mode}  (score > {opt.threshold})")
    print(f"Output JSONL    : {output_path}")
    print("========================================")
    print(f"Total rows       : {stats['total']}")
    print(f"Kept rows        : {stats['kept']}")
    print(f"Dropped low score: {stats['low_score']}")
    print(f"Missing verdict  : {stats['missing_verdict']}")
    print(f"Bad verdict      : {stats['bad_verdict']}")
    print(f"No score         : {stats['no_score']}")
    print(f"No anchor        : {stats['no_anchor']}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter counting triplets by verdict score")
    p.add_argument("--input_jsonl", type=str, default="data/train_triplets/counting_triplets.jsonl")
    p.add_argument("--generated_root", type=str, default="data/generated_images")
    p.add_argument("--output_jsonl", type=str, default="")
    p.add_argument("--threshold", type=float, default=0.8, help="keep sample if score > threshold")
    p.add_argument(
        "--mode",
        type=str,
        default="max",
        choices=["max", "avg"],
        help="max: 使用单目录下最高单图分数；avg: 使用 verdict.json 的 avg_score",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
