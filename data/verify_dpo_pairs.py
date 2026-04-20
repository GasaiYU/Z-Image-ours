"""
verify_dpo_pairs.py
===================
检查 generate_dpo_pairs.py 输出的 DPO 样本是否完整：

(1) 生成完整性：所有期望的 (noun, count) 目录是否存在，且 seed 图片数量是否达标
(2) 验证完整性：所有已生成图片是否都被写入对应 verdict.json 的 results

默认检查范围与 generate_dpo_pairs.py 一致（jsonl / nouns_file / count 范围）。

用法示例：
  python data/verify_dpo_pairs.py \
      --jsonl data/train_triplets/counting_triplets_minimal_origin.jsonl \
      --nouns_file data/train_triplets/counting_nouns.txt \
      --outdir data/dpo_edit_images \
      --min_count 1 --max_count 5 --n_edits 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional


INT_TO_WORD = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}


def sanitize(text: str, maxlen: int = 80) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:maxlen]


def load_nouns_from_txt(txt_path: str) -> list[str]:
    nouns: list[str] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            noun = line.strip().lower()
            if noun:
                nouns.append(noun)
    return nouns


def load_nouns_from_jsonl(jsonl_path: str) -> list[str]:
    nouns: set[str] = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("task") != "counting":
                continue
            noun = str(obj.get("noun", "")).strip().lower()
            if noun:
                nouns.add(noun)
    return sorted(nouns)


def load_expected_nouns(args) -> list[str]:
    if args.nouns_file:
        nouns_file = Path(args.nouns_file)
        if not nouns_file.exists():
            print(f"[ERROR] nouns_file not found: {nouns_file}", file=sys.stderr)
            sys.exit(1)
        return load_nouns_from_txt(str(nouns_file))

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        print(f"[ERROR] jsonl not found: {jsonl_path}", file=sys.stderr)
        sys.exit(1)
    return load_nouns_from_jsonl(str(jsonl_path))


def expected_images_for_count_dir(count_dir: Path, default_n_edits: int) -> tuple[int, Optional[str]]:
    """
    返回该 count 目录应有的图片数量。
    优先读 meta.json 的 n_edits（可识别 source=copy 时 n_edits=1），
    若缺失或异常则回退 default_n_edits。
    """
    meta_path = count_dir / "meta.json"
    if not meta_path.exists():
        return default_n_edits, "missing_meta"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return default_n_edits, "bad_meta"

    n_edits = meta.get("n_edits")
    if isinstance(n_edits, int) and n_edits > 0:
        return n_edits, None
    return default_n_edits, "invalid_meta_n_edits"


def verdict_covered_images(count_dir: Path, image_names: set[str]) -> tuple[bool, str]:
    """
    检查 verdict.json 是否覆盖目录下全部 seed 图片。
    """
    verdict_path = count_dir / "verdict.json"
    if not verdict_path.exists():
        return False, "missing_verdict"

    try:
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
    except Exception:
        return False, "bad_verdict"

    results = verdict.get("results", [])
    if not isinstance(results, list):
        return False, "bad_verdict_results"

    result_images: set[str] = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        img = item.get("image")
        if isinstance(img, str) and img:
            result_images.add(img)

    missing = sorted(image_names - result_images)
    if missing:
        return False, f"verdict_missing_images:{','.join(missing[:5])}"

    return True, ""


def parse_args():
    p = argparse.ArgumentParser(description="Verify DPO pair generation and VLM verification status.")
    p.add_argument(
        "--jsonl",
        type=str,
        default="data/train_triplets/counting_triplets_minimal_origin.jsonl",
        help="counting JSONL（当不提供 nouns_file 时用于提取 noun）",
    )
    p.add_argument(
        "--nouns_file",
        type=str,
        default="data/train_triplets/counting_nouns.txt",
        help="名词列表 txt（每行一个 noun）；提供时以该文件为准",
    )
    p.add_argument("--outdir", type=str, default="data/dpo_edit_images", help="DPO 编辑图输出目录")
    p.add_argument("--min_count", type=int, default=1, help="数量下界（含）")
    p.add_argument("--max_count", type=int, default=5, help="数量上界（含）")
    p.add_argument("--n_edits", type=int, default=3, help="默认每个 count 期望图片数")
    p.add_argument("--max_report", type=int, default=30, help="每类问题最多打印多少条")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir)

    if args.min_count > args.max_count:
        print("[ERROR] min_count should be <= max_count", file=sys.stderr)
        sys.exit(1)
    if args.n_edits <= 0:
        print("[ERROR] n_edits should be > 0", file=sys.stderr)
        sys.exit(1)
    if not outdir.exists():
        print(f"[ERROR] outdir not found: {outdir}", file=sys.stderr)
        sys.exit(1)

    nouns = load_expected_nouns(args)
    counts = list(range(args.min_count, args.max_count + 1))
    total_pairs = len(nouns) * len(counts)

    generation_issues: list[str] = []
    verification_issues: list[str] = []
    missing_meta_count = 0
    bad_meta_count = 0

    for noun in nouns:
        noun_dir = outdir / sanitize(noun)
        for count in counts:
            count_word = INT_TO_WORD.get(count, str(count))
            count_dir = noun_dir / count_word
            pair_key = f"{noun} | {count_word}"

            if not count_dir.exists():
                generation_issues.append(f"{pair_key} -> missing_dir")
                verification_issues.append(f"{pair_key} -> no_images_to_verify")
                continue

            pngs = sorted(count_dir.glob("seed*.png"))
            expected, meta_hint = expected_images_for_count_dir(count_dir, args.n_edits)
            if meta_hint == "missing_meta":
                missing_meta_count += 1
            elif meta_hint in ("bad_meta", "invalid_meta_n_edits"):
                bad_meta_count += 1

            if len(pngs) < expected:
                generation_issues.append(
                    f"{pair_key} -> images={len(pngs)} < expected={expected}"
                )

            if not pngs:
                verification_issues.append(f"{pair_key} -> no_images_to_verify")
                continue

            image_names = {p.name for p in pngs}
            verified, reason = verdict_covered_images(count_dir, image_names)
            if not verified:
                verification_issues.append(f"{pair_key} -> {reason}")

    generated_ok = len(generation_issues) == 0
    verified_ok = len(verification_issues) == 0

    print("\n[CHECK 1] DPO pairs 是否都生成完")
    print(f"  expected_pairs      : {total_pairs}")
    print(f"  generation_issues   : {len(generation_issues)}")
    print(f"  missing_meta_count  : {missing_meta_count}")
    print(f"  bad_meta_count      : {bad_meta_count}")
    print(f"  all_generated       : {'YES' if generated_ok else 'NO'}")

    if generation_issues:
        print(f"  --- examples (max {args.max_report}) ---")
        for line in generation_issues[: args.max_report]:
            print(f"  - {line}")

    print("\n[CHECK 2] DPO pairs 是否都经过验证")
    print(f"  expected_pairs      : {total_pairs}")
    print(f"  verification_issues : {len(verification_issues)}")
    print(f"  all_verified        : {'YES' if verified_ok else 'NO'}")
    if verification_issues:
        print(f"  --- examples (max {args.max_report}) ---")
        for line in verification_issues[: args.max_report]:
            print(f"  - {line}")

    if generated_ok and verified_ok:
        print("\n[RESULT] PASS: 全部 DPO pairs 已生成并完成验证。")
        sys.exit(0)

    print("\n[RESULT] FAIL: 存在未生成完整或未验证完成的 DPO pairs。")
    sys.exit(1)


if __name__ == "__main__":
    main()
