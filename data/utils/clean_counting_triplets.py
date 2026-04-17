"""
clean_counting_triplets.py
==========================
数据清洗脚本，针对 counting_triplets_filtered.jsonl：

  (1) 检查每个 anchor prompt 对应的图片目录是否存在，以及该目录下是否有
      >= 1 张 score > threshold 的图片（通过 verdict.json 判断）。

  (2) 从 anchor 中解析出名词部分（格式为 "数量词 + noun"），写入字段 "noun"。

  (3) 只保留通过检查的条目，写出到新的 JSONL 文件，同时打印统计信息。

输出字段（在原有字段基础上新增）：
  - noun            : anchor 中去掉数量词后的名词短语
  - valid_image_count : 通过 score > threshold 的图片数量

用法：
  python data/clean_counting_triplets.py \
      --triplets_jsonl data/train_triplets/counting_triplets_filtered.jsonl \
      --generated_root data/generated_images \
      --output_jsonl   data/train_triplets/counting_triplets_clean.jsonl \
      --threshold      0.8

注：该脚本以 anchor 为粒度去重，同一 anchor 的多条 triplet 行会合并后
   整体检查，只要 anchor 有效，所有对应的 triplet 行都被保留。
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# ── 数量词表（与 filter_triplet_images.py 保持一致） ────────────────────────────
NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve",
}


def sanitize(text: str, maxlen: int = 80) -> str:
    """与训练脚本中 sanitize() 完全一致，用于得到目录名。"""
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"[\s]+", "_", s).strip("_")
    return s[:maxlen]


def extract_noun(anchor: str, target_word: str) -> str:
    """
    从 anchor 中去掉开头的数量词，返回名词短语。
    例：anchor="three cat trees", target_word="three" → "cat trees"
    """
    tw = target_word.strip().lower()
    text = anchor.strip()
    # 忽略大小写地去掉开头的数量词（含后面的空格）
    pattern = re.compile(r"^" + re.escape(tw) + r"\s+", re.IGNORECASE)
    noun = pattern.sub("", text).strip()
    return noun if noun else text  # 如果剥离失败，回退到完整 anchor


def count_valid_images(sample_dir: Path, threshold: float) -> int:
    """
    读取 verdict.json，返回 score > threshold 的图片数量。
    目录不存在 / verdict.json 不存在 / 解析失败 → 返回 0。
    """
    if not sample_dir.exists():
        return 0

    verdict_path = sample_dir / "verdict.json"
    if not verdict_path.exists():
        return 0

    try:
        with open(verdict_path, "r", encoding="utf-8") as f:
            verdict = json.load(f)
    except Exception:
        return 0

    valid_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    # 先建立 image_name → score 映射
    score_by_name: dict[str, float] = {}
    for item in verdict.get("results", []):
        if not isinstance(item, dict):
            continue
        image_name = item.get("image")
        score = item.get("score")
        if isinstance(image_name, str) and isinstance(score, (int, float)):
            score_by_name[image_name] = float(score)

    if not score_by_name:
        return 0

    count = 0
    for p in sample_dir.iterdir():
        if not p.is_file() or p.suffix.lower() not in valid_suffixes:
            continue
        s = score_by_name.get(p.name)
        if s is not None and s > threshold:
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Clean counting triplets JSONL.")
    parser.add_argument(
        "--triplets_jsonl",
        default="data/train_triplets/counting_triplets_filtered.jsonl",
        help="输入 JSONL 文件路径",
    )
    parser.add_argument(
        "--generated_root",
        default="data/generated_images",
        help="生成图片的根目录（含 counting/<anchor>/ 子目录）",
    )
    parser.add_argument(
        "--output_jsonl",
        default="data/train_triplets/counting_triplets_clean.jsonl",
        help="输出 JSONL 文件路径",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="VQA score 阈值（严格大于），默认 0.8",
    )
    args = parser.parse_args()

    triplets_path = Path(args.triplets_jsonl)
    generated_root = Path(args.generated_root)
    output_path = Path(args.output_jsonl)
    threshold = args.threshold

    if not triplets_path.exists():
        print(f"[ERROR] Input file not found: {triplets_path}", file=sys.stderr)
        sys.exit(1)

    # ── 读取全部 triplet 行 ────────────────────────────────────────────────────
    rows: list[dict] = []
    with open(triplets_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Line {lineno}: JSON parse error — {e}", file=sys.stderr)
                continue
            if obj.get("task", "counting") != "counting":
                continue
            if not all(k in obj for k in ("anchor", "target_word")):
                print(f"[WARN] Line {lineno}: missing required fields, skipped.", file=sys.stderr)
                continue
            rows.append(obj)

    print(f"[INFO] Total rows loaded      : {len(rows)}")

    # ── 按 anchor 去重，收集 target_word ──────────────────────────────────────
    # 同一 anchor 可能出现在多行，target_word 应该一致
    anchor_to_tw: dict[str, str] = {}
    for obj in rows:
        anchor = obj["anchor"].strip()
        tw = obj["target_word"].strip().lower()
        if anchor not in anchor_to_tw:
            anchor_to_tw[anchor] = tw
        elif anchor_to_tw[anchor] != tw:
            print(
                f"[WARN] anchor '{anchor}' has conflicting target_words: "
                f"'{anchor_to_tw[anchor]}' vs '{tw}'. Keeping first.",
                file=sys.stderr,
            )

    unique_anchors = len(anchor_to_tw)
    print(f"[INFO] Unique anchors         : {unique_anchors}")

    # ── 逐 anchor 检查目录和图片质量 ──────────────────────────────────────────
    anchor_valid_count: dict[str, int] = {}   # anchor → valid image count
    no_dir = 0
    no_verdict = 0
    no_valid_img = 0
    valid_anchors = 0

    for anchor, tw in anchor_to_tw.items():
        sample_dir = generated_root / "counting" / sanitize(anchor)
        n = count_valid_images(sample_dir, threshold)
        anchor_valid_count[anchor] = n

        if not sample_dir.exists():
            no_dir += 1
        elif not (sample_dir / "verdict.json").exists():
            no_verdict += 1
        elif n == 0:
            no_valid_img += 1
        else:
            valid_anchors += 1

    print(f"\n[STATS] Anchor检查结果（threshold={threshold}）：")
    print(f"  ✓ 有效 anchor（目录存在 + ≥1 张高分图）: {valid_anchors}")
    print(f"  ✗ 目录不存在                          : {no_dir}")
    print(f"  ✗ 目录存在但无 verdict.json            : {no_verdict}")
    print(f"  ✗ verdict 存在但无高分图片             : {no_valid_img}")

    # ── 对每条 triplet 行增加字段并过滤 ────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for obj in rows:
            anchor = obj["anchor"].strip()
            tw = anchor_to_tw[anchor]
            n_valid = anchor_valid_count.get(anchor, 0)

            if n_valid == 0:
                skipped += 1
                continue

            # 解析名词
            noun = extract_noun(anchor, tw)

            out_obj = dict(obj)               # 保留原有所有字段
            out_obj["noun"] = noun            # 新增：名词短语
            out_obj["valid_image_count"] = n_valid  # 新增：高分图片数

            out_f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            kept += 1

    print(f"\n[RESULT] 输入 triplet 行: {len(rows)}")
    print(f"         保留行        : {kept}")
    print(f"         过滤掉行      : {skipped}")
    print(f"         输出文件      : {output_path}")

    # ── 额外：对 noun 和 target_word 做分布汇总，方便排查 ─────────────────────
    noun_to_numbers: dict[str, set] = defaultdict(set)
    number_to_nouns: dict[str, set] = defaultdict(set)
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            noun_to_numbers[obj["noun"]].add(obj["target_word"])
            number_to_nouns[obj["target_word"]].add(obj["noun"])

    print(f"\n[STATS] 名词种类数: {len(noun_to_numbers)}")
    print(f"[STATS] 数量词种类数: {len(number_to_nouns)}")
    # 打印每个数量词对应的 anchor 数量
    for num in sorted(number_to_nouns, key=lambda x: list(NUMBER_WORDS).index(x) if x in NUMBER_WORDS else 99):
        print(f"  {num:8s}: {len(number_to_nouns[num])} nouns")


if __name__ == "__main__":
    main()
