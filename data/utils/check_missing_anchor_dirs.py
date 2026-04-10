"""
check_missing_anchor_dirs.py
============================
检查某个 task（默认 counting）在 generated_images 下是否包含所有应有的 anchor 文件夹。

规则与 generate_triplet_images.py 保持一致：
  - 从 train_triplets/*.jsonl 读取 (task, anchor)
  - 使用 sanitize(anchor) 映射为输出目录名
  - 期望目录：<generated_root>/<task>/<sanitized_anchor>/

示例：
  python data/check_missing_anchor_dirs.py
  python data/check_missing_anchor_dirs.py --task counting
  python data/check_missing_anchor_dirs.py --generated_root data/generated_images
"""

import argparse
import json
import re
from pathlib import Path


def sanitize(text: str, maxlen: int = 80) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"[\s]+", "_", s).strip("_")
    return s[:maxlen]


def load_expected_anchors(triplet_dir: Path, task: str) -> set[str]:
    expected: set[str] = set()
    for jsonl_file in sorted(triplet_dir.glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                row_task = obj.get("task", jsonl_file.stem)
                anchor = obj.get("anchor", "").strip()
                if row_task != task or not anchor:
                    continue
                expected.add(sanitize(anchor))
    return expected


def list_existing_dirs(task_dir: Path) -> set[str]:
    if not task_dir.exists():
        return set()
    return {p.name for p in task_dir.iterdir() if p.is_dir()}


def main(opt: argparse.Namespace) -> None:
    triplet_dir = Path(opt.triplet_dir)
    generated_root = Path(opt.generated_root)
    task_dir = generated_root / opt.task

    expected = load_expected_anchors(triplet_dir, opt.task)
    existing = list_existing_dirs(task_dir)

    missing = sorted(expected - existing)
    extra = sorted(existing - expected)

    print("========================================")
    print(f"Task           : {opt.task}")
    print(f"Triplet dir    : {triplet_dir}")
    print(f"Generated root : {generated_root}")
    print(f"Task dir       : {task_dir}")
    print("========================================")

    if not task_dir.exists():
        print(f"[WARN] 目录不存在: {task_dir}")
        print("       按当前规则，所有期望 anchor 文件夹都视为缺失。")

    print(f"Expected anchor dirs : {len(expected)}")
    print(f"Existing anchor dirs : {len(existing)}")
    print(f"Missing anchor dirs  : {len(missing)}")
    print(f"Extra anchor dirs    : {len(extra)}")

    if missing:
        print("\n--- Missing (first 50) ---")
        for name in missing[:50]:
            print(name)
        if len(missing) > 50:
            print(f"... ({len(missing) - 50} more)")

    if extra and opt.show_extra:
        print("\n--- Extra (first 50) ---")
        for name in extra[:50]:
            print(name)
        if len(extra) > 50:
            print(f"... ({len(extra) - 50} more)")

    if opt.save_missing:
        out_path = Path(opt.save_missing)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for name in missing:
                f.write(name + "\n")
        print(f"\nSaved missing list to: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check missing generated anchor directories")
    p.add_argument("--task", type=str, default="counting")
    p.add_argument("--triplet_dir", type=str, default="data/train_triplets")
    p.add_argument("--generated_root", type=str, default="data/generated_images")
    p.add_argument("--show_extra", action="store_true", help="显示多余目录")
    p.add_argument("--save_missing", type=str, default="", help="把缺失目录名写入文件")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

