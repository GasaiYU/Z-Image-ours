"""
Normalize counting triplets and image folders.

Goals:
1) Merge prompt variants like "two apple" and "two apples".
2) Remove non-compliant images (by verdict pass flag and/or corrupted files).

Usage:
  python data/normalize_counting_data.py \
    --input_jsonl data/train_triplets/counting_triplets_filtered.jsonl \
    --output_jsonl data/train_triplets/counting_triplets_normalized.jsonl \
    --generated_root data/generated_images \
    --dry_run

Then apply:
  python data/normalize_counting_data.py ... --apply
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


NUMBER_WORD_TO_INT = {
    "a": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def sanitize(text: str, maxlen: int = 80) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"[\s]+", "_", s).strip("_")
    return s[:maxlen]


def strip_punct(s: str) -> str:
    return re.sub(r"[\.,;:!?]+$", "", s.strip().lower())


def singularize_word(word: str) -> str:
    w = word.lower()
    if len(w) <= 2:
        return w
    if w.endswith("ies") and len(w) > 3:
        return w[:-3] + "y"
    if w.endswith("ves") and len(w) > 3:
        # rough heuristic: knives->knife, leaves->leaf
        if w[-4] == "i":
            return w[:-3] + "fe"
        return w[:-3] + "f"
    if re.search(r"(ches|shes|ses|xes|zes)$", w):
        return re.sub(r"es$", "", w)
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def pluralize_word(word: str) -> str:
    w = word.lower()
    if len(w) <= 2:
        return w
    if re.search(r"[^aeiou]y$", w):
        return w[:-1] + "ies"
    if re.search(r"(s|x|z|ch|sh)$", w):
        return w + "es"
    if w.endswith("fe"):
        return w[:-2] + "ves"
    if w.endswith("f"):
        return w[:-1] + "ves"
    return w + "s"


def normalize_noun_phrase(noun_phrase: str, number_value: int) -> str:
    phrase = strip_punct(noun_phrase)
    if not phrase:
        return phrase

    # Handle "cup of tea" style: pluralize/singularize head noun.
    if " of " in phrase:
        head, tail = phrase.split(" of ", 1)
        head_tokens = head.split()
        if head_tokens:
            last = head_tokens[-1]
            head_tokens[-1] = singularize_word(last)
            if number_value != 1:
                head_tokens[-1] = pluralize_word(head_tokens[-1])
        norm_head = " ".join(head_tokens)
        return f"{norm_head} of {tail}"

    tokens = phrase.split()
    if not tokens:
        return phrase

    tokens[-1] = singularize_word(tokens[-1])
    if number_value != 1:
        tokens[-1] = pluralize_word(tokens[-1])
    return " ".join(tokens)


def normalize_counting_prompt(prompt: str) -> tuple[str, str]:
    """
    Returns (normalized_prompt, normalized_noun_phrase).
    If prompt does not start with a known number word, returns cleaned prompt.
    """
    p = strip_punct(prompt)
    tokens = p.split()
    if len(tokens) < 2:
        return p, ""

    nword = tokens[0]
    nval = NUMBER_WORD_TO_INT.get(nword)
    if nval is None:
        return p, " ".join(tokens[1:])

    noun_phrase = " ".join(tokens[1:])
    noun_norm = normalize_noun_phrase(noun_phrase, nval)
    return f"{nword} {noun_norm}".strip(), noun_norm


def is_image_readable(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def load_verdict_pass_map(folder: Path) -> dict[str, bool]:
    verdict_path = folder / "verdict.json"
    if not verdict_path.exists():
        return {}
    try:
        obj = json.loads(verdict_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    res = {}
    for item in obj.get("results", []):
        name = item.get("image")
        passed = bool(item.get("pass", False))
        if name:
            res[name] = passed
    return res


def gather_compliant_images(folder: Path, require_verdict_pass: bool) -> tuple[list[Path], list[Path]]:
    """
    Returns (kept_images, dropped_images).
    """
    all_images = [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    ]

    verdict_map = load_verdict_pass_map(folder) if require_verdict_pass else {}
    kept: list[Path] = []
    dropped: list[Path] = []

    for img in all_images:
        if require_verdict_pass and verdict_map and not verdict_map.get(img.name, False):
            dropped.append(img)
            continue
        if not is_image_readable(img):
            dropped.append(img)
            continue
        kept.append(img)
    return kept, dropped


def ensure_unique_copy(src: Path, dst_dir: Path) -> Path:
    dst = dst_dir / src.name
    if not dst.exists():
        return dst
    stem, suffix = src.stem, src.suffix
    idx = 1
    while True:
        cand = dst_dir / f"{stem}_m{idx}{suffix}"
        if not cand.exists():
            return cand
        idx += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", type=str, required=True)
    ap.add_argument("--output_jsonl", type=str, required=True)
    ap.add_argument("--generated_root", type=str, default="data/generated_images")
    ap.add_argument("--task", type=str, default="counting")
    ap.add_argument("--min_valid_images", type=int, default=1)
    ap.add_argument("--merge_by_anchor", action="store_true",
                    help="Merge multiple rows with the same normalized anchor into one row.")
    ap.add_argument("--max_positive_pool", type=int, default=64,
                    help="Max number of positives kept in positive_pool when --merge_by_anchor is enabled.")
    ap.add_argument("--max_negative_pool", type=int, default=64,
                    help="Max number of negatives kept in negative_pool when --merge_by_anchor is enabled.")
    ap.add_argument("--keep_all_fields", action="store_true",
                    help="Keep full fields in output rows. Default is minimal fields only.")
    ap.add_argument("--require_verdict_pass", action="store_true",
                    help="If set, only keep images with pass=true in verdict.json.")
    ap.add_argument("--prune_source_bad_images", action="store_true",
                    help="Delete dropped images from source folders when --apply.")
    ap.add_argument("--apply", action="store_true",
                    help="Apply filesystem moves/copies. Without this, only write output jsonl/report.")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print summary only; no filesystem changes.")
    args = ap.parse_args()

    if args.apply and args.dry_run:
        raise ValueError("--apply and --dry_run cannot be used together.")

    input_jsonl = Path(args.input_jsonl)
    output_jsonl = Path(args.output_jsonl)
    generated_root = Path(args.generated_root)
    task_root = generated_root / args.task

    rows: list[dict[str, Any]] = []
    with input_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("task") != args.task:
                continue
            rows.append(obj)

    # Normalize prompts and build mapping old_anchor -> normalized_anchor
    old_to_new_anchor: dict[str, str] = {}
    normalized_rows: list[dict[str, Any]] = []
    for r in rows:
        anchor_norm, noun_norm = normalize_counting_prompt(r.get("anchor", ""))
        pos_norm, _ = normalize_counting_prompt(r.get("positive", ""))
        neg_norm, _ = normalize_counting_prompt(r.get("negative", ""))

        out = dict(r)
        out["anchor"] = anchor_norm
        out["positive"] = pos_norm
        out["negative"] = neg_norm
        if noun_norm:
            out["noun"] = noun_norm

        old_anchor = strip_punct(r.get("anchor", ""))
        old_to_new_anchor[old_anchor] = anchor_norm
        normalized_rows.append(out)

    # Merge duplicate triplets after normalization
    dedup = {}
    for r in normalized_rows:
        key = (
            r.get("task", ""),
            r.get("target_word", ""),
            r.get("anchor", ""),
            r.get("positive", ""),
            r.get("negative", ""),
        )
        dedup[key] = r
    normalized_rows = list(dedup.values())

    # Build canonical anchor -> source dirs
    anchor_to_source_dirs: dict[str, list[Path]] = defaultdict(list)
    for old_anchor, new_anchor in old_to_new_anchor.items():
        src_dir = task_root / sanitize(old_anchor)
        if src_dir.exists():
            anchor_to_source_dirs[new_anchor].append(src_dir)

    # Prepare destination dirs and image counts
    anchor_valid_count: dict[str, int] = {}
    merge_report = []
    for new_anchor, src_dirs in anchor_to_source_dirs.items():
        dst_dir = task_root / sanitize(new_anchor)
        kept_total = 0
        dropped_total = 0
        moved_total = 0

        if args.apply:
            dst_dir.mkdir(parents=True, exist_ok=True)

        for src_dir in src_dirs:
            kept, dropped = gather_compliant_images(src_dir, require_verdict_pass=args.require_verdict_pass)
            kept_total += len(kept)
            dropped_total += len(dropped)

            if args.apply:
                for img in kept:
                    target = ensure_unique_copy(img, dst_dir)
                    if img.resolve() != target.resolve():
                        shutil.copy2(img, target)
                        moved_total += 1
                if args.prune_source_bad_images:
                    for bad in dropped:
                        try:
                            bad.unlink()
                        except Exception:
                            pass

        anchor_valid_count[new_anchor] = kept_total
        merge_report.append({
            "anchor": new_anchor,
            "source_dirs": [str(p) for p in src_dirs],
            "kept_images": kept_total,
            "dropped_images": dropped_total,
            "moved_images": moved_total,
        })

    # Recompute valid_image_count and filter rows by min image count
    if args.merge_by_anchor:
        grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
        for r in normalized_rows:
            task = r.get("task", "")
            target_word = r.get("target_word", "")
            anchor = r.get("anchor", "")
            count = anchor_valid_count.get(anchor, 0)
            if count < args.min_valid_images:
                continue

            key = (task, target_word, anchor)
            g = grouped.get(key)
            if g is None:
                g = {
                    "task": task,
                    "target_word": target_word,
                    "anchor": anchor,
                    "noun": r.get("noun", ""),
                    "positive_pool": [],
                    "negative_pool": [],
                    "_pos_set": set(),
                    "_neg_set": set(),
                    "valid_image_count": count,
                    "num_merged_rows": 0,
                }
                grouped[key] = g

            g["num_merged_rows"] += 1

            p = r.get("positive", "")
            if p and p not in g["_pos_set"] and len(g["positive_pool"]) < args.max_positive_pool:
                g["positive_pool"].append(p)
                g["_pos_set"].add(p)

            n = r.get("negative", "")
            if n and n not in g["_neg_set"] and len(g["negative_pool"]) < args.max_negative_pool:
                g["negative_pool"].append(n)
                g["_neg_set"].add(n)

        final_rows: list[dict[str, Any]] = []
        for g in grouped.values():
            # keep compatibility with old consumers that expect single positive/negative fields
            g["positive"] = g["positive_pool"][0] if g["positive_pool"] else ""
            g["negative"] = g["negative_pool"][0] if g["negative_pool"] else ""
            g.pop("_pos_set", None)
            g.pop("_neg_set", None)
            if args.keep_all_fields:
                final_rows.append(g)
            else:
                final_rows.append(
                    {
                        "task": g["task"],
                        "target_word": g["target_word"],
                        "anchor": g["anchor"],
                        "noun": g.get("noun", ""),
                        "valid_image_count": g["valid_image_count"],
                        "num_merged_rows": g["num_merged_rows"],
                    }
                )
    else:
        final_rows = []
        for r in normalized_rows:
            anchor = r.get("anchor", "")
            count = anchor_valid_count.get(anchor, 0)
            if count < args.min_valid_images:
                continue
            if args.keep_all_fields:
                rr = dict(r)
                rr["valid_image_count"] = count
                final_rows.append(rr)
            else:
                final_rows.append(
                    {
                        "task": r.get("task", ""),
                        "target_word": r.get("target_word", ""),
                        "anchor": r.get("anchor", ""),
                        "noun": r.get("noun", ""),
                        "valid_image_count": count,
                    }
                )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for r in final_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "input_rows": len(rows),
        "normalized_rows_before_filter": len(normalized_rows),
        "final_rows": len(final_rows),
        "unique_old_anchors": len(old_to_new_anchor),
        "unique_new_anchors": len({v for v in old_to_new_anchor.values()}),
        "apply": args.apply,
        "require_verdict_pass": args.require_verdict_pass,
        "min_valid_images": args.min_valid_images,
        "merge_by_anchor": args.merge_by_anchor,
        "keep_all_fields": args.keep_all_fields,
        "merge_report_head": merge_report[:20],
    }
    report_path = output_jsonl.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== normalization done ===")
    print(f"input rows: {len(rows)}")
    print(f"rows after normalize+dedup: {len(normalized_rows)}")
    print(f"rows after image filter: {len(final_rows)}")
    print(f"output jsonl: {output_jsonl}")
    print(f"report: {report_path}")
    if not args.apply:
        print("Note: filesystem not modified (use --apply to merge/copy images).")


if __name__ == "__main__":
    main()

