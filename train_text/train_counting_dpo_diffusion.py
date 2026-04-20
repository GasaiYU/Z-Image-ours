import argparse
import json
import math
import os
import random
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from utils import load_from_local_dir  # noqa: E402
from zimage.pipeline import generate as pipeline_generate  # noqa: E402


def sanitize(text: str, maxlen: int = 80) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"[\s]+", "_", s).strip("_")
    return s[:maxlen]


def format_prompt(tokenizer, text: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return text
    messages = [{"role": "user", "content": text}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def find_target_idx(tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor, target_word: str) -> int:
    valid_ids = input_ids[attention_mask.bool()].tolist()
    tw = target_word.lower().strip()
    for idx, tid in enumerate(valid_ids):
        token_str = tokenizer.decode([tid], skip_special_tokens=True).lower().strip()
        if tw in token_str or token_str in tw:
            return idx
    return -1


def extract_token_feature(hidden_states: torch.Tensor, token_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    feats = []
    valid = []
    seq_len = hidden_states.shape[1]
    for i, tidx in enumerate(token_indices):
        if 0 <= tidx < seq_len:
            feats.append(F.normalize(hidden_states[i, tidx, :], dim=-1))
            valid.append(True)
        else:
            feats.append(torch.zeros_like(hidden_states[i, 0, :]))
            valid.append(False)
    return torch.stack(feats, dim=0), torch.tensor(valid, device=hidden_states.device, dtype=torch.bool)


def chunked_encode_and_extract(
    text_encoder,
    transformer: torch.nn.Module,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    chunk_size: int,
    source_mode: str,
    source_layer_idx: int,
    source_range_start: int,
    source_range_end: int,
    target_words: list[str] = None,
    target_token_weight: float = 1.0,
) -> torch.Tensor:
    special_ids = set(tokenizer.all_special_ids)
    all_feats: list[torch.Tensor] = []
    N = input_ids.shape[0]

    for i in range(0, N, chunk_size):
        ids  = input_ids[i : i + chunk_size]
        mask = attention_mask[i : i + chunk_size]

        is_special = torch.zeros_like(ids, dtype=torch.bool)
        for sid in special_ids:
            is_special |= (ids == sid)
        content_mask = mask.bool() & ~is_special
        float_mask = content_mask.float()

        if target_words is not None and target_token_weight != 1.0:
            chunk_tws = target_words[i : i + chunk_size]
            tidx = [find_target_idx(tokenizer, ids[j], mask[j], chunk_tws[j]) for j in range(len(ids))]
            for j, idx in enumerate(tidx):
                if idx >= 0 and content_mask[j, idx]:
                    float_mask[j, idx] = target_token_weight

        with torch.no_grad():
            out = text_encoder(input_ids=ids, attention_mask=mask.bool(), output_hidden_states=True)
        h = select_text_source_hidden(
            out.hidden_states,
            source_mode=source_mode,
            layer_idx=source_layer_idx,
            range_start=source_range_start,
            range_end=source_range_end,
        ).detach().clone()
        del out

        h_ref = run_context_refiner(transformer, h, mask)
        del h

        float_mask = float_mask.unsqueeze(-1)
        denom = float_mask.sum(dim=1).clamp(min=1e-6)
        pooled = (h_ref * float_mask).sum(dim=1) / denom
        pooled = F.normalize(pooled.float(), dim=-1)
        del h_ref

        all_feats.append(pooled)

    return torch.cat(all_feats, dim=0)


def select_text_source_hidden(
    hidden_states: tuple[torch.Tensor, ...],
    source_mode: str,
    layer_idx: int,
    range_start: int,
    range_end: int,
) -> torch.Tensor:
    if source_mode == "layer":
        idx = layer_idx
        if idx < 0:
            idx = len(hidden_states) + idx
        idx = max(0, min(idx, len(hidden_states) - 1))
        return hidden_states[idx]
    if source_mode == "avg_range":
        s = max(0, range_start)
        e = min(len(hidden_states) - 1, range_end)
        if e < s:
            raise ValueError(f"Invalid text source range [{range_start}, {range_end}]")
        picked = hidden_states[s : e + 1]
        if len(picked) == 0:
            raise ValueError(f"Empty text source range [{range_start}, {range_end}]")
        stacked = torch.stack([h.float() for h in picked], dim=0)
        return stacked.mean(dim=0).to(hidden_states[s].dtype)
    raise ValueError(f"Unknown source_mode: {source_mode}")


def zscore_then_l2(
    ea: torch.Tensor,
    ep: torch.Tensor,
    en_flat: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    merged = torch.cat([ea.float(), ep.float(), en_flat.float()], dim=0)
    
    mu = merged.mean(dim=0, keepdim=True).detach()
    sigma = merged.std(dim=0, keepdim=True, unbiased=False).detach()
    
    merged = (merged - mu) / (sigma + eps)
    merged = F.normalize(merged, dim=-1)
    b = ea.shape[0]
    return merged[:b], merged[b : 2 * b], merged[2 * b :]


def infonce_loss(ea: torch.Tensor, ep: torch.Tensor, en: torch.Tensor, temperature: float) -> torch.Tensor:
    ea, ep, en = ea.float(), ep.float(), en.float()
    pos_logits = (ea * ep).sum(dim=-1, keepdim=True)
    neg_logits = (ea.unsqueeze(1) * en).sum(dim=-1)
    logits = torch.cat([pos_logits, neg_logits], dim=1) / temperature
    labels = torch.zeros(ea.shape[0], dtype=torch.long, device=ea.device)
    return F.cross_entropy(logits, labels)


def dcl_loss(ea: torch.Tensor, ep: torch.Tensor, en: torch.Tensor, temperature: float) -> torch.Tensor:
    ea, ep, en = ea.float(), ep.float(), en.float()
    pos_logits = (ea * ep).sum(dim=-1) / temperature
    neg_logits = (ea.unsqueeze(1) * en).sum(dim=-1) / temperature
    return (-pos_logits + torch.logsumexp(neg_logits, dim=-1)).mean()


def unfreeze_transformer_refiner_layers(transformer: torch.nn.Module) -> list[str]:
    for p in transformer.parameters():
        p.requires_grad_(False)

    trainable_names: list[str] = []
    for name, param in transformer.named_parameters():
        if "context_refiner" in name.lower():
            param.requires_grad_(True)
            trainable_names.append(name)

    if not trainable_names:
        raise RuntimeError(
            "No context_refiner parameters found in transformer. "
            "Check the model structure."
        )
    return trainable_names


def run_context_refiner(
    transformer: torch.nn.Module,
    token_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    refiner_override: torch.nn.Module | None = None,
) -> torch.Tensor:
    model = transformer.module if hasattr(transformer, "module") else transformer
    bsz, seq_len, _ = token_hidden.shape
    device = token_hidden.device
    dtype = next(model.parameters()).dtype
    attn_mask = attention_mask.bool()

    cap_feats = model.cap_embedder(token_hidden.to(dtype))
    cap_feats = cap_feats.clone()
    cap_feats[~attn_mask] = model.cap_pad_token.to(dtype)

    pos_ids = torch.zeros((bsz, seq_len, 3), dtype=torch.int32, device=device)
    pos_ids[:, :, 0] = torch.arange(1, seq_len + 1, dtype=torch.int32, device=device).unsqueeze(0).expand(bsz, -1)
    cap_freqs = model.rope_embedder(pos_ids.view(-1, 3)).view(bsz, seq_len, -1)

    refiner_layers = model.context_refiner if refiner_override is None else refiner_override
    refined = cap_feats
    for layer in refiner_layers:
        refined = layer(refined, attn_mask, cap_freqs)
    return refined


def build_cap_feats(
    transformer: torch.nn.Module,
    token_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    target_dtype: torch.dtype,
    refiner_override: torch.nn.Module | None = None,
) -> list[torch.Tensor]:
    refined = run_context_refiner(transformer, token_hidden, attention_mask, refiner_override=refiner_override)
    return [refined[i][attention_mask[i].bool()].to(dtype=target_dtype) for i in range(refined.shape[0])]


def forward_with_refiner_override(
    transformer: torch.nn.Module,
    x: list[torch.Tensor],
    t: torch.Tensor,
    cap_feats: list[torch.Tensor],
    refiner_override: torch.nn.Module | None = None,
) -> list[torch.Tensor]:
    if refiner_override is None:
        return transformer(x, t, cap_feats)[0]

    model = transformer.module if hasattr(transformer, "module") else transformer
    original_refiner = model.context_refiner
    model.context_refiner = refiner_override
    try:
        return transformer(x, t, cap_feats)[0]
    finally:
        model.context_refiner = original_refiner


def make_anchor_variants(anchor: str) -> list[str]:
    base = anchor.strip()
    candidates = [
        base,
        f"a photo of {base}",
        f"an image of {base}",
        f"a picture of {base}",
        f"a realistic photo of {base}",
        f"a high-quality photo of {base}",
        f"{base} on a plain background",
        f"{base}, studio photography",
        f"a detailed image of {base}",
        f"a clear photo of {base}",
    ]
    seen: set[str] = set()
    variants: list[str] = []
    for v in candidates:
        k = v.strip().lower()
        if k and k not in seen:
            variants.append(v)
            seen.add(k)
    return variants


NUMBER_WORDS = ["one", "two", "three", "four", "five"]


def extract_noun(anchor: str, target_word: str) -> str:
    """
    Strip the leading number word from an anchor prompt and return the noun phrase.
    e.g. "three cat trees", "three" -> "cat trees"
    """
    tw = target_word.strip().lower()
    noun = re.sub(r"^" + re.escape(tw) + r"\s+", "", anchor.strip(), flags=re.IGNORECASE).strip()
    return noun if noun else anchor.strip()


def compose_anchor(count_word: str, noun: str) -> str:
    return f"{count_word.strip()} {noun.strip()}".strip()


def make_anchor_negatives(noun: str, target_word: str) -> list[tuple[str, str]]:
    tw = target_word.strip().lower()
    negatives: list[tuple[str, str]] = []
    seen: set[str] = set()
    for num in NUMBER_WORDS:
        if num == tw:
            continue
        neg_base = compose_anchor(num, noun)
        for variant in make_anchor_variants(neg_base):
            key = variant.strip().lower()
            if key not in seen:
                negatives.append((variant, num))
                seen.add(key)
    return negatives


@dataclass
class AnchorPool:
    count_word: str
    anchor: str
    image_paths: list[Path]


@dataclass
class NounEntry:
    noun: str
    pools_by_count: dict[str, AnchorPool]
    available_counts: list[str]


class CountingVerdictDataset(Dataset):
    def __init__(
        self,
        triplets_jsonl: str,
        threshold: float,
        resolution: int,
        num_negatives: int,
    ):
        self.triplets_jsonl = Path(triplets_jsonl)
        self.project_root = Path(__file__).resolve().parents[1]
        self.threshold = threshold
        self.num_negatives = num_negatives

        self.image_tf = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        records = self._load_records()
        self.noun_store, total_rows, unique_anchors = self._build_noun_store(records)
        self.text_nouns = sorted(self.noun_store.keys())
        self.sample_keys = [
            (noun, count_word)
            for noun, entry in self.noun_store.items()
            if len(entry.available_counts) >= 2
            for count_word in entry.available_counts
        ]
        kept_pools = sum(len(entry.available_counts) for entry in self.noun_store.values())

        print(f"[Dataset] Source format: noun_grouped_dpo_index")
        print(f"[Dataset] Total anchor pools: {total_rows}")
        print(f"[Dataset] Unique anchors: {unique_anchors}")
        print(f"[Dataset] Kept with available images: {kept_pools}")
        print(f"[Dataset] Unique nouns: {len(self.noun_store)}")
        print(f"[Dataset] DPO-eligible (noun, count) keys: {len(self.sample_keys)}")

    def _load_records(self) -> list[dict[str, Any]]:
        text = self.triplets_jsonl.read_text(encoding="utf-8")
        stripped = text.lstrip()
        if not stripped:
            return []

        if stripped.startswith("["):
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                return []
            if isinstance(data, list):
                return [obj for obj in data if isinstance(obj, dict)]
            return []

        raise ValueError(
            f"{self.triplets_jsonl} should be a JSON array file produced by build_dpo_training_index.py"
        )

    def _extract_target_word(self, anchor: str) -> str:
        parts = anchor.strip().split()
        return parts[0].lower() if parts else ""

    def _resolve_index_image_paths(self, image_paths: list[str]) -> list[Path]:
        resolved: list[Path] = []
        for image_path in image_paths:
            if not isinstance(image_path, str) or not image_path.strip():
                continue
            p = Path(image_path)
            candidates = [p] if p.is_absolute() else [self.project_root / p]
            for candidate in candidates:
                if candidate.exists():
                    resolved.append(candidate)
                    break
        return sorted(set(resolved))

    def _build_noun_store(
        self,
        records: list[dict[str, Any]],
    ) -> tuple[dict[str, NounEntry], int, int]:
        noun_store: dict[str, dict[str, AnchorPool]] = {}
        total_pools = 0
        unique_anchors: set[str] = set()

        for obj in records:
            if obj.get("task", "counting") != "counting":
                continue
            noun = str(obj.get("noun", "")).strip()
            pools = obj.get("anchor_pools", [])
            if not isinstance(pools, list):
                continue

            for pool in pools:
                if not isinstance(pool, dict):
                    continue
                anchor = str(pool.get("anchor", "")).strip()
                if not anchor:
                    continue
                image_paths = self._resolve_index_image_paths(pool.get("image_paths", []))
                total_pools += 1
                unique_anchors.add(anchor)
                if not image_paths:
                    continue
                tw = self._extract_target_word(anchor)
                if tw not in NUMBER_WORDS:
                    continue
                sample_noun = noun or extract_noun(anchor, tw)
                if not sample_noun:
                    continue

                pools_by_count = noun_store.setdefault(sample_noun, {})
                existing = pools_by_count.get(tw)
                if existing is None:
                    pools_by_count[tw] = AnchorPool(
                        count_word=tw,
                        anchor=anchor,
                        image_paths=image_paths,
                    )
                    continue

                merged_paths = sorted(set(existing.image_paths + image_paths))
                pools_by_count[tw] = AnchorPool(
                    count_word=tw,
                    anchor=existing.anchor or anchor,
                    image_paths=merged_paths,
                )

        finalized: dict[str, NounEntry] = {}
        for noun_key, pools_by_count in noun_store.items():
            available_counts = [count_word for count_word in NUMBER_WORDS if count_word in pools_by_count]
            if not available_counts:
                continue
            finalized[noun_key] = NounEntry(
                noun=noun_key,
                pools_by_count={count_word: pools_by_count[count_word] for count_word in available_counts},
                available_counts=available_counts,
            )
        return finalized, total_pools, len(unique_anchors)


    def __len__(self) -> int:
        return len(self.sample_keys)

    def _get_anchor_text(self, entry: NounEntry, count_word: str) -> str:
        pool = entry.pools_by_count.get(count_word)
        if pool is not None and pool.anchor.strip():
            return pool.anchor
        return compose_anchor(count_word, entry.noun)

    def _sample_negatives(self, noun: str, target_word: str) -> tuple[list[str], list[str]]:
        pool = make_anchor_negatives(noun, target_word)
        if not pool:
            return [], []
        if len(pool) >= self.num_negatives:
            chosen = random.sample(pool, self.num_negatives)
        else:
            chosen = [random.choice(pool) for _ in range(self.num_negatives)]
        return [t for t, _ in chosen], [tw for _, tw in chosen]

    def _sample_anchor_positive(self, anchor: str) -> str:
        pool = [v for v in make_anchor_variants(anchor) if v.strip().lower() != anchor.strip().lower()]
        if not pool:
            return anchor
        return random.choice(pool)

    def _sample_loser(self, noun: str, winner_count: str) -> tuple[Path | None, AnchorPool | None]:
        """
        Sample a loser from the same noun but a different count.
        """
        entry = self.noun_store.get(noun)
        if entry is None:
            return None, None
        candidates = [count_word for count_word in entry.available_counts if count_word != winner_count]
        if candidates:
            loser_count = random.choice(candidates)
            loser_pool = entry.pools_by_count[loser_count]
            return random.choice(loser_pool.image_paths), loser_pool
        return None, None

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        """Returns image samples for DPO/CPO loss (includes pixel_values_w and pixel_values_l).
        Returns None if no same-noun hard negative exists for this anchor (collate_fn will drop it)."""
        noun, count_word = self.sample_keys[idx]
        entry = self.noun_store[noun]
        winner_pool = entry.pools_by_count[count_word]

        # 1. Winner Image: from the current anchor (score > threshold guaranteed)
        winner_path = random.choice(winner_pool.image_paths)
        winner_img = self.image_tf(Image.open(winner_path).convert("RGB"))

        # 2. Loser Image: must be same noun, different count (hard negative).
        #    If no such anchor exists, skip this sample entirely.
        loser_path, _loser_sample = self._sample_loser(noun, count_word)
        if loser_path is None:
            return None
        loser_img = self.image_tf(Image.open(loser_path).convert("RGB"))

        return {
            "anchor": winner_pool.anchor,
            "target_word": winner_pool.count_word,
            "pixel_values_w": winner_img,
            "pixel_values_l": loser_img,
            "pixel_values_w_path": winner_path,
            "pixel_values_l_path": loser_path,
        }

    def sample_text_batch(self, n: int) -> list[dict[str, Any]]:
        if not self.text_nouns:
            return []
        chosen_nouns = random.choices(self.text_nouns, k=n)
        items = []
        for noun in chosen_nouns:
            entry = self.noun_store[noun]
            target_word = random.choice(entry.available_counts)
            anchor = self._get_anchor_text(entry, target_word)
            positive = self._sample_anchor_positive(anchor)
            negatives, neg_target_words = self._sample_negatives(entry.noun, target_word)
            items.append({
                "anchor": anchor,
                "positive": positive,
                "negatives": negatives,
                "neg_target_words": neg_target_words,
                "target_word": target_word,
            })
        return items


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate image batch (for DPO/CPO loss).
    Drops None entries returned by __getitem__ when no hard negative was found."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return {}
    return {
        "anchor": [b["anchor"] for b in batch],
        "target_word": [b["target_word"] for b in batch],
        "pixel_values_w": torch.stack([b["pixel_values_w"] for b in batch], dim=0),
        "pixel_values_l": torch.stack([b["pixel_values_l"] for b in batch], dim=0),
        "pixel_values_w_path": [b["pixel_values_w_path"] for b in batch],
        "pixel_values_l_path": [b["pixel_values_l_path"] for b in batch],
    }


def collate_text_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "anchor": [b["anchor"] for b in items],
        "positive": [b["positive"] for b in items],
        "negatives": [b["negatives"] for b in items],
        "neg_target_words": [b["neg_target_words"] for b in items],
        "target_word": [b["target_word"] for b in items],
    }


def tokenize_texts(tokenizer, texts: list[str], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return enc.input_ids, enc.attention_mask


def main(args: argparse.Namespace) -> None:
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    device = accelerator.device
    is_main = accelerator.is_main_process

    if args.seed is not None:
        set_seed(args.seed)

    use_wandb = args.use_wandb and _WANDB_AVAILABLE and is_main
    if args.use_wandb and not _WANDB_AVAILABLE and is_main:
        print("[WandB] wandb not installed. Install via: pip install wandb")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or None,
            config=vars(args),
        )
        print(f"[WandB] Run: {wandb.run.url}")

    components = load_from_local_dir(
        args.model_dir,
        device="cpu",
        dtype=torch.bfloat16,
        verbose=is_main,
    )
    transformer = components["transformer"].to(device)
    ref_context_refiner = deepcopy(transformer.context_refiner).to(device)
    vae = components["vae"].to(device)
    text_encoder = components["text_encoder"].to(device)
    tokenizer = components["tokenizer"]

    for p in text_encoder.parameters():
        p.requires_grad_(False)
    for p in vae.parameters():
        p.requires_grad_(False)
    ref_context_refiner.requires_grad_(False)
    ref_context_refiner.eval()
    text_encoder.eval()
    vae.eval()

    trainable_names = unfreeze_transformer_refiner_layers(transformer)

    if args.gradient_checkpointing:
        if hasattr(transformer, "enable_gradient_checkpointing"):
            transformer.enable_gradient_checkpointing()
            if is_main:
                print("[Init] Gradient checkpointing enabled.")
        else:
            if is_main:
                print("[Init] Warning: transformer does not support enable_gradient_checkpointing().")

    _raw_transformer = transformer.module if hasattr(transformer, "module") else transformer
    _transformer_dim = _raw_transformer.dim

    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    if is_main:
        n_refiner = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        print(f"[Init] trainable transformer params: {n_refiner:,}")
        print(f"[Init] projection head: disabled")
        print("[Init] reference refiner: frozen snapshot enabled for DPO")
        print(f"[Init] gradient accumulation steps: {args.gradient_accumulation_steps}")
        if args.text_source_mode == "avg_range":
            print(f"[Init] refiner input source: avg layers [{args.text_source_range_start}, {args.text_source_range_end}]")
        else:
            print(f"[Init] refiner input source: layer {args.text_source_layer_idx}")
        print(f"[Init] zscore before contrastive loss: {args.apply_zscore_before_loss} (eps={args.zscore_eps})")
    if args.print_trainable and is_main:
        print("[Init] trainable parameter names:")
        for name in trainable_names[:50]:
            print("  ", name)

    dataset = CountingVerdictDataset(
        triplets_jsonl=args.triplets_jsonl,
        threshold=args.verdict_threshold,
        resolution=args.resolution,
        num_negatives=args.num_negatives,
    )
    if len(dataset) == 0:
        raise RuntimeError("No valid training samples with available images.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(
        [{"params": [p for p in transformer.parameters() if p.requires_grad], "lr": args.refiner_lr}],
        weight_decay=args.weight_decay,
    )

    transformer, optimizer, loader = accelerator.prepare(transformer, optimizer, loader)

    vis_dir = Path(args.output_dir) / "vis"
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)

    VIS_PROMPTS = [
        "five apples on a table",
        "seven birds on a wire",
        "a photo of four computer keyboards",
        "three red flowers in a vase",
    ]

    def visualize(step: int) -> None:
        if not is_main:
            return
        unwrapped = accelerator.unwrap_model(transformer)
        unwrapped.eval()
        
        gen = torch.Generator(device=device)
        if args.seed is not None:
            gen.manual_seed(args.seed + 2026)
            
        images = pipeline_generate(
            transformer=unwrapped,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=components["scheduler"],
            prompt=VIS_PROMPTS,
            height=args.resolution,
            width=args.resolution,
            num_inference_steps=8,
            guidance_scale=0.0,
            generator=gen,
        )
        for i, (img, prompt) in enumerate(zip(images, VIS_PROMPTS)):
            fname = vis_dir / f"step{step:06d}_p{i}.png"
            img.save(fname)
        if use_wandb:
            wandb.log(
                {"vis": [wandb.Image(img, caption=p) for img, p in zip(images, VIS_PROMPTS)]},
                step=step,
            )
        unwrapped.train()
        print(f"[Vis] saved {len(images)} images at step {step} → {vis_dir}")

    global_step = 0
    transformer_dtype = next(transformer.parameters()).dtype

    steps_per_epoch = math.ceil(len(loader) / args.gradient_accumulation_steps) if len(loader) > 0 else 0
    total_steps = args.epochs * steps_per_epoch
    if args.no_ctr_decay:
        ctr_decay_steps = -1
    else:
        ctr_decay_steps = args.ctr_decay_steps if args.ctr_decay_steps > 0 else total_steps
    if is_main:
        if args.no_ctr_decay:
            print(f"[Init] contrastive weight: {args.contrastive_weight:.3f} (fixed, no decay)")
        else:
            print(f"[Init] contrastive weight: {args.contrastive_weight:.3f} → 0.0 over {ctr_decay_steps} steps")
        print(f"[Init] diffusion weight: {args.diffusion_weight:.3f} (fixed)")
        print(f"[Init] DPO beta: {args.beta_dpo:.3f}")

    def get_ctr_weight(step: int) -> float:
        if args.no_ctr_decay:
            return args.contrastive_weight
        if step >= ctr_decay_steps:
            return 0.0
        return args.contrastive_weight * (1.0 - step / ctr_decay_steps)

    if args.vis_every > 0:
        visualize(0)

    for epoch in range(1, args.epochs + 1):
        transformer.train()
        ref_context_refiner.eval()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not is_main)
        running_total = 0.0
        running_dpo = 0.0
        running_ctr = 0.0
        running_implicit_acc = 0.0
        running_policy_mse = 0.0
        running_ref_mse = 0.0
        steps = 0

        for batch in pbar:
            # Skip batches where every sample lacked a same-noun hard negative
            if not batch:
                continue
            with accelerator.accumulate(transformer):
                # ── Contrastive loss (large text-only batch, no images) ──────────────
                if args.contrastive_weight > 0:
                    text_items = dataset.sample_text_batch(args.contrastive_batch_size)
                    text_batch = collate_text_batch(text_items)

                    anchor_texts = [format_prompt(tokenizer, t, args.use_chat_template) for t in text_batch["anchor"]]
                    pos_texts = [format_prompt(tokenizer, t, args.use_chat_template) for t in text_batch["positive"]]
                    neg_texts_nested = [
                        [format_prompt(tokenizer, t, args.use_chat_template) for t in negs]
                        for negs in text_batch["negatives"]
                    ]
                    flat_neg_texts = [x for negs in neg_texts_nested for x in negs]

                    a_ids, a_mask = tokenize_texts(tokenizer, anchor_texts, args.max_length)
                    p_ids, p_mask = tokenize_texts(tokenizer, pos_texts, args.max_length)
                    n_ids, n_mask = tokenize_texts(tokenizer, flat_neg_texts, args.max_length)

                    a_ids = a_ids.to(device);  a_mask = a_mask.to(device)
                    p_ids = p_ids.to(device);  p_mask = p_mask.to(device)
                    n_ids = n_ids.to(device);  n_mask = n_mask.to(device)

                    a_tws = text_batch["target_word"]
                    p_tws = text_batch["target_word"]
                    flat_neg_tws = [x for negs in text_batch["neg_target_words"] for x in negs]

                    ea = chunked_encode_and_extract(
                        text_encoder, transformer, tokenizer, a_ids, a_mask, args.text_chunk_size,
                        source_mode=args.text_source_mode,
                        source_layer_idx=args.text_source_layer_idx,
                        source_range_start=args.text_source_range_start,
                        source_range_end=args.text_source_range_end,
                        target_words=a_tws,
                        target_token_weight=args.target_token_weight,
                    )
                    ep = chunked_encode_and_extract(
                        text_encoder, transformer, tokenizer, p_ids, p_mask, args.text_chunk_size,
                        source_mode=args.text_source_mode,
                        source_layer_idx=args.text_source_layer_idx,
                        source_range_start=args.text_source_range_start,
                        source_range_end=args.text_source_range_end,
                        target_words=p_tws,
                        target_token_weight=args.target_token_weight,
                    )
                    en_flat = chunked_encode_and_extract(
                        text_encoder, transformer, tokenizer, n_ids, n_mask, args.text_chunk_size,
                        source_mode=args.text_source_mode,
                        source_layer_idx=args.text_source_layer_idx,
                        source_range_start=args.text_source_range_start,
                        source_range_end=args.text_source_range_end,
                        target_words=flat_neg_tws,
                        target_token_weight=args.target_token_weight,
                    )
                    del a_ids, p_ids, n_ids, a_mask, p_mask, n_mask

                    B_ctr = ea.shape[0]
                    K = args.num_negatives

                    with torch.no_grad():
                        en_diag = en_flat.view(B_ctr, K, -1)
                        pos_sim = (ea * ep).sum(dim=-1).mean().item()
                        neg_sim = (ea.unsqueeze(1) * en_diag).sum(dim=-1).mean().item()
                        del en_diag

                    if args.apply_zscore_before_loss:
                        ea, ep, en_flat = zscore_then_l2(ea, ep, en_flat, eps=args.zscore_eps)

                    en = en_flat.view(B_ctr, K, -1)
                    if args.loss_type == "dcl":
                        loss_ctr = dcl_loss(ea, ep, en, temperature=args.temperature)
                    else:
                        loss_ctr = infonce_loss(ea, ep, en, temperature=args.temperature)
                    del ea, ep, en, en_flat
                else:
                    loss_ctr = torch.tensor(0.0, device=device)
                    pos_sim = neg_sim = 0.0
                    B_ctr = args.contrastive_batch_size
                    K = args.num_negatives

                # ── DPO / CPO loss (small image batch, GPU-memory bound) ─────────────
                diff_anchor_texts = [format_prompt(tokenizer, t, args.use_chat_template) for t in batch["anchor"]]
                da_ids, da_mask = tokenize_texts(tokenizer, diff_anchor_texts, args.max_length)
                da_ids = da_ids.to(device)
                da_mask = da_mask.to(device)

                with torch.no_grad():
                    da_out = text_encoder(input_ids=da_ids, attention_mask=da_mask.bool(), output_hidden_states=True)
                    da_h = select_text_source_hidden(
                        da_out.hidden_states,
                        source_mode=args.text_source_mode,
                        layer_idx=args.text_source_layer_idx,
                        range_start=args.text_source_range_start,
                        range_end=args.text_source_range_end,
                    ).detach().clone()
                    del da_out

                raw_cap_feats = [da_h[i][da_mask[i].bool()].to(dtype=transformer_dtype) for i in range(da_h.shape[0])]

                pixel_values_w = batch["pixel_values_w"].to(device, dtype=vae.dtype)
                pixel_values_l = batch["pixel_values_l"].to(device, dtype=vae.dtype)

                with torch.no_grad():
                    # Encode winner
                    h_w = vae.encoder(pixel_values_w)
                    moments_w = vae.quant_conv(h_w) if vae.quant_conv is not None else h_w
                    mean_w, log_var_w = moments_w.chunk(2, dim=1)
                    std_w = torch.exp(0.5 * log_var_w.clamp(-30, 20))
                    latents_w = mean_w + std_w * torch.randn_like(mean_w)
                    latents_w = latents_w * vae.config.scaling_factor

                    # Encode loser
                    h_l = vae.encoder(pixel_values_l)
                    moments_l = vae.quant_conv(h_l) if vae.quant_conv is not None else h_l
                    mean_l, log_var_l = moments_l.chunk(2, dim=1)
                    std_l = torch.exp(0.5 * log_var_l.clamp(-30, 20))
                    latents_l = mean_l + std_l * torch.randn_like(mean_l)
                    latents_l = latents_l * vae.config.scaling_factor

                # Shared noise and sigma for fair comparison
                noise = torch.randn_like(latents_w)
                sigma = torch.rand((latents_w.shape[0],), device=device, dtype=latents_w.dtype)
                sigma_b = sigma.view(-1, 1, 1, 1)

                noisy_w = (1.0 - sigma_b) * latents_w + sigma_b * noise
                target_w = latents_w - noise

                noisy_l = (1.0 - sigma_b) * latents_l + sigma_b * noise
                target_l = latents_l - noise

                t_norm = (1.0 - sigma).to(dtype=transformer_dtype)

                # Policy forward pass
                noisy_w_inputs = [x.unsqueeze(1).to(dtype=transformer_dtype) for x in noisy_w]
                noisy_l_inputs = [x.unsqueeze(1).to(dtype=transformer_dtype) for x in noisy_l]
                pred_w_list = forward_with_refiner_override(transformer, noisy_w_inputs, t_norm, raw_cap_feats)
                pred_w = torch.stack(pred_w_list, dim=0).squeeze(2).float()
                loss_w_policy = F.mse_loss(pred_w, target_w.float(), reduction="none").mean(dim=[1,2,3])

                pred_l_list = forward_with_refiner_override(transformer, noisy_l_inputs, t_norm, raw_cap_feats)
                pred_l = torch.stack(pred_l_list, dim=0).squeeze(2).float()
                loss_l_policy = F.mse_loss(pred_l, target_l.float(), reduction="none").mean(dim=[1,2,3])

                with torch.no_grad():
                    ref_pred_w_list = forward_with_refiner_override(
                        transformer,
                        noisy_w_inputs,
                        t_norm,
                        raw_cap_feats,
                        refiner_override=ref_context_refiner,
                    )
                    ref_pred_w = torch.stack(ref_pred_w_list, dim=0).squeeze(2).float()
                    loss_w_ref = F.mse_loss(ref_pred_w, target_w.float(), reduction="none").mean(dim=[1,2,3])

                    ref_pred_l_list = forward_with_refiner_override(
                        transformer,
                        noisy_l_inputs,
                        t_norm,
                        raw_cap_feats,
                        refiner_override=ref_context_refiner,
                    )
                    ref_pred_l = torch.stack(ref_pred_l_list, dim=0).squeeze(2).float()
                    loss_l_ref = F.mse_loss(ref_pred_l, target_l.float(), reduction="none").mean(dim=[1,2,3])

                model_diff = loss_w_policy - loss_l_policy
                ref_diff = loss_w_ref - loss_l_ref
                inside_term = -0.5 * args.beta_dpo * (model_diff - ref_diff)
                implicit_acc = (inside_term > 0).float().mean()
                policy_mse = 0.5 * (loss_w_policy.mean() + loss_l_policy.mean())
                ref_mse = 0.5 * (loss_w_ref.mean() + loss_l_ref.mean())
                loss_dpo = -F.logsigmoid(inside_term).mean()

                ctr_w = get_ctr_weight(global_step)
                loss = args.diffusion_weight * loss_dpo + ctr_w * loss_ctr

                if global_step == 0 and accelerator.sync_gradients and is_main:
                    print(f"[Pretrained baseline] L_dpo={loss_dpo.item():.4f}  L_ctr={loss_ctr.item():.4f}  "
                          f"sigma_mean={sigma.mean().item():.3f}  ctr_batch={B_ctr}  "
                          f"implicit_acc={implicit_acc.item():.3f}  "
                          f"model_diff={model_diff.mean().item():+.4f}  ref_diff={ref_diff.mean().item():+.4f}  "
                          f"p-n={pos_sim - neg_sim:+.4f}  "
                          f"random_baseline={torch.log(torch.tensor(K+1.0)):.4f}")

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if not accelerator.sync_gradients:
                continue

            global_step += 1
            steps += 1
            running_total += loss.item()
            running_dpo += loss_dpo.item()
            running_ctr += loss_ctr.item()
            running_implicit_acc += implicit_acc.item()
            running_policy_mse += policy_mse.item()
            running_ref_mse += ref_mse.item()

            if is_main:
                pbar.set_postfix(
                    {
                        "L_dpo": f"{loss_dpo.item():.4f}",
                        "L_ctr": f"{loss_ctr.item():.4f}",
                        "imp": f"{implicit_acc.item():.3f}",
                        "p-n": f"{pos_sim - neg_sim:+.3f}",
                    }
                )
                if use_wandb:
                    wandb.log(
                        {
                            "train/loss_total": loss.item(),
                            "train/loss_dpo": loss_dpo.item(),
                            "train/loss_ctr": loss_ctr.item(),
                            "train/pos_sim": pos_sim,
                            "train/neg_sim": neg_sim,
                            "train/pos_neg_gap": pos_sim - neg_sim,
                            "train/dpo_inside_term": inside_term.mean().item(),
                            "train/dpo_implicit_acc": implicit_acc.item(),
                            "train/dpo_model_diff": model_diff.mean().item(),
                            "train/dpo_ref_diff": ref_diff.mean().item(),
                            "train/dpo_policy_mse": policy_mse.item(),
                            "train/dpo_ref_mse": ref_mse.item(),
                            "train/dpo_policy_win_mse": loss_w_policy.mean().item(),
                            "train/dpo_policy_lose_mse": loss_l_policy.mean().item(),
                            "train/dpo_ref_win_mse": loss_w_ref.mean().item(),
                            "train/dpo_ref_lose_mse": loss_l_ref.mean().item(),
                            "train/ctr_weight": ctr_w,
                            "train/lr": optimizer.param_groups[0]["lr"],
                            "train/epoch": epoch,
                        },
                        step=global_step,
                    )

            if is_main and args.save_every > 0 and global_step % args.save_every == 0:
                save_path = Path(args.output_dir) / f"transformer_refiner_step{global_step}.pt"
                unwrapped = accelerator.unwrap_model(transformer)
                torch.save(
                    {
                        "step": global_step,
                        "epoch": epoch,
                        "transformer_state_dict": unwrapped.state_dict(),
                        "args": vars(args),
                    },
                    save_path,
                )
                print(f"[Checkpoint] saved: {save_path}")

            if args.vis_every > 0 and global_step % args.vis_every == 0:
                visualize(global_step)

        if steps == 0:
            if is_main:
                print(f"[Epoch {epoch}] no valid step.")
            continue

        avg_total = running_total / steps
        avg_dpo = running_dpo / steps
        avg_ctr = running_ctr / steps
        avg_implicit_acc = running_implicit_acc / steps
        avg_policy_mse = running_policy_mse / steps
        avg_ref_mse = running_ref_mse / steps
        if is_main:
            print(
                f"[Epoch {epoch}] avg_total={avg_total:.4f} avg_dpo={avg_dpo:.4f} "
                f"avg_ctr={avg_ctr:.4f} avg_imp={avg_implicit_acc:.4f} "
                f"avg_policy_mse={avg_policy_mse:.4f} avg_ref_mse={avg_ref_mse:.4f}"
            )
            if use_wandb:
                wandb.log(
                    {
                        "epoch/avg_total": avg_total,
                        "epoch/avg_dpo": avg_dpo,
                        "epoch/avg_ctr": avg_ctr,
                        "epoch/avg_implicit_acc": avg_implicit_acc,
                        "epoch/avg_policy_mse": avg_policy_mse,
                        "epoch/avg_ref_mse": avg_ref_mse,
                        "epoch/epoch": epoch,
                    },
                    step=global_step,
                )

    accelerator.wait_for_everyone()
    if is_main:
        final_path = Path(args.output_dir) / "transformer_refiner_final.pt"
        unwrapped = accelerator.unwrap_model(transformer)
        torch.save(
            {
                "step": global_step,
                "epoch": args.epochs,
                "transformer_state_dict": unwrapped.state_dict(),
                "args": vars(args),
            },
            final_path,
        )
        print(f"[Done] final checkpoint: {final_path}")
        if use_wandb:
            wandb.finish()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train transformer refiners with counting contrastive + DPO/CPO losses")
    p.add_argument("--model_dir", type=str, default="ckpts/Z-Image-Turbo")
    p.add_argument(
        "--triplets_jsonl",
        type=str,
        default="data/train_triplets/DPO/counting_dpo_index.jsonl",
        help="训练数据文件；仅支持 build_dpo_training_index.py 产出的 noun-grouped DPO index JSON/JSONL。",
    )
    p.add_argument("--output_dir", type=str, default="checkpoints/counting_text_refiner_dpo")

    p.add_argument("--verdict_threshold", type=float, default=0.8)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--use_chat_template", action="store_true")
    p.add_argument(
        "--text_source_mode",
        type=str,
        default="avg_range",
        choices=["layer", "avg_range"],
        help="Text encoder feature source fed into refiner.",
    )
    p.add_argument(
        "--text_source_layer_idx",
        type=int,
        default=-2,
        help="Used when text_source_mode=layer (supports negative index).",
    )
    p.add_argument(
        "--text_source_range_start",
        type=int,
        default=10,
        help="Used when text_source_mode=avg_range (inclusive).",
    )
    p.add_argument(
        "--text_source_range_end",
        type=int,
        default=20,
        help="Used when text_source_mode=avg_range (inclusive).",
    )

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=1, help="Per-GPU image batch size for diffusion loss")
    p.add_argument("--contrastive_batch_size", type=int, default=32,
                   help="Text-only batch size for contrastive loss (no images, cheap to scale up)")
    p.add_argument("--text_chunk_size", type=int, default=16,
                   help="Chunk size for text encoder / context_refiner forward pass (controls peak memory)")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="Number of micro-batches to accumulate before each optimizer step.")
    p.add_argument("--lr", type=float, default=1e-3, help="(Deprecated) kept for launch-script compatibility; unused.")
    p.add_argument("--refiner_lr", type=float, default=5e-4, help="Learning rate for context_refiner (pre-trained, fine-tuned)")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_negatives", type=int, default=12, help="InfoNCE negatives per anchor (1:K)")
    p.add_argument("--loss_type", type=str, default="infonce", choices=["infonce", "dcl"], help="Contrastive loss type")
    p.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature")
    p.add_argument(
        "--apply_zscore_before_loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply (x-mean)/std + L2 to [anchor,positive,negative] features before contrastive loss.",
    )
    p.add_argument(
        "--target_token_weight",
        type=float,
        default=2.5,
        help="Weight for the target token during mean pooling (e.g., 2.5). 1.0 means standard mean pooling.",
    )
    p.add_argument("--zscore_eps", type=float, default=1e-6)

    p.add_argument("--contrastive_weight", type=float, default=1.0,
                   help="Initial contrastive loss weight (linearly decays to 0 over ctr_decay_steps).")
    p.add_argument("--diffusion_weight", type=float, default=1.0, help="Diffusion loss weight (fixed).")
    p.add_argument("--beta_dpo", type=float, default=50.0, help="Beta parameter for reference-based DPO loss.")
    p.add_argument("--ctr_decay_steps", type=int, default=0,
                   help="Steps over which contrastive weight decays to 0. 0 = decay over all training steps.")
    p.add_argument("--no_ctr_decay", action="store_true",
                   help="Disable contrastive weight decay; keep contrastive_weight fixed throughout training.")
    p.add_argument("--proj_hidden_dim", type=int, default=512,
                   help="(Deprecated) kept for launch-script compatibility; unused.")
    p.add_argument("--proj_out_dim", type=int, default=256,
                   help="(Deprecated) kept for launch-script compatibility; unused.")

    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--vis_every", type=int, default=500,
                   help="Generate visualization images every N steps (0 = disabled)")
    p.add_argument("--print_trainable", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"],
                   help="Accelerate mixed precision mode")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="z-image-counting")
    p.add_argument("--wandb_run", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())