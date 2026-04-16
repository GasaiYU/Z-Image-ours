"""
Train counting-aware transformer refiners with joint losses:
1) Counting-token contrastive loss (InfoNCE, 1 positive : K negatives)
2) Diffusion denoising loss

Losses are decoupled:
- Diffusion loss: small image batch (batch_size, GPU-memory bound)
- Contrastive loss: large text-only batch (contrastive_batch_size, very cheap, no images)

Key constraints:
- Only train transformer context_refiner layers.
- Data source: data/train_triplets/counting_triplets_filtered.jsonl
- The JSONL is assumed pre-filtered; uses all images under
  data/generated_images/counting/<sanitized_anchor>/
"""

import argparse
import json
import os

import random
import re
import sys
from collections import defaultdict
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
) -> torch.Tensor:
    """
    Pipeline: encode → refine → content-only mean pooling, chunk by chunk.

    Chat-template special tokens (<|im_start|>, <|im_end|>, role words, etc.) are excluded
    from the pool so the sentence vector reflects only real content tokens (number + object).
    This way "seven canes" and "one canes" differ because both the number token AND the
    "canes" token (which is aware of its neighbour via bidirectional attention) contribute.

    Peak memory = chunk_size × seq_len × D (full-sequence tensors freed after pooling).
    Returns L2-normalised sentence vectors [N, D].
    """
    special_ids = set(tokenizer.all_special_ids)
    all_feats: list[torch.Tensor] = []
    N = input_ids.shape[0]

    for i in range(0, N, chunk_size):
        ids  = input_ids[i : i + chunk_size]
        mask = attention_mask[i : i + chunk_size]

        # Content mask: real token (mask==1) AND not a special token
        is_special = torch.zeros_like(ids, dtype=torch.bool)
        for sid in special_ids:
            is_special |= (ids == sid)
        content_mask = mask.bool() & ~is_special          # [chunk, seq_len]

        # Step 1: frozen text encoder — use full attention_mask for the encoder itself
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

        # Step 2: context_refiner — trainable, gradients flow; full mask for padding
        h_ref = run_context_refiner(transformer, h, mask)
        del h

        # Step 3: mean pool over content tokens only, then L2-normalise
        float_mask = content_mask.unsqueeze(-1).float()
        denom = float_mask.sum(dim=1).clamp(min=1.0)
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
    merged = merged - merged.mean(dim=0, keepdim=True)
    merged = merged / (merged.std(dim=0, keepdim=True, unbiased=False) + eps)
    merged = F.normalize(merged, dim=-1)
    b = ea.shape[0]
    return merged[:b], merged[b : 2 * b], merged[2 * b :]


def infonce_loss(ea: torch.Tensor, ep: torch.Tensor, en: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    InfoNCE with one positive and K negatives per anchor.
    ea: [B, D], ep: [B, D], en: [B, K, D]
    """
    ea, ep, en = ea.float(), ep.float(), en.float()
    pos_logits = (ea * ep).sum(dim=-1, keepdim=True)          # [B, 1]
    neg_logits = (ea.unsqueeze(1) * en).sum(dim=-1)           # [B, K]
    logits = torch.cat([pos_logits, neg_logits], dim=1) / temperature
    labels = torch.zeros(ea.shape[0], dtype=torch.long, device=ea.device)
    return F.cross_entropy(logits, labels)


def unfreeze_transformer_refiner_layers(transformer: torch.nn.Module) -> list[str]:
    """Freeze all transformer params, then unfreeze only context_refiner (text conditioning side)."""
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


def run_context_refiner(transformer: torch.nn.Module, token_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Project text features to transformer dim, then run context_refiner blocks."""
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

    refined = cap_feats
    for layer in model.context_refiner:
        refined = layer(refined, attn_mask, cap_freqs)
    return refined


def make_anchor_variants(anchor: str) -> list[str]:
    """
    Generate template rewrites of the same anchor (same number + same object, different phrasing).
    These are used as positives for contrastive learning, replacing cross-anchor positives.
    All variants share the same semantic content, so context_refiner pollution from
    the object is symmetric between anchor and positive — much easier to pull together.
    """
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


NUMBER_WORDS = [
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
]


def make_anchor_negatives(anchor: str, target_word: str) -> list[tuple[str, str]]:
    """
    Generate synthetic hard negatives by replacing the number word in the anchor
    with every other number word, then expanding each with template variants.
    Returns list of (neg_text, neg_target_word).

    9 number substitutions × up to 10 templates = up to 90 unique negatives,
    so any num_negatives ≤ 90 can be satisfied without repetition.
    """
    tw = target_word.strip().lower()
    negatives: list[tuple[str, str]] = []
    seen: set[str] = set()
    for num in NUMBER_WORDS:
        if num == tw:
            continue
        neg_base = anchor.replace(target_word, num, 1)
        if neg_base == anchor:          # replacement didn't happen (word not found)
            continue
        for variant in make_anchor_variants(neg_base):
            key = variant.strip().lower()
            if key not in seen:
                negatives.append((variant, num))
                seen.add(key)
    return negatives


@dataclass
class CountingSample:
    anchor: str
    anchor_variants: list[str]          # template rewrites of anchor (used as positives)
    synthetic_negatives: list[tuple[str, str]]  # (neg_text, neg_target_word) — same object, diff number
    positive_pool: list[str]            # kept for reference only
    negative_pool: list[str]            # kept for reference only
    target_word: str
    image_paths: list[Path]


class CountingVerdictDataset(Dataset):
    def __init__(
        self,
        triplets_jsonl: str,
        generated_root: str,
        threshold: float,
        resolution: int,
        num_negatives: int,
    ):
        self.triplets_jsonl = Path(triplets_jsonl)
        self.generated_root = Path(generated_root)
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

        rows = []
        with open(self.triplets_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("task", "counting") != "counting":
                    continue
                if all(k in obj for k in ("anchor", "positive", "negative", "target_word")):
                    rows.append(obj)

        image_cache: dict[str, list[Path]] = {}
        same_word_texts: dict[str, set[str]] = defaultdict(set)
        grouped: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"target_word": "", "positive_pool": set(), "negative_pool": set()}
        )
        for obj in rows:
            anchor = obj["anchor"].strip()
            target_word = obj["target_word"].strip()
            positive = obj["positive"].strip()
            negative = obj["negative"].strip()

            grouped[anchor]["target_word"] = target_word
            if positive:
                grouped[anchor]["positive_pool"].add(positive)
                same_word_texts[target_word].add(positive)
            if negative:
                grouped[anchor]["negative_pool"].add(negative)
            if anchor:
                same_word_texts[target_word].add(anchor)

        self.samples: list[CountingSample] = []
        for anchor, item in grouped.items():
            if anchor not in image_cache:
                image_cache[anchor] = self._collect_anchor_images(anchor)
            image_paths = image_cache[anchor]
            if not image_paths:
                continue
            tw = item["target_word"]
            self.samples.append(
                CountingSample(
                    anchor=anchor,
                    anchor_variants=make_anchor_variants(anchor),
                    synthetic_negatives=make_anchor_negatives(anchor, tw),
                    positive_pool=[x for x in item["positive_pool"] if x],
                    negative_pool=[x for x in item["negative_pool"] if x],
                    target_word=tw,
                    image_paths=image_paths,
                )
            )

        print(f"[Dataset] Total counting triplets: {len(rows)}")
        print(f"[Dataset] Unique anchors: {len(grouped)}")
        print(f"[Dataset] Kept with available images: {len(self.samples)}")
  
    def _collect_anchor_images(self, anchor: str) -> list[Path]:
        sample_dir = self.generated_root / "counting" / sanitize(anchor)
        if not sample_dir.exists():
            return []
        valid_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        return sorted(
            [
                p for p in sample_dir.iterdir()
                if p.is_file() and p.suffix.lower() in valid_suffixes
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_negatives(self, synthetic_negatives: list[tuple[str, str]]) -> tuple[list[str], list[str]]:
        """Return (neg_texts, neg_target_words) sampled from synthetic hard negatives.

        Synthetic negatives are generated by replacing the anchor's number word with
        every other number word — guaranteed same object, different number.
        """
        pool = synthetic_negatives
        if not pool:
            return [], []
        if len(pool) >= self.num_negatives:
            chosen = random.sample(pool, self.num_negatives)
        else:
            chosen = [random.choice(pool) for _ in range(self.num_negatives)]
        return [t for t, _ in chosen], [tw for _, tw in chosen]

    def _sample_anchor_positive(self, anchor: str, variants: list[str]) -> str:
        """
        Sample a template rewrite of the same anchor as the positive.
        Positive shares the same number word AND object as the anchor — only phrasing differs.
        This ensures context_refiner's bidirectional object-context pollution is symmetric
        between anchor and positive, making them naturally easier to pull together.
        """
        pool = [v for v in variants if v.strip().lower() != anchor.strip().lower()]
        if not pool:
            return anchor
        return random.choice(pool)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Returns one image sample for diffusion loss (includes pixel_values)."""
        s = self.samples[idx]
        image_path = random.choice(s.image_paths)
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_tf(image)
        return {
            "anchor": s.anchor,
            "target_word": s.target_word,
            "pixel_values": pixel_values,
        }

    def sample_text_batch(self, n: int) -> list[dict[str, Any]]:
        """Sample n text-only items for contrastive loss (no image loading)."""
        chosen = random.choices(self.samples, k=n)
        items = []
        for s in chosen:
            positive = self._sample_anchor_positive(s.anchor, s.anchor_variants)
            negatives, neg_target_words = self._sample_negatives(s.synthetic_negatives)
            items.append({
                "anchor": s.anchor,
                "positive": positive,
                "negatives": negatives,
                "neg_target_words": neg_target_words,
                "target_word": s.target_word,
            })
        return items


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate image batch (for diffusion loss). No text fields needed here."""
    return {
        "anchor": [b["anchor"] for b in batch],
        "target_word": [b["target_word"] for b in batch],
        "pixel_values": torch.stack([b["pixel_values"] for b in batch], dim=0),
    }


def collate_text_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate text-only batch (for contrastive loss)."""
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
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
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

    # Load all components on CPU first to save GPU memory during setup
    components = load_from_local_dir(
        args.model_dir,
        device="cpu",
        dtype=torch.bfloat16,
        verbose=is_main,
    )
    transformer = components["transformer"].to(device)
    vae = components["vae"].to(device)
    text_encoder = components["text_encoder"].to(device)
    tokenizer = components["tokenizer"]

    for p in text_encoder.parameters():
        p.requires_grad_(False)
    for p in vae.parameters():
        p.requires_grad_(False)
    text_encoder.eval()
    vae.eval()

    trainable_names = unfreeze_transformer_refiner_layers(transformer)

    # Get transformer hidden dim before accelerator wraps it
    _raw_transformer = transformer.module if hasattr(transformer, "module") else transformer
    _transformer_dim = _raw_transformer.dim

    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    if is_main:
        n_refiner = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        print(f"[Init] trainable transformer params: {n_refiner:,}")
        print(f"[Init] projection head: disabled")
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
        generated_root=args.generated_root,
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

    # accelerate wraps transformer (trainable) + optimizer + loader
    transformer, optimizer, loader = accelerator.prepare(transformer, optimizer, loader)

    vis_dir = Path(args.output_dir) / "vis"
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)

    # Fixed prompts for periodic visualization
    VIS_PROMPTS = [
        "three cats sitting on a sofa",
        "five apples on a table",
        "two dogs running in a park",
        "seven birds on a wire",
        "a photo of three baseballs",
        "a photo of four computer keyboards",
        "four oranges on a white table",
        "three red flowers in a vase",
    ]

    def visualize(step: int) -> None:
        """Generate images with current weights and save/log them (main process only)."""
        if not is_main:
            return
        unwrapped = accelerator.unwrap_model(transformer)
        unwrapped.eval()
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

    total_steps = args.epochs * len(loader)
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

    def get_ctr_weight(step: int) -> float:
        if args.no_ctr_decay:
            return args.contrastive_weight
        if step >= ctr_decay_steps:
            return 0.0
        return args.contrastive_weight * (1.0 - step / ctr_decay_steps)

    # Baseline visualization before any training
    if args.vis_every > 0:
        visualize(0)

    for epoch in range(1, args.epochs + 1):
        transformer.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not is_main)
        running_total = 0.0
        running_diff = 0.0
        running_ctr = 0.0
        steps = 0

        for batch in pbar:
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

                ea = chunked_encode_and_extract(
                    text_encoder, transformer, tokenizer, a_ids, a_mask, args.text_chunk_size,
                    source_mode=args.text_source_mode,
                    source_layer_idx=args.text_source_layer_idx,
                    source_range_start=args.text_source_range_start,
                    source_range_end=args.text_source_range_end,
                )
                ep = chunked_encode_and_extract(
                    text_encoder, transformer, tokenizer, p_ids, p_mask, args.text_chunk_size,
                    source_mode=args.text_source_mode,
                    source_layer_idx=args.text_source_layer_idx,
                    source_range_start=args.text_source_range_start,
                    source_range_end=args.text_source_range_end,
                )
                en_flat = chunked_encode_and_extract(
                    text_encoder, transformer, tokenizer, n_ids, n_mask, args.text_chunk_size,
                    source_mode=args.text_source_mode,
                    source_layer_idx=args.text_source_layer_idx,
                    source_range_start=args.text_source_range_start,
                    source_range_end=args.text_source_range_end,
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
                loss_ctr = infonce_loss(ea, ep, en, temperature=args.temperature)
                del ea, ep, en, en_flat
            else:
                loss_ctr = torch.tensor(0.0, device=device)
                pos_sim = neg_sim = 0.0
                B_ctr = args.contrastive_batch_size
                K = args.num_negatives

            # ── Diffusion loss (small image batch, GPU-memory bound) ─────────────
            # Use anchor text from the image batch for diffusion conditioning
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

            pixel_values = batch["pixel_values"].to(device, dtype=vae.dtype)
            with torch.no_grad():
                h = vae.encoder(pixel_values)
                moments = vae.quant_conv(h) if vae.quant_conv is not None else h
                mean, log_var = moments.chunk(2, dim=1)
                std = torch.exp(0.5 * log_var.clamp(-30, 20))
                latents = mean + std * torch.randn_like(mean)
                latents = latents * vae.config.scaling_factor

            noise = torch.randn_like(latents)
            sigma = torch.rand((latents.shape[0],), device=device, dtype=latents.dtype)
            sigma_b = sigma.view(-1, 1, 1, 1)
            noisy_latents = (1.0 - sigma_b) * latents + sigma_b * noise
            target = latents - noise

            t_norm = (1.0 - sigma).to(dtype=transformer_dtype)
            lat_list = [x.unsqueeze(1).to(dtype=transformer_dtype) for x in noisy_latents]
            cap_feats = [da_h[i][da_mask[i].bool()].to(dtype=transformer_dtype) for i in range(da_h.shape[0])]

            pred_list = transformer(lat_list, t_norm, cap_feats)[0]
            pred = torch.stack(pred_list, dim=0).squeeze(2).float()

            loss_diff = F.mse_loss(pred.float(), target.float(), reduction="mean")
            ctr_w = get_ctr_weight(global_step)
            loss = args.diffusion_weight * loss_diff + ctr_w * loss_ctr

            if global_step == 0 and is_main:
                print(f"[Pretrained baseline] L_diff={loss_diff.item():.4f}  L_ctr={loss_ctr.item():.4f}  "
                      f"sigma_mean={sigma.mean().item():.3f}  ctr_batch={B_ctr}  "
                      f"p-n={pos_sim - neg_sim:+.4f}  "
                      f"random_baseline={torch.log(torch.tensor(K+1.0)):.4f}")

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)

            accelerator.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            global_step += 1
            steps += 1
            running_total += loss.item()
            running_diff += loss_diff.item()
            running_ctr += loss_ctr.item()

            if is_main:
                pbar.set_postfix(
                    {
                        "L_diff": f"{loss_diff.item():.4f}",
                        "L_ctr": f"{loss_ctr.item():.4f}",
                        "p-n": f"{pos_sim - neg_sim:+.3f}",
                    }
                )
                if use_wandb:
                    wandb.log(
                        {
                            "train/loss_total": loss.item(),
                            "train/loss_diff": loss_diff.item(),
                            "train/loss_ctr": loss_ctr.item(),
                            "train/pos_sim": pos_sim,
                            "train/neg_sim": neg_sim,
                            "train/pos_neg_gap": pos_sim - neg_sim,
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
        avg_diff = running_diff / steps
        avg_ctr = running_ctr / steps
        if is_main:
            print(f"[Epoch {epoch}] avg_total={avg_total:.4f} avg_diff={avg_diff:.4f} avg_ctr={avg_ctr:.4f}")
            if use_wandb:
                wandb.log(
                    {
                        "epoch/avg_total": avg_total,
                        "epoch/avg_diff": avg_diff,
                        "epoch/avg_ctr": avg_ctr,
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
    p = argparse.ArgumentParser(description="Train transformer refiners with counting contrastive + diffusion losses")
    p.add_argument("--model_dir", type=str, default="ckpts/Z-Image-Turbo")
    p.add_argument("--triplets_jsonl", type=str, default="data/train_triplets/counting_triplets_filtered.jsonl")
    p.add_argument("--generated_root", type=str, default="data/generated_images")
    p.add_argument("--output_dir", type=str, default="checkpoints/counting_text_refiner")

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
    p.add_argument("--lr", type=float, default=1e-3, help="(Deprecated) kept for launch-script compatibility; unused.")
    p.add_argument("--refiner_lr", type=float, default=5e-4, help="Learning rate for context_refiner (pre-trained, fine-tuned)")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_negatives", type=int, default=12, help="InfoNCE negatives per anchor (1:K)")
    p.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature")
    p.add_argument(
        "--apply_zscore_before_loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply (x-mean)/std + L2 to [anchor,positive,negative] features before contrastive loss.",
    )
    p.add_argument("--zscore_eps", type=float, default=1e-6)

    p.add_argument("--contrastive_weight", type=float, default=1.0,
                   help="Initial contrastive loss weight (linearly decays to 0 over ctr_decay_steps).")
    p.add_argument("--diffusion_weight", type=float, default=1.0, help="Diffusion loss weight (fixed).")
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
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"],
                   help="Accelerate mixed precision mode")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="z-image-counting")
    p.add_argument("--wandb_run", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())