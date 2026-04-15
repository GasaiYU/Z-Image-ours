"""
Router-based counting training:
1) Train a lightweight DynamicTokenRouter on frozen text encoder hidden states.
2) Router routes counting tokens (e.g. "seven") to shallower, more discriminative layers.
3) Contrastive loss (InfoNCE / DCL) on router-fused sentence embeddings.
4) Optionally: diffusion loss uses router-fused embeddings as DiT text conditioning.

Why Router instead of Refiner:
- Refiner output collapses (cos_sim ~0.99) due to self-attention mixing → InfoNCE gradient → 0
- h_1 (LLM layer 1) already has cos_sim ~0.84 for counting words → non-zero gradient from start
- Router is a small MLP (~4M params) → large per-parameter gradient, no bf16 floor problem
- Frozen DiT is not disturbed by narrow counting data (no catastrophic forgetting)
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
import torch.nn as nn
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


# =============================================================================
# Utilities
# =============================================================================

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


# =============================================================================
# DynamicTokenRouter
# =============================================================================

class DynamicTokenRouter(nn.Module):
    """
    Lightweight MLP: takes h_1 (shallowest LLM layer) as decision input and
    outputs per-token routing weights over a mid-layer range [route_start, route_end).

    Design rationale
    ----------------
    h_1 carries the strongest lexical identity signal. For counting words
    (two/three/seven…), h_1 similarities are ~0.84, far more discriminative
    than deeper layers (0.93~0.99).

    Layers [route_start, route_end) (default 10–21) balance discriminability
    and semantic richness. Deep layers (25+) are nearly identical for counting
    words and carry no discriminative signal.

    Initialisation: bias[-1] = +5 → softmax ≈ 1 on hidden_states[route_end-1].
    At t=0 the router behaves like the original pipeline (deep layer). The
    contrastive loss gradually shifts counting tokens to shallower layers.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        mid_dim: int = 1024,
        route_start: int = 10,
        route_end: int = 21,
    ):
        super().__init__()
        self.route_start = max(1, min(route_start, num_layers - 2))
        self.route_end   = max(self.route_start + 1, min(route_end, num_layers))
        self.n_route     = self.route_end - self.route_start

        self.router_mlp = nn.Sequential(
            nn.Linear(hidden_size, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(mid_dim, mid_dim // 2),
            nn.SiLU(),
            nn.Linear(mid_dim // 2, self.n_route),
        )

        # Deep-biased init: route to deepest layer in range at t=0
        nn.init.zeros_(self.router_mlp[-1].bias)
        self.router_mlp[-1].bias.data[-1] = 5.0
        nn.init.normal_(self.router_mlp[-1].weight, std=0.01)

    def forward(
        self,
        all_hidden_states: tuple,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            all_hidden_states: tuple of (num_layers+1) tensors [B, S, D]
            attention_mask:    [B, S] bool/int mask (1 = valid token)
        Returns:
            fused_embeds    [B, S, D]        weighted sum over routing layers
            routing_weights [B, S, n_route]  softmax weights
            h1              [B, S, D]        raw h_1 features
        """
        h1            = all_hidden_states[1].float()
        decision_feat = F.normalize(h1.detach(), dim=-1)     # frozen → no grad to encoder

        routing_logits  = self.router_mlp(decision_feat)     # [B, S, n_route]
        routing_weights = F.softmax(routing_logits, dim=-1)

        route_layers = all_hidden_states[self.route_start : self.route_end]
        stacked = torch.stack([l.float() for l in route_layers], dim=2)  # [B, S, n, D]
        rw      = routing_weights.to(stacked.dtype).unsqueeze(-1)         # [B, S, n, 1]
        fused   = (stacked * rw).sum(dim=2)                               # [B, S, D]

        if attention_mask is not None:
            mask  = attention_mask.unsqueeze(-1).to(fused.dtype)
            fused = fused * mask

        return fused, routing_weights, h1


# =============================================================================
# Loss functions
# =============================================================================

def infonce_loss(ea: torch.Tensor, ep: torch.Tensor, en: torch.Tensor, temperature: float) -> torch.Tensor:
    """InfoNCE: 1 positive + K negatives per anchor. ea/ep/en: L2-normalised."""
    ea, ep, en = ea.float(), ep.float(), en.float()
    pos_logits = (ea * ep).sum(dim=-1, keepdim=True)       # [B, 1]
    neg_logits = (ea.unsqueeze(1) * en).sum(dim=-1)        # [B, K]
    logits = torch.cat([pos_logits, neg_logits], dim=1) / temperature
    labels = torch.zeros(ea.shape[0], dtype=torch.long, device=ea.device)
    return F.cross_entropy(logits, labels)


def dcl_loss(ea: torch.Tensor, ep: torch.Tensor, en: torch.Tensor, temperature: float) -> torch.Tensor:
    """Decoupled Contrastive Learning: removes positive from denominator.
    L = -(ea·ep)/τ + logsumexp((ea·en)/τ)
    Stronger negative push vs InfoNCE, less risk of gradient cancellation.
    """
    ea, ep, en = ea.float(), ep.float(), en.float()
    pos_logits = (ea * ep).sum(dim=-1) / temperature
    neg_logits = (ea.unsqueeze(1) * en).sum(dim=-1) / temperature
    return (-pos_logits + torch.logsumexp(neg_logits, dim=-1)).mean()


def triplet_margin_loss(ea: torch.Tensor, ep: torch.Tensor, en: torch.Tensor, margin: float) -> torch.Tensor:
    """Batch-hard triplet loss. Picks the hardest (closest) negative per anchor."""
    ea, ep, en = ea.float(), ep.float(), en.float()
    pos_dist         = 1.0 - (ea * ep).sum(dim=-1)
    neg_dist         = 1.0 - (ea.unsqueeze(1) * en).sum(dim=-1)
    hardest_neg_dist = neg_dist.min(dim=1).values
    return F.relu(pos_dist - hardest_neg_dist + margin).mean()


# =============================================================================
# Router-based encoding
# =============================================================================

def chunked_router_encode(
    text_encoder,
    router: DynamicTokenRouter,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """
    Encode texts:
      frozen text_encoder → all hidden states
      → DynamicTokenRouter (trainable) → fused_embeds [B, S, D]
      → content-only mean pool → L2 normalise → [B, D]
    """
    special_ids = set(tokenizer.all_special_ids)
    all_feats: list[torch.Tensor] = []
    N = input_ids.shape[0]

    for i in range(0, N, chunk_size):
        ids  = input_ids[i : i + chunk_size]
        mask = attention_mask[i : i + chunk_size]

        is_special   = torch.zeros_like(ids, dtype=torch.bool)
        for sid in special_ids:
            is_special |= (ids == sid)
        content_mask = mask.bool() & ~is_special

        with torch.no_grad():
            out = text_encoder(input_ids=ids, attention_mask=mask.bool(), output_hidden_states=True)

        fused, _, _ = router(out.hidden_states, attention_mask=mask)
        del out

        float_mask = content_mask.unsqueeze(-1).float()
        denom      = float_mask.sum(dim=1).clamp(min=1.0)
        pooled     = (fused * float_mask).sum(dim=1) / denom
        pooled     = F.normalize(pooled.float(), dim=-1)
        del fused

        all_feats.append(pooled)

    return torch.cat(all_feats, dim=0)


# =============================================================================
# Dataset
# =============================================================================

NUMBER_WORDS = ["one", "two", "three", "four", "five",
                "six", "seven", "eight", "nine", "ten"]


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


def make_anchor_negatives(anchor: str, target_word: str) -> list[tuple[str, str]]:
    tw = target_word.strip().lower()
    negatives: list[tuple[str, str]] = []
    seen: set[str] = set()
    for num in NUMBER_WORDS:
        if num == tw:
            continue
        neg_base = anchor.replace(target_word, num, 1)
        if neg_base == anchor:
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
    anchor_variants: list[str]
    synthetic_negatives: list[tuple[str, str]]
    target_word: str
    image_paths: list[Path]


class CountingVerdictDataset(Dataset):
    def __init__(
        self,
        triplets_jsonl: str,
        generated_root: str,
        resolution: int,
        num_negatives: int,
    ):
        self.generated_root = Path(generated_root)
        self.num_negatives  = num_negatives

        self.image_tf = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        rows = []
        with open(triplets_jsonl, "r", encoding="utf-8") as f:
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
                if all(k in obj for k in ("anchor", "target_word")):
                    rows.append(obj)

        grouped: dict[str, str] = {}
        for obj in rows:
            anchor = obj["anchor"].strip()
            if anchor not in grouped:
                grouped[anchor] = obj["target_word"].strip()

        self.samples: list[CountingSample] = []
        for anchor, tw in grouped.items():
            img_paths = self._collect_anchor_images(anchor)
            if not img_paths:
                continue
            self.samples.append(CountingSample(
                anchor=anchor,
                anchor_variants=make_anchor_variants(anchor),
                synthetic_negatives=make_anchor_negatives(anchor, tw),
                target_word=tw,
                image_paths=img_paths,
            ))

        print(f"[Dataset] Unique anchors with images: {len(self.samples)}")

    def _collect_anchor_images(self, anchor: str) -> list[Path]:
        sample_dir = self.generated_root / "counting" / sanitize(anchor)
        if not sample_dir.exists():
            return []
        valid_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        return sorted([
            p for p in sample_dir.iterdir()
            if p.is_file() and p.suffix.lower() in valid_suffixes
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def _sample_negatives(self, synthetic_negatives):
        pool = synthetic_negatives
        if not pool:
            return [], []
        chosen = (random.sample(pool, self.num_negatives)
                  if len(pool) >= self.num_negatives
                  else [random.choice(pool) for _ in range(self.num_negatives)])
        return [t for t, _ in chosen], [tw for _, tw in chosen]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s = self.samples[idx]
        image_path   = random.choice(s.image_paths)
        pixel_values = self.image_tf(Image.open(image_path).convert("RGB"))
        return {"anchor": s.anchor, "target_word": s.target_word, "pixel_values": pixel_values}

    def sample_text_batch(self, n: int) -> list[dict[str, Any]]:
        chosen = random.choices(self.samples, k=n)
        items  = []
        for s in chosen:
            # positive: anchor variant ≠ anchor itself
            variants = [v for v in s.anchor_variants if v.strip().lower() != s.anchor.strip().lower()]
            positive = random.choice(variants) if variants else s.anchor

            negatives, neg_tws = self._sample_negatives(s.synthetic_negatives)
            items.append({
                "anchor":          s.anchor,
                "positive":        positive,
                "negatives":       negatives,
                "neg_target_words":neg_tws,
                "target_word":     s.target_word,
            })
        return items


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "anchor":       [b["anchor"] for b in batch],
        "target_word":  [b["target_word"] for b in batch],
        "pixel_values": torch.stack([b["pixel_values"] for b in batch], dim=0),
    }


def tokenize_texts(tokenizer, texts: list[str], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    enc = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=max_length, return_tensors="pt")
    return enc.input_ids, enc.attention_mask


# =============================================================================
# Main
# =============================================================================

def main(args: argparse.Namespace) -> None:
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    device  = accelerator.device
    is_main = accelerator.is_main_process

    if args.seed is not None:
        set_seed(args.seed)

    use_wandb = args.use_wandb and _WANDB_AVAILABLE and is_main
    if args.use_wandb and not _WANDB_AVAILABLE and is_main:
        print("[WandB] wandb not installed.")
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run or None, config=vars(args))
        print(f"[WandB] Run: {wandb.run.url}")

    components   = load_from_local_dir(args.model_dir, device="cpu", dtype=torch.bfloat16, verbose=is_main)
    transformer  = components["transformer"].to(device)
    vae          = components["vae"].to(device)
    text_encoder = components["text_encoder"].to(device)
    tokenizer    = components["tokenizer"]

    # Freeze everything — only router is trained
    for p in text_encoder.parameters():
        p.requires_grad_(False)
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in transformer.parameters():
        p.requires_grad_(False)
    text_encoder.eval()
    vae.eval()
    transformer.eval()

    # Probe hidden_size and num_layers
    _ids  = torch.zeros(1, 4, dtype=torch.long, device=device)
    _mask = torch.ones(1, 4, dtype=torch.long, device=device)
    with torch.no_grad():
        _out = text_encoder(input_ids=_ids, attention_mask=_mask, output_hidden_states=True)
    hidden_size = _out.hidden_states[0].shape[-1]
    num_layers  = len(_out.hidden_states) - 1
    del _out, _ids, _mask
    if is_main:
        print(f"[Init] text_encoder: hidden_size={hidden_size}, num_layers={num_layers}")

    router = DynamicTokenRouter(
        hidden_size=hidden_size,
        num_layers=num_layers,
        mid_dim=args.mid_dim,
        route_start=args.route_start,
        route_end=args.route_end,
    ).to(device)

    if is_main:
        n_router = sum(p.numel() for p in router.parameters())
        print(f"[Init] DynamicTokenRouter: {n_router:,} params  "
              f"routing layers [{router.route_start}, {router.route_end}), n_route={router.n_route}")

    dataset = CountingVerdictDataset(
        triplets_jsonl=args.triplets_jsonl,
        generated_root=args.generated_root,
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

    optimizer = torch.optim.AdamW(router.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    router, optimizer, loader = accelerator.prepare(router, optimizer, loader)

    vis_dir = Path(args.output_dir) / "vis"
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        vis_dir.mkdir(parents=True, exist_ok=True)

    VIS_PROMPTS = [
        "three cats sitting on a sofa",
        "five apples on a table",
        "two dogs running in a park",
        "seven birds on a wire",
    ]

    transformer_dtype = next(transformer.parameters()).dtype

    def visualize(step: int) -> None:
        if not is_main:
            return
        raw_router = accelerator.unwrap_model(router)
        raw_router.eval()

        orig_forward = text_encoder.forward

        def patched_forward(*a, **kw):
            kw["output_hidden_states"] = True
            out     = orig_forward(*a, **kw)
            fused, _, _ = raw_router(out.hidden_states)
            hs_list = list(out.hidden_states)
            hs_list[-2] = fused.to(out.hidden_states[-2].dtype)
            out.hidden_states = tuple(hs_list)
            return out

        text_encoder.forward = patched_forward
        images = pipeline_generate(
            transformer=transformer,
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
        text_encoder.forward = orig_forward

        for i, (img, prompt) in enumerate(zip(images, VIS_PROMPTS)):
            img.save(vis_dir / f"step{step:06d}_p{i}.png")
        if use_wandb:
            wandb.log({"vis": [wandb.Image(img, caption=p) for img, p in zip(images, VIS_PROMPTS)]}, step=step)
        raw_router.train()
        print(f"[Vis] saved {len(images)} images at step {step} → {vis_dir}")

    global_step = 0

    if args.vis_every > 0:
        visualize(0)

    for epoch in range(1, args.epochs + 1):
        router.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not is_main)
        running_total = running_diff = running_ctr = 0.0
        steps = 0

        for batch in pbar:

            # ── Contrastive loss ──────────────────────────────────────────────
            if args.contrastive_weight > 0:
                text_items = dataset.sample_text_batch(args.contrastive_batch_size)
                anchors  = [format_prompt(tokenizer, b["anchor"],   args.use_chat_template) for b in text_items]
                positives= [format_prompt(tokenizer, b["positive"], args.use_chat_template) for b in text_items]
                flat_negs= [format_prompt(tokenizer, t, args.use_chat_template)
                            for b in text_items for t in b["negatives"]]

                a_ids, a_mask = tokenize_texts(tokenizer, anchors,   args.max_length)
                p_ids, p_mask = tokenize_texts(tokenizer, positives, args.max_length)
                n_ids, n_mask = tokenize_texts(tokenizer, flat_negs, args.max_length)
                a_ids = a_ids.to(device); a_mask = a_mask.to(device)
                p_ids = p_ids.to(device); p_mask = p_mask.to(device)
                n_ids = n_ids.to(device); n_mask = n_mask.to(device)

                ea     = chunked_router_encode(text_encoder, router, tokenizer, a_ids, a_mask, args.text_chunk_size)
                ep     = chunked_router_encode(text_encoder, router, tokenizer, p_ids, p_mask, args.text_chunk_size)
                en_flat= chunked_router_encode(text_encoder, router, tokenizer, n_ids, n_mask, args.text_chunk_size)
                del a_ids, p_ids, n_ids, a_mask, p_mask, n_mask

                B = ea.shape[0]
                K = args.num_negatives
                en = en_flat.view(B, K, -1)

                with torch.no_grad():
                    pos_sim = (ea * ep).sum(dim=-1).mean().item()
                    neg_sim = (ea.unsqueeze(1) * en).sum(dim=-1).mean().item()

                if args.loss_type == "dcl":
                    loss_ctr = dcl_loss(ea, ep, en, temperature=args.temperature)
                elif args.loss_type == "triplet":
                    loss_ctr = triplet_margin_loss(ea, ep, en, margin=args.triplet_margin)
                else:
                    loss_ctr = infonce_loss(ea, ep, en, temperature=args.temperature)

                del ea, ep, en_flat, en
            else:
                loss_ctr = torch.tensor(0.0, device=device)
                pos_sim = neg_sim = 0.0
                B, K = args.contrastive_batch_size, args.num_negatives

            # ── Diffusion loss ────────────────────────────────────────────────
            if args.diffusion_weight > 0:
                diff_texts = [format_prompt(tokenizer, t, args.use_chat_template) for t in batch["anchor"]]
                da_ids, da_mask = tokenize_texts(tokenizer, diff_texts, args.max_length)
                da_ids  = da_ids.to(device)
                da_mask = da_mask.to(device)

                with torch.no_grad():
                    da_out = text_encoder(input_ids=da_ids, attention_mask=da_mask.bool(), output_hidden_states=True)

                fused, _, _ = router(da_out.hidden_states, attention_mask=da_mask)
                del da_out

                pixel_values = batch["pixel_values"].to(device, dtype=vae.dtype)
                with torch.no_grad():
                    h       = vae.encoder(pixel_values)
                    moments = vae.quant_conv(h) if vae.quant_conv is not None else h
                    mean, lv = moments.chunk(2, dim=1)
                    std      = torch.exp(0.5 * lv.clamp(-30, 20))
                    latents  = (mean + std * torch.randn_like(mean)) * vae.config.scaling_factor

                noise         = torch.randn_like(latents)
                sigma         = torch.rand((latents.shape[0],), device=device, dtype=latents.dtype)
                sigma_b       = sigma.view(-1, 1, 1, 1)
                noisy_latents = (1.0 - sigma_b) * latents + sigma_b * noise
                target        = latents - noise
                t_norm        = (1.0 - sigma).to(dtype=transformer_dtype)

                lat_list  = [x.unsqueeze(1).to(dtype=transformer_dtype) for x in noisy_latents]
                cap_feats = [fused[i][da_mask[i].bool()].to(dtype=transformer_dtype) for i in range(fused.shape[0])]
                del fused

                with torch.no_grad():
                    pred_list = transformer(lat_list, t_norm, cap_feats)[0]
                pred = torch.stack(pred_list, dim=0).squeeze(2).float()
                loss_diff = F.mse_loss(pred.float(), target.float())
                del da_ids, da_mask
            else:
                loss_diff = torch.tensor(0.0, device=device)

            loss = args.contrastive_weight * loss_ctr + args.diffusion_weight * loss_diff

            if global_step == 0 and is_main:
                print(f"[Pretrained baseline]  L_diff={loss_diff.item():.4f}  L_ctr={loss_ctr.item():.4f}  "
                      f"p-n={pos_sim - neg_sim:+.4f}  "
                      f"random_baseline={torch.log(torch.tensor(float(K + 1))):.4f}")

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(list(router.parameters()), max_norm=1.0)
            optimizer.step()

            global_step   += 1
            steps         += 1
            running_total += loss.item()
            running_diff  += loss_diff.item()
            running_ctr   += loss_ctr.item()

            if is_main:
                pbar.set_postfix({
                    "L_diff": f"{loss_diff.item():.4f}",
                    "L_ctr":  f"{loss_ctr.item():.4f}",
                    "p-n":    f"{pos_sim - neg_sim:+.3f}",
                })
                if use_wandb:
                    wandb.log({
                        "train/loss_total":  loss.item(),
                        "train/loss_diff":   loss_diff.item(),
                        "train/loss_ctr":    loss_ctr.item(),
                        "train/pos_sim":     pos_sim,
                        "train/neg_sim":     neg_sim,
                        "train/pos_neg_gap": pos_sim - neg_sim,
                        "train/lr":          optimizer.param_groups[0]["lr"],
                        "train/epoch":       epoch,
                    }, step=global_step)

            if is_main and args.save_every > 0 and global_step % args.save_every == 0:
                save_path  = Path(args.output_dir) / f"router_step{global_step}.pt"
                raw_router = accelerator.unwrap_model(router)
                torch.save({
                    "step":              global_step,
                    "epoch":             epoch,
                    "router_state_dict": raw_router.state_dict(),
                    "hidden_size":       hidden_size,
                    "num_layers":        num_layers,
                    "route_start":       raw_router.route_start,
                    "route_end":         raw_router.route_end,
                    "args":              vars(args),
                }, save_path)
                print(f"[Checkpoint] saved: {save_path}")

            if args.vis_every > 0 and global_step % args.vis_every == 0:
                visualize(global_step)

        if steps > 0 and is_main:
            avg_total = running_total / steps
            avg_diff  = running_diff  / steps
            avg_ctr   = running_ctr   / steps
            print(f"[Epoch {epoch}] avg_total={avg_total:.4f}  avg_diff={avg_diff:.4f}  avg_ctr={avg_ctr:.4f}")
            if use_wandb:
                wandb.log({
                    "epoch/avg_total": avg_total,
                    "epoch/avg_diff":  avg_diff,
                    "epoch/avg_ctr":   avg_ctr,
                    "epoch/epoch":     epoch,
                }, step=global_step)

    accelerator.wait_for_everyone()
    if is_main:
        final_path = Path(args.output_dir) / "router_final.pt"
        raw_router = accelerator.unwrap_model(router)
        torch.save({
            "step":              global_step,
            "epoch":             args.epochs,
            "router_state_dict": raw_router.state_dict(),
            "hidden_size":       hidden_size,
            "num_layers":        num_layers,
            "route_start":       raw_router.route_start,
            "route_end":         raw_router.route_end,
            "args":              vars(args),
        }, final_path)
        print(f"[Done] final checkpoint: {final_path}")
        if use_wandb:
            wandb.finish()


# =============================================================================
# Args
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Paths
    p.add_argument("--model_dir",      type=str, default="ckpts/Z-Image-Turbo")
    p.add_argument("--triplets_jsonl", type=str, default="data/train_triplets/counting_triplets_filtered.jsonl")
    p.add_argument("--generated_root", type=str, default="data/generated_images")
    p.add_argument("--output_dir",     type=str, default="train_text/checkpoints/counting_router")

    # Data
    p.add_argument("--resolution",        type=int,  default=1024)
    p.add_argument("--max_length",        type=int,  default=128)
    p.add_argument("--use_chat_template", action="store_true")

    # Router
    p.add_argument("--mid_dim",     type=int, default=1024)
    p.add_argument("--route_start", type=int, default=10)
    p.add_argument("--route_end",   type=int, default=21)

    # Training
    p.add_argument("--epochs",                type=int,   default=20)
    p.add_argument("--batch_size",            type=int,   default=1)
    p.add_argument("--contrastive_batch_size",type=int,   default=32)
    p.add_argument("--text_chunk_size",       type=int,   default=16)
    p.add_argument("--num_workers",           type=int,   default=2)
    p.add_argument("--lr",                    type=float, default=3e-4)
    p.add_argument("--weight_decay",          type=float, default=1e-4)
    p.add_argument("--mixed_precision",       type=str,   default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--seed",                  type=int,   default=None)

    # Loss
    p.add_argument("--num_negatives",      type=int,   default=12)
    p.add_argument("--loss_type",          type=str,   default="dcl", choices=["infonce", "dcl", "triplet"])
    p.add_argument("--temperature",        type=float, default=0.07)
    p.add_argument("--triplet_margin",     type=float, default=0.2)
    p.add_argument("--contrastive_weight", type=float, default=1.0)
    p.add_argument("--diffusion_weight",   type=float, default=0.0)

    # Logging
    p.add_argument("--save_every",    type=int,  default=200)
    p.add_argument("--vis_every",     type=int,  default=100)
    p.add_argument("--use_wandb",     action="store_true")
    p.add_argument("--wandb_project", type=str,  default="z-image-router-counting")
    p.add_argument("--wandb_run",     type=str,  default="")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
