"""
train_counting_grpo.py — Flow-GRPO-Fast for Z-Image counting (1–5 objects).

Training strategy
─────────────────
  • Reward model : Qwen2.5-VL-7B-Instruct  (counting accuracy)
  • Trainable    : context_refiner layers only  (same scope as DPO script)
  • Algorithm    : Flow-GRPO-Fast
      – Run a full ODE trajectory for each image
      – Switch to SDE only inside a randomly selected window of `sde_window_size`
        denoising steps, record (latent_before, latent_after, log_prob) per step
      – Collect rewards, compute per-prompt group advantages
      – Update policy with PPO-clip objective (ratio × advantage)

Usage example
─────────────
  accelerate launch train_text/train_counting_grpo.py \\
      --model_dir   ckpts/Z-Image-Turbo \\
      --reward_gpu  7 \\
      --output_dir  checkpoints/counting_grpo
"""

import argparse
import contextlib
import math
import os
import random
import re
import sys
from collections import defaultdict
from concurrent import futures
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ── project imports ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils import load_from_local_dir  # noqa: E402
from zimage.pipeline import calculate_shift, retrieve_timesteps  # noqa: E402
from zimage.pipeline import generate as pipeline_generate  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
NUMBER_WORDS = ["one", "two", "three", "four", "five"]
NUMBER_TO_INT = {w: i + 1 for i, w in enumerate(NUMBER_WORDS)}
INT_TO_NUMBER = {v: k for k, v in NUMBER_TO_INT.items()}

# Default nouns file (relative to repo root); overridden by --nouns_file arg
DEFAULT_NOUNS_FILE = "data/train_triplets/counting_nouns.txt"

# Minimal fallback list used only when the file is missing
_FALLBACK_NOUNS = [
    "apple", "cat", "dog", "bird", "car",
    "flower", "book", "cup", "ball", "fish",
]


def load_nouns(path: str) -> list[str]:
    """Load noun list from a text file (one noun per line), deduplicated."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    if not p.exists():
        print(f"[Nouns] WARNING: {p} not found, using built-in fallback list.")
        return _FALLBACK_NOUNS
    nouns = []
    seen: set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        n = line.strip()
        if n and n not in seen:
            nouns.append(n)
            seen.add(n)
    print(f"[Nouns] Loaded {len(nouns)} nouns from {p}")
    return nouns

# ─────────────────────────────────────────────────────────────────────────────
# 1a. Prompt contrastive helpers
# ─────────────────────────────────────────────────────────────────────────────

def compose_ctr_anchor(count_word: str, noun: str) -> str:
    return f"{count_word} {noun}"


def make_ctr_anchor_variants(anchor: str) -> list[str]:
    """Surface paraphrases of anchor — used as positives in InfoNCE."""
    base = anchor.strip()
    candidates = [
        f"a photo of {base}",
        f"an image of {base}",
        f"a picture of {base}",
        f"{base} on a white background",
        f"a clear photo of {base}",
        f"a detailed photo of {base}",
    ]
    return candidates  # never contains the original anchor itself


def make_ctr_negatives(noun: str, target_count: str) -> list[str]:
    """All other count variants of the same noun — hard negatives."""
    return [compose_ctr_anchor(c, noun) for c in NUMBER_WORDS if c != target_count]


def sample_ctr_batch(
    nouns: list[str],
    batch_size: int,
    num_negatives: int,
) -> dict[str, list[str]]:
    """
    Build anchor / positive / negative text triplets for InfoNCE.

    Returns dict with keys:
      anchors         : list[str]  length = batch_size
      positives       : list[str]  length = batch_size
      negatives_flat  : list[str]  length = batch_size * num_negatives
    """
    sampled_nouns = random.choices(nouns, k=batch_size)
    anchors, positives, negatives_flat = [], [], []
    for noun in sampled_nouns:
        count = random.choice(NUMBER_WORDS)
        anchor = compose_ctr_anchor(count, noun)
        positive = random.choice(make_ctr_anchor_variants(anchor))
        neg_pool = make_ctr_negatives(noun, count)          # ≤ 4 unique strings
        if len(neg_pool) >= num_negatives:
            chosen_negs = random.sample(neg_pool, num_negatives)
        else:
            chosen_negs = random.choices(neg_pool, k=num_negatives)
        anchors.append(anchor)
        positives.append(positive)
        negatives_flat.extend(chosen_negs)
    return {"anchors": anchors, "positives": positives, "negatives_flat": negatives_flat}


def compute_ctr_feats(
    raw_transformer,
    text_encoder,
    tokenizer,
    texts: list[str],
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    """
    text_encoder (frozen, no grad) → cap_embedder + context_refiner (trainable, grad)
    → mean-pool valid tokens → L2-norm → (N, D_out) float32 tensor.

    This is the standalone text-only path for contrastive learning.
    It does NOT go through the full DiT forward pass.
    """
    enc = tokenizer(
        texts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc.input_ids.to(device)
    attn_mask  = enc.attention_mask.to(device).bool()

    # ── Frozen text encoder ────────────────────────────────────────────────
    with torch.no_grad():
        out = text_encoder(
            input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True
        )
        hidden = out.hidden_states[-2].detach()  # (N, L, D_text)

    # ── Trainable cap_embedder + context_refiner ───────────────────────────
    N, L, _ = hidden.shape
    dtype = next(raw_transformer.parameters()).dtype

    cap_feats = raw_transformer.cap_embedder(hidden.to(dtype))
    cap_feats = cap_feats.clone()
    cap_feats[~attn_mask] = raw_transformer.cap_pad_token.to(dtype)

    pos_ids = torch.zeros((N, L, 3), dtype=torch.int32, device=device)
    pos_ids[:, :, 0] = (
        torch.arange(1, L + 1, dtype=torch.int32, device=device).unsqueeze(0).expand(N, -1)
    )
    freqs = raw_transformer.rope_embedder(pos_ids.view(-1, 3)).view(N, L, -1)

    refined = cap_feats
    for layer in raw_transformer.context_refiner:
        refined = layer(refined, attn_mask, freqs)   # (N, L, D_out)

    # ── Mean-pool over valid tokens ────────────────────────────────────────
    mask_f = attn_mask.float().unsqueeze(-1)         # (N, L, 1)
    pooled = (refined.float() * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-6)
    return F.normalize(pooled, dim=-1)               # (N, D_out)


def infonce_loss(
    ea: torch.Tensor,   # (B, D)  anchor features
    ep: torch.Tensor,   # (B, D)  positive features
    en: torch.Tensor,   # (B, K, D) negative features
    temperature: float,
) -> torch.Tensor:
    """
    InfoNCE: for each anchor, one positive vs K negatives.
    Loss = -log( exp(a·p/τ) / (exp(a·p/τ) + Σ exp(a·n_k/τ)) )
    """
    ea, ep, en = ea.float(), ep.float(), en.float()
    pos_logit = (ea * ep).sum(dim=-1, keepdim=True) / temperature          # (B, 1)
    neg_logits = (ea.unsqueeze(1) * en).sum(dim=-1) / temperature          # (B, K)
    logits = torch.cat([pos_logit, neg_logits], dim=1)                     # (B, 1+K)
    labels = torch.zeros(ea.shape[0], dtype=torch.long, device=ea.device)
    return F.cross_entropy(logits, labels)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SDE step with log-probability  (Z-Image adaptation of flow_grpo)
# ─────────────────────────────────────────────────────────────────────────────

def sde_step_with_logprob(
    scheduler,
    velocity: torch.Tensor,          # raw transformer output  ≈ x1 − x0
    timestep: torch.Tensor,          # scalar or (B,) raw scheduler timestep
    sample: torch.Tensor,            # x_t  shape (B, C, H, W)
    noise_level: float = 0.8,
    prev_sample: torch.Tensor | None = None,   # if given, evaluate log-prob at it
    sde_type: str = "cps",           # "sde" (augmented EM) or "cps" (coeff-preserving)
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One reverse-SDE step (or ODE when noise_level=0).

    sde_type="sde"  – Augmented Euler–Maruyama (Flow-GRPO §3.2)
    sde_type="cps"  – Coefficients-Preserving Sampling (recommended for fast variant)

    Returns
    ───────
    prev_sample      : (B, C, H, W)
    log_prob         : (B,)
    prev_sample_mean : (B, C, H, W)
    std_dev_t        : scalar tensor
    """
    velocity = velocity.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    t_scalar = timestep.view(-1)[0]
    step_idx = scheduler.index_for_timestep(t_scalar)
    sigma      = scheduler.sigmas[step_idx    ].to(velocity.device)
    sigma_next = scheduler.sigmas[step_idx + 1].to(velocity.device)
    dt = sigma_next - sigma  # negative: sigma decreases during denoising

    if noise_level <= 0.0:
        # Pure ODE (Z-Image sign convention: noise_pred = −velocity)
        prev_sample_mean = sample + dt * (-velocity)
        log_prob = torch.zeros(sample.shape[0], device=sample.device)
        std_dev_t = torch.zeros(1, device=sample.device)
        return prev_sample_mean, log_prob, prev_sample_mean, std_dev_t

    if sde_type == "cps":
        # ── Coefficients-Preserving Sampling ──────────────────────────────
        # Equivalent to flow_grpo sd3_sde_with_logprob.py, sde_type='cps'
        # sigma here = current σ, sigma_next = σ at next (more denoised) step
        std_dev_t = sigma_next * math.sin(noise_level * math.pi / 2.0)

        # Predicted x0 and x1 from current latent and velocity
        pred_x0 = sample - sigma * velocity           # ≈ x_0
        pred_x1 = sample + velocity * (1.0 - sigma)  # ≈ x_1

        determ_coef = (sigma_next ** 2 - std_dev_t ** 2).clamp(min=0.0).sqrt()
        prev_sample_mean = pred_x0 * (1.0 - sigma_next) + pred_x1 * determ_coef

        if prev_sample is None:
            prev_sample = prev_sample_mean + std_dev_t * torch.randn_like(sample)

        # CPS log-prob omits constant terms (they cancel in the ratio anyway)
        log_prob = -((prev_sample.detach() - prev_sample_mean) ** 2)
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))  # → (B,)

    else:
        # ── Augmented Euler–Maruyama ("sde") ──────────────────────────────
        sigma_c = sigma.clamp(1e-6, 1.0 - 1e-6)
        std_dev_t = (sigma_c / (1.0 - sigma_c)).sqrt() * noise_level

        prev_sample_mean = (
            sample * (1.0 + std_dev_t ** 2 / (2.0 * sigma) * dt)
            + velocity * (1.0 + std_dev_t ** 2 * (1.0 - sigma) / (2.0 * sigma)) * dt
        )

        if prev_sample is None:
            prev_sample = prev_sample_mean + std_dev_t * (-dt).sqrt() * torch.randn_like(sample)

        diffusion_std = std_dev_t * (-dt).sqrt()
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2.0 * diffusion_std ** 2)
            - diffusion_std.log()
            - 0.5 * math.log(2.0 * math.pi)
        )
        log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))  # → (B,)

    return prev_sample, log_prob, prev_sample_mean, std_dev_t


# ─────────────────────────────────────────────────────────────────────────────
# 2. Text-encoding helpers  (text_encoder + context_refiner)
# ─────────────────────────────────────────────────────────────────────────────

def encode_text_hidden(tokenizer, text_encoder, prompts, max_length, device):
    """
    Tokenize prompts and run the text encoder.
    Returns (text_hidden, attn_mask); both are detached, no grad.
    """
    formatted = []
    for p in prompts:
        msg = [{"role": "user", "content": p}]
        formatted.append(
            tokenizer.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
        )
    enc = tokenizer(
        formatted,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc.input_ids.to(device)
    attn_mask = enc.attention_mask.to(device).bool()
    with torch.no_grad():
        out = text_encoder(
            input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True
        )
        hidden = out.hidden_states[-2].detach()  # (B, L, D)
    return hidden, attn_mask


def run_context_refiner(raw_transformer, text_hidden, attn_mask):
    """
    [Unused in main training loop — kept for debugging / standalone testing]
    Run cap_embedder + context_refiner standalone.
    NOTE: do NOT pass the output of this function to transformer_forward_velocity;
    the transformer's forward already runs cap_embedder + context_refiner internally.

    Returns list of B variable-length tensors, each (valid_len, D).
    """
    B, L, _ = text_hidden.shape
    device = text_hidden.device
    dtype = next(raw_transformer.parameters()).dtype

    cap_feats = raw_transformer.cap_embedder(text_hidden.to(dtype))
    cap_feats = cap_feats.clone()
    cap_feats[~attn_mask] = raw_transformer.cap_pad_token.to(dtype)

    pos_ids = torch.zeros((B, L, 3), dtype=torch.int32, device=device)
    pos_ids[:, :, 0] = (
        torch.arange(1, L + 1, dtype=torch.int32, device=device).unsqueeze(0).expand(B, -1)
    )
    freqs = raw_transformer.rope_embedder(pos_ids.view(-1, 3)).view(B, L, -1)

    refined = cap_feats
    for layer in raw_transformer.context_refiner:
        refined = layer(refined, attn_mask, freqs)

    return [refined[i][attn_mask[i]].to(dtype) for i in range(B)]


def transformer_forward_velocity(raw_transformer, latents, t_raw, text_hidden, attn_mask):
    """
    One DiT forward pass; returns velocity ≈ x1−x0  shape (B, C, H, W).

    Passes raw text_hidden (2560-dim) directly to the transformer so that
    the transformer's internal cap_embedder + context_refiner run once.
    Gradients flow through context_refiner naturally when called with grad enabled.
    """
    dtype = next(raw_transformer.parameters()).dtype
    B = latents.shape[0]
    t_norm = (1000.0 - t_raw) / 1000.0          # Z-Image normalisation
    if t_norm.ndim == 0:
        t_norm = t_norm.expand(B)
    lat_in = latents.to(dtype).unsqueeze(2)      # (B, C, 1, H, W)
    lat_list = list(lat_in.unbind(0))            # list[(C, 1, H, W)]
    # Transformer expects list of (valid_len, raw_dim) tensors — raw 2560-dim features
    cap_feats_list = [text_hidden[i][attn_mask[i]].to(dtype) for i in range(B)]
    raw_out_list = raw_transformer(lat_list, t_norm, cap_feats_list)[0]
    velocity = torch.stack([o.squeeze(1).float() for o in raw_out_list])  # (B,C,H,W)
    return velocity


# ─────────────────────────────────────────────────────────────────────────────
# 3. Sampling pipeline with log-probability  (Flow-GRPO-Fast for Z-Image)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def sample_with_logprob(
    transformer,
    vae,
    text_encoder,
    tokenizer,
    scheduler,
    prompts: list[str],
    height: int,
    width: int,
    num_inference_steps: int,
    noise_level: float,
    sde_window_size: int,
    sde_window_range: tuple[int, int],
    max_length: int,
    device: torch.device,
    sde_type: str = "cps",
    generator=None,
) -> tuple[list[Image.Image], dict]:
    """
    Generates images and records (latents, next_latents, log_probs, timesteps)
    for the randomly selected SDE window.

    Returns
    ───────
    images     : list of PIL images (len = B)
    trajectory : dict with tensors shaped (B, W, C, H, W) / (B, W) / (W,)
    """
    B = len(prompts)
    raw = transformer.module if hasattr(transformer, "module") else transformer

    # ── Text encoding ──────────────────────────────────────────────────────
    text_hidden, attn_mask = encode_text_hidden(
        tokenizer, text_encoder, prompts, max_length, device
    )
    # raw text_hidden (2560-dim) is stored and re-used during training;
    # the transformer handles cap_embedder + context_refiner internally.

    # ── Latent initialisation ──────────────────────────────────────────────
    vae_scale = 2 ** (len(vae.config.block_out_channels) - 1) * 2
    H_lat = 2 * (height // vae_scale)
    W_lat = 2 * (width // vae_scale)
    latents = torch.randn(
        B, raw.in_channels, H_lat, W_lat,
        generator=generator, device=device, dtype=torch.float32
    )

    # ── Timestep schedule ──────────────────────────────────────────────────
    image_seq_len = (H_lat // 2) * (W_lat // 2)
    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    scheduler.sigma_min = 0.0
    timesteps, _ = retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=None, mu=mu)

    # ── SDE window selection ───────────────────────────────────────────────
    n_steps = len(timesteps)
    lo = max(sde_window_range[0], 0)
    hi = max(lo, min(sde_window_range[1] - sde_window_size, n_steps - sde_window_size))
    sde_start = random.randint(lo, hi)
    sde_end = sde_start + sde_window_size  # exclusive

    all_latents_before: list[torch.Tensor] = []
    all_latents_after: list[torch.Tensor] = []
    all_log_probs: list[torch.Tensor] = []
    all_timesteps: list[torch.Tensor] = []

    for i, t in enumerate(timesteps):
        if t == 0 and i == n_steps - 1:
            continue

        in_window = sde_start <= i < sde_end
        cur_noise_level = noise_level if in_window else 0.0

        if in_window:
            all_latents_before.append(latents.clone())
            all_timesteps.append(t.clone())

        t_batch = t.expand(B)
        velocity = transformer_forward_velocity(raw, latents, t_batch, text_hidden, attn_mask)

        latents, log_prob, _, _ = sde_step_with_logprob(
            scheduler, velocity, t_batch[:1], latents,
            noise_level=cur_noise_level, sde_type=sde_type,
        )

        if in_window:
            all_latents_after.append(latents.clone())
            all_log_probs.append(log_prob.clone())  # (B,)

    # ── Decode to PIL ──────────────────────────────────────────────────────
    shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
    dec = (latents.to(vae.dtype) / vae.config.scaling_factor) + shift_factor
    imgs_tensor = vae.decode(dec, return_dict=False)[0]
    imgs_tensor = (imgs_tensor / 2 + 0.5).clamp(0, 1)
    imgs_np = imgs_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
    images = [Image.fromarray((img * 255).round().astype("uint8")) for img in imgs_np]

    # ── Pack trajectory ────────────────────────────────────────────────────
    W = len(all_latents_before)
    trajectory = {
        "latents":      torch.stack(all_latents_before, dim=1),  # (B, W, C, H, W)
        "next_latents": torch.stack(all_latents_after, dim=1),   # (B, W, C, H, W)
        "log_probs":    torch.stack(all_log_probs, dim=1),        # (B, W)
        "timesteps":    torch.stack(all_timesteps),               # (W,)
        "text_hidden":  text_hidden,   # (B, L, D) — reuse for training
        "attn_mask":    attn_mask,     # (B, L)
    }
    return images, trajectory


# ─────────────────────────────────────────────────────────────────────────────
# 4. Qwen-VL counting reward
# ─────────────────────────────────────────────────────────────────────────────

class QwenVLCountingReward:
    """
    Wraps Qwen2.5-VL-7B-Instruct to score counting accuracy.

    Reward mapping:
      offset = |predicted_count − target_count|
      reward = max(0.0, 1.0 − 0.5 * offset)
    """

    _PROMPT_TEMPLATE = (
        "Look at this image carefully. "
        "How many {noun}s can you count? "
        "Reply with ONLY a single integer number, nothing else."
    )

    def __init__(self, model_path: str, device: str = "cuda"):
        from transformers import AutoProcessor

        # Qwen3-VL → Qwen3VLForConditionalGeneration (transformers ≥ 4.52)
        # Qwen2.5-VL → Qwen2_5_VLForConditionalGeneration
        # Fall back to AutoModel so any future Qwen VL version works out-of-box.
        try:
            from transformers import Qwen3VLForConditionalGeneration as _VLModel
        except ImportError:
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as _VLModel
            except ImportError:
                from transformers import AutoModelForVision2Seq as _VLModel

        self.device = device
        try:
            import flash_attn  # noqa: F401
            _attn_impl = "flash_attention_2"
        except ImportError:
            _attn_impl = "sdpa"
        self.model = _VLModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation=_attn_impl,
            device_map=None,
        ).to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        # Decoder-only models require left-padding for correct batch generation.
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

    @torch.no_grad()
    def __call__(
        self,
        images: list[Image.Image],
        target_nouns: list[str],
        target_counts: list[int],
    ) -> list[float]:
        import base64
        from io import BytesIO

        def img_to_base64(img: Image.Image) -> str:
            buf = BytesIO()
            img.save(buf, format="PNG")
            return "data:image;base64," + base64.b64encode(buf.getvalue()).decode()

        messages = []
        for img, noun in zip(images, target_nouns):
            messages.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": img_to_base64(img)},
                    {"type": "text",  "text": self._PROMPT_TEMPLATE.format(noun=noun)},
                ],
            }])

        texts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
        except ImportError:
            # Fallback without qwen_vl_utils
            inputs = self.processor(
                text=texts, padding=True, return_tensors="pt"
            ).to(self.device)

        gen_ids = self.model.generate(**inputs, max_new_tokens=8)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, gen_ids)]
        texts_out = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        rewards = []
        for text, target in zip(texts_out, target_counts):
            nums = re.findall(r"\d+", text.strip())
            if nums:
                predicted = int(nums[0])
                offset = abs(predicted - target)
                reward = max(0.0, 1.0 - 0.5 * offset)
            else:
                reward = 0.0
            rewards.append(reward)
        return rewards


# ─────────────────────────────────────────────────────────────────────────────
# 5. Per-prompt stat tracker  (group-wise advantage normalization)
# ─────────────────────────────────────────────────────────────────────────────

class PerPromptStatTracker:
    """
    Standard GRPO within-group advantage normalisation.

    For each unique prompt, normalise rewards by the mean and std of the group
    (the group_size samples generated for that prompt in this iteration).
    This is the formulation from the GRPO paper and is independent of history,
    ensuring advantages are always well-scaled even at the start of training.

    If all samples in a group share the same reward (std ≈ 0), advantages for
    that group remain 0 — the sample will be skipped by the caller.
    """

    def __init__(self, global_std: bool = True):
        # global_std kept for API compatibility; within-group std is always used
        self.global_std = global_std

    def update(self, prompts: list[str], rewards: np.ndarray) -> np.ndarray:
        prompts_arr = np.array(prompts)
        advantages = np.zeros_like(rewards)
        for p in np.unique(prompts_arr):
            mask = prompts_arr == p
            gr = rewards[mask]                     # rewards for this group
            mean = gr.mean()
            std = gr.std() + 1e-6                  # per-group std
            advantages[mask] = (gr - mean) / std
        return advantages

    def clear(self):
        pass  # stateless — nothing to clear


# ─────────────────────────────────────────────────────────────────────────────
# 6. Counting prompt dataset
# ─────────────────────────────────────────────────────────────────────────────

class CountingPromptDataset(Dataset):
    """
    Each item is (prompt_str, noun, count_int).
    Prompts follow the template: "a photo of {count_word} {noun}s on a white background"
    """

    PROMPT_TEMPLATE = "a photo of {count_word} {noun}s on a white background"

    def __init__(self, nouns: list[str], max_count: int = 5):
        self.items: list[tuple[str, str, int]] = []
        for noun in nouns:
            for cnt in range(1, max_count + 1):
                cw = NUMBER_WORDS[cnt - 1]
                prompt = self.PROMPT_TEMPLATE.format(count_word=cw, noun=noun)
                self.items.append((prompt, noun, cnt))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        prompt, noun, count = self.items[idx]
        return {"prompt": prompt, "noun": noun, "count": count}

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        return {
            "prompts":  [b["prompt"] for b in batch],
            "nouns":    [b["noun"]   for b in batch],
            "counts":   [b["count"]  for b in batch],
        }


# ─────────────────────────────────────────────────────────────────────────────
# 7. Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def unfreeze_context_refiner(transformer) -> list[str]:
    """Freeze everything, then unfreeze context_refiner layers."""
    for p in transformer.parameters():
        p.requires_grad_(False)
    names = []
    for name, p in transformer.named_parameters():
        if "context_refiner" in name.lower():
            p.requires_grad_(True)
            names.append(name)
    if not names:
        raise RuntimeError("No context_refiner parameters found in transformer.")
    return names


def compute_train_logprob(
    transformer,
    scheduler,
    sample_latents: torch.Tensor,       # (B, C, H, W) latent before step j
    sample_next_latents: torch.Tensor,  # (B, C, H, W) latent after step j
    sample_timestep: torch.Tensor,      # scalar: timestep for step j
    text_hidden: torch.Tensor,          # (B, L, D)
    attn_mask: torch.Tensor,            # (B, L)
    noise_level: float,
    sde_type: str = "cps",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Re-computes log_prob under the current policy (with gradients through
    context_refiner) for one SDE window step j.

    Returns (log_prob, prev_sample_mean, std_dev_t).
    """
    raw = transformer.module if hasattr(transformer, "module") else transformer
    B = sample_latents.shape[0]

    # Gradient flows through context_refiner inside transformer_forward_velocity
    t_batch = sample_timestep.expand(B)
    velocity = transformer_forward_velocity(raw, sample_latents, t_batch, text_hidden, attn_mask)

    _, log_prob, prev_mean, std_dev = sde_step_with_logprob(
        scheduler,
        velocity,
        t_batch[:1],
        sample_latents.float(),
        noise_level=noise_level,
        prev_sample=sample_next_latents.float(),
        sde_type=sde_type,
    )
    return log_prob, prev_mean, std_dev


@torch.no_grad()
def visualize(
    transformer, vae, text_encoder, tokenizer, scheduler,
    components, vis_prompts, resolution, device, vis_dir, global_step,
    use_wandb, seed,
):
    raw = transformer.module if hasattr(transformer, "module") else transformer
    raw.eval()
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed + 9999)
    images = pipeline_generate(
        transformer=raw,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=components["scheduler"],
        prompt=vis_prompts,
        height=resolution,
        width=resolution,
        num_inference_steps=8,
        guidance_scale=0.0,
        generator=gen,
    )
    vis_dir.mkdir(parents=True, exist_ok=True)
    for i, (img, p) in enumerate(zip(images, vis_prompts)):
        img.save(vis_dir / f"step{global_step:06d}_p{i}.png")
    if use_wandb:
        wandb.log(
            {"vis": [wandb.Image(img, caption=p) for img, p in zip(images, vis_prompts)]},
            step=global_step,
        )
    raw.train()


# ─────────────────────────────────────────────────────────────────────────────
# 8. Main
# ─────────────────────────────────────────────────────────────────────────────

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
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or None,
            config=vars(args),
        )
        if is_main:
            print(f"[WandB] {wandb.run.url}")

    # ── Load Z-Image model ─────────────────────────────────────────────────
    if is_main:
        print(f"[Init] Loading Z-Image from {args.model_dir} ...")
    components = load_from_local_dir(
        args.model_dir, device="cpu", dtype=torch.bfloat16, verbose=is_main
    )
    transformer = components["transformer"].to(device)
    vae         = components["vae"].to(device)
    text_encoder = components["text_encoder"].to(device)
    tokenizer   = components["tokenizer"]
    scheduler   = components["scheduler"]

    # Decoder-only text encoders need left-padding so that the last token
    # (used as the sequence embedding) is always a real token, not a pad token.
    tokenizer.padding_side = "left"

    for p in vae.parameters():
        p.requires_grad_(False)
    for p in text_encoder.parameters():
        p.requires_grad_(False)
    vae.eval()
    text_encoder.eval()

    trainable_names = unfreeze_context_refiner(transformer)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if is_main:
            print("[Init] Gradient checkpointing enabled on DiT layers.")
    if is_main:
        n_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        print(f"[Init] Trainable params (context_refiner): {n_params:,}")
        print(f"[Init] sde_type={args.sde_type}  "
              f"sde_window_size={args.sde_window_size}  "
              f"sde_window_range={args.sde_window_range}  "
              f"noise_level={args.noise_level}")

    # ── Reward model (only on main process or designated GPU) ─────────────
    reward_fn: QwenVLCountingReward | None = None
    if is_main:
        reward_device = f"cuda:{args.reward_gpu}" if args.reward_gpu >= 0 else str(device)
        if is_main:
            print(f"[Init] Loading Qwen-VL reward model on {reward_device} ...")
        reward_fn = QwenVLCountingReward(args.reward_model_path, device=reward_device)

    # ── Dataset & dataloader ───────────────────────────────────────────────
    nouns = load_nouns(args.nouns_file)
    dataset = CountingPromptDataset(nouns=nouns, max_count=args.max_count)
    # Each iteration: sample `train_batch_size` unique prompts
    loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=CountingPromptDataset.collate_fn,
        drop_last=True,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    transformer, optimizer, loader = accelerator.prepare(transformer, optimizer, loader)

    # ── Stat tracker & vis setup ───────────────────────────────────────────
    stat_tracker = PerPromptStatTracker(global_std=args.global_std)
    vis_dir = Path(args.output_dir) / "vis"
    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    VIS_PROMPTS = [
        "a photo of one apple on a white background",
        "a photo of three cats on a white background",
        "a photo of five birds on a white background",
        "a photo of two dogs on a white background",
    ]

    # ── Thread pool for async reward computation ───────────────────────────
    executor = futures.ThreadPoolExecutor(max_workers=2)

    global_step = 0
    raw = transformer.module if hasattr(transformer, "module") else transformer

    # ── Visualise at step 0 ────────────────────────────────────────────────
    if is_main and args.vis_every > 0:
        visualize(
            transformer, vae, text_encoder, tokenizer, scheduler,
            components, VIS_PROMPTS, args.resolution, device,
            vis_dir, 0, use_wandb, args.seed,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Training loop
    # ─────────────────────────────────────────────────────────────────────
    for epoch in range(1, args.num_epochs + 1):
        transformer.train()
        train_iter = iter(loader)
        steps_per_epoch = len(loader) * args.num_batches_per_epoch
        info_epoch: dict[str, list[float]] = defaultdict(list)

        for _batch_idx in tqdm(
            range(args.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sample+train",
            disable=not is_main,
        ):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(loader)
                batch = next(train_iter)

            base_prompts: list[str] = batch["prompts"]
            base_nouns:   list[str] = batch["nouns"]
            base_counts:  list[int] = batch["counts"]
            B_unique = len(base_prompts)

            # ── Expand prompts by group size ──────────────────────────────
            prompts       = [p  for p in base_prompts for _ in range(args.group_size)]
            target_nouns  = [n  for n in base_nouns   for _ in range(args.group_size)]
            target_counts = [c  for c in base_counts  for _ in range(args.group_size)]
            B = len(prompts)   # B_unique * group_size

            # ── Sample trajectories (no grad) ─────────────────────────────
            transformer.eval()
            with torch.no_grad():
                images, traj = sample_with_logprob(
                    transformer=transformer,
                    vae=vae,
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    scheduler=scheduler,
                    prompts=prompts,
                    height=args.resolution,
                    width=args.resolution,
                    num_inference_steps=args.num_inference_steps,
                    noise_level=args.noise_level,
                    sde_window_size=args.sde_window_size,
                    sde_window_range=args.sde_window_range,
                    max_length=args.max_length,
                    device=device,
                    sde_type=args.sde_type,
                )
            transformer.train()

            # Trajectory tensors
            latents_traj      = traj["latents"]       # (B, W, C, H, W)
            next_latents_traj = traj["next_latents"]  # (B, W, C, H, W)
            old_log_probs     = traj["log_probs"]     # (B, W)
            timesteps_traj    = traj["timesteps"]     # (W,)
            text_hidden       = traj["text_hidden"]   # (B, L, D)
            attn_mask         = traj["attn_mask"]     # (B, L)
            W = latents_traj.shape[1]

            # ── Compute rewards (async on main; others wait) ───────────────
            if is_main:
                reward_future = executor.submit(
                    reward_fn, images, target_nouns, target_counts
                )
                rewards_raw = reward_future.result()
                rewards_np = np.array(rewards_raw, dtype=np.float32)
            else:
                rewards_np = np.zeros(B, dtype=np.float32)

            # Broadcast rewards across processes
            rewards_t = torch.tensor(rewards_np, device=device, dtype=torch.float32)
            if accelerator.num_processes > 1:
                rewards_t = accelerator.reduce(rewards_t.unsqueeze(0), reduction="sum").squeeze(0)

            # ── Compute per-prompt advantages ──────────────────────────────
            if is_main:
                advantages_np = stat_tracker.update(prompts, rewards_t.cpu().numpy())
            else:
                advantages_np = np.zeros(B, dtype=np.float32)

            advantages = torch.as_tensor(advantages_np, device=device, dtype=torch.float32)
            advantages = advantages.clamp(-args.adv_clip_max, args.adv_clip_max)

            # Drop samples where all advantages in group are zero (all same reward)
            group_adv = advantages.view(B_unique, args.group_size)
            active_mask = group_adv.abs().sum(dim=1) != 0  # (B_unique,)
            active_mask = active_mask.repeat_interleave(args.group_size)  # (B,)
            if active_mask.sum() == 0:
                if is_main:
                    print(f"[Step {global_step}] All advantages zero, skipping.")
                continue

            # Log reward stats
            if is_main:
                wandb_reward_dict = {
                    "train/reward_mean":  rewards_t.mean().item(),
                    "train/reward_std":   rewards_t.std().item(),
                    "train/adv_mean":     advantages.mean().item(),
                    "train/adv_abs_mean": advantages.abs().mean().item(),
                    "train/zero_adv_frac": (advantages.abs() < 1e-4).float().mean().item(),
                }
                if use_wandb:
                    wandb.log(wandb_reward_dict, step=global_step)
                    # Log generated images with reward captions (one group per prompt)
                    vis_imgs = []
                    rewards_list = rewards_t.cpu().tolist()
                    for i, (img, prompt, r) in enumerate(zip(images, prompts, rewards_list)):
                        caption = f"r={r:.2f} | {prompt}"
                        vis_imgs.append(wandb.Image(img, caption=caption))
                    wandb.log({"train/samples": vis_imgs}, step=global_step)

            # ── Policy gradient update ─────────────────────────────────────
            # For each SDE window step j, re-compute log_prob under current policy
            info_step: dict[str, list] = defaultdict(list)

            for j in range(W):
                lat_j      = latents_traj[:, j]       # (B, C, H, W)
                next_lat_j = next_latents_traj[:, j]  # (B, C, H, W)
                old_lp_j   = old_log_probs[:, j]      # (B,)
                t_j        = timesteps_traj[j]         # scalar

                # Apply active mask
                lat_j      = lat_j[active_mask]
                next_lat_j = next_lat_j[active_mask]
                old_lp_j   = old_lp_j[active_mask]
                adv_j      = advantages[active_mask]
                th_j       = text_hidden[active_mask]
                am_j       = attn_mask[active_mask]

                with accelerator.accumulate(transformer):
                    new_lp_j, prev_mean, std_dev = compute_train_logprob(
                        transformer=transformer,
                        scheduler=scheduler,
                        sample_latents=lat_j,
                        sample_next_latents=next_lat_j,
                        sample_timestep=t_j,
                        text_hidden=th_j,
                        attn_mask=am_j,
                        noise_level=args.noise_level,
                        sde_type=args.sde_type,
                    )

                    # GRPO clipped ratio loss
                    ratio = (new_lp_j - old_lp_j.detach()).exp()
                    unclipped = -adv_j * ratio
                    clipped   = -adv_j * ratio.clamp(
                        1.0 - args.clip_range, 1.0 + args.clip_range
                    )
                    loss = torch.maximum(unclipped, clipped).mean()

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                # Diagnostics
                approx_kl = 0.5 * ((new_lp_j - old_lp_j.detach()) ** 2).mean()
                clipfrac  = (ratio.detach() - 1.0).abs().gt(args.clip_range).float().mean()

                info_step["loss"].append(loss.item())
                info_step["approx_kl"].append(approx_kl.item())
                info_step["clipfrac"].append(clipfrac.item())
                info_step["ratio_mean"].append(ratio.detach().mean().item())

            # ── Prompt contrastive loss (text-only, independent of images) ────
            loss_ctr_val = 0.0
            if args.contrastive_weight > 0:
                ctr_data = sample_ctr_batch(nouns, args.contrastive_batch_size, args.num_ctr_negatives)
                all_texts = ctr_data["anchors"] + ctr_data["positives"] + ctr_data["negatives_flat"]
                with accelerator.accumulate(transformer):
                    all_feats = compute_ctr_feats(
                        raw, text_encoder, tokenizer, all_texts, args.max_length, device
                    )
                    B_ctr = len(ctr_data["anchors"])
                    K     = args.num_ctr_negatives
                    ea      = all_feats[:B_ctr]
                    ep      = all_feats[B_ctr : 2 * B_ctr]
                    en_flat = all_feats[2 * B_ctr :]
                    en = en_flat.view(B_ctr, K, -1)

                    with torch.no_grad():
                        pos_sim = (ea * ep).sum(dim=-1).mean().item()
                        neg_sim = (ea.unsqueeze(1) * en).sum(dim=-1).mean().item()

                    loss_ctr = infonce_loss(ea, ep, en, args.temperature) * args.contrastive_weight
                    loss_ctr_val = loss_ctr.item()

                    accelerator.backward(loss_ctr)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            else:
                pos_sim = neg_sim = 0.0

            if accelerator.sync_gradients:
                global_step += 1
                step_metrics = {k: float(np.mean(v)) for k, v in info_step.items()}
                step_metrics["loss_ctr"] = loss_ctr_val
                for k, v in step_metrics.items():
                    info_epoch[k].append(v)

                if is_main and use_wandb:
                    wandb.log(
                        {
                            **{f"train/{k}": v for k, v in step_metrics.items()},
                            "train/ctr_pos_sim": pos_sim,
                            "train/ctr_neg_sim": neg_sim,
                            "train/ctr_pos_neg_gap": pos_sim - neg_sim,
                        },
                        step=global_step,
                    )
                if is_main:
                    print(
                        f"[Step {global_step}] "
                        f"loss={step_metrics['loss']:.4f}  "
                        f"kl={step_metrics['approx_kl']:.4f}  "
                        f"clip={step_metrics['clipfrac']:.3f}  "
                        f"reward={rewards_t.mean().item():.3f}  "
                        f"ctr={loss_ctr_val:.4f}  p-n={pos_sim - neg_sim:+.3f}"
                    )

                # Save checkpoint
                if (is_main and args.save_every > 0
                        and global_step % args.save_every == 0):
                    ckpt_path = Path(args.output_dir) / f"refiner_step{global_step}.pt"
                    unwrapped = accelerator.unwrap_model(transformer)
                    torch.save(
                        {
                            "step": global_step,
                            "epoch": epoch,
                            "state_dict": unwrapped.state_dict(),
                            "args": vars(args),
                        },
                        ckpt_path,
                    )
                    print(f"[Checkpoint] {ckpt_path}")

                # Visualise
                if (is_main and args.vis_every > 0
                        and global_step % args.vis_every == 0):
                    visualize(
                        transformer, vae, text_encoder, tokenizer, scheduler,
                        components, VIS_PROMPTS, args.resolution, device,
                        vis_dir, global_step, use_wandb, args.seed,
                    )

        # Epoch summary
        if is_main:
            epoch_summary = {k: float(np.mean(v)) for k, v in info_epoch.items()}
            print(
                f"[Epoch {epoch}] "
                + "  ".join(f"{k}={v:.4f}" for k, v in epoch_summary.items())
            )
            if use_wandb:
                wandb.log(
                    {f"epoch/{k}": v for k, v in epoch_summary.items()},
                    step=global_step,
                )

    # ── Final save ────────────────────────────────────────────────────────
    accelerator.wait_for_everyone()
    if is_main:
        final_path = Path(args.output_dir) / "refiner_final.pt"
        unwrapped = accelerator.unwrap_model(transformer)
        torch.save(
            {
                "step": global_step,
                "epoch": args.num_epochs,
                "state_dict": unwrapped.state_dict(),
                "args": vars(args),
            },
            final_path,
        )
        print(f"[Done] Final checkpoint: {final_path}")
        if use_wandb:
            wandb.finish()


# ─────────────────────────────────────────────────────────────────────────────
# 9. Argument parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Flow-GRPO-Fast for Z-Image counting")

    # ── Paths ─────────────────────────────────────────────────────────────
    p.add_argument("--model_dir", type=str, default="ckpts/Z-Image-Turbo",
                   help="Path to Z-Image model directory")
    p.add_argument("--reward_model_path", type=str,
                   default="Qwen/Qwen3-VL-8B-Instruct",
                   help="HuggingFace model ID or local path for Qwen-VL reward")
    p.add_argument("--reward_gpu", type=int, default=-1,
                   help="GPU index for reward model; -1 = same as training GPU")
    p.add_argument("--output_dir", type=str,
                   default="checkpoints/counting_grpo")
    p.add_argument("--nouns_file", type=str,
                   default=DEFAULT_NOUNS_FILE,
                   help="Path to noun list file (one noun per line); "
                        "relative paths resolved from repo root")

    # ── Model / generation ────────────────────────────────────────────────
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--max_length", type=int, default=128,
                   help="Max tokeniser length for prompts")
    p.add_argument("--num_inference_steps", type=int, default=10,
                   help="Total ODE/SDE denoising steps per image")

    # ── GRPO-Fast SDE window ──────────────────────────────────────────────
    # Defaults aligned with flow_grpo fast configs (geneval_sd3_fast_nocfg etc.)
    p.add_argument("--sde_type", type=str, default="cps",
                   choices=["cps", "sde"],
                   help="SDE step type: 'cps' (Coefficients-Preserving, recommended) "
                        "or 'sde' (Augmented Euler-Maruyama)")
    p.add_argument("--sde_window_size", type=int, default=3,
                   help="Number of SDE steps in the training window (GRPO-Fast)")
    p.add_argument("--sde_window_range", type=int, nargs=2, default=[0, 5],
                   metavar=("START", "END"),
                   help="Range [start, end) for SDE window placement; "
                        "default (0, num_steps//2) → (0,5) for num_steps=10")
    p.add_argument("--noise_level", type=float, default=0.8,
                   help="SDE noise scale (0 = ODE)")

    # ── Training ──────────────────────────────────────────────────────────
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--num_batches_per_epoch", type=int, default=4,
                   help="Number of sample-then-train rounds per epoch")
    p.add_argument("--train_batch_size", type=int, default=2,
                   help="Unique prompts per round (per GPU)")
    p.add_argument("--group_size", type=int, default=4,
                   help="Images per prompt (group for advantage normalisation)")
    p.add_argument("--max_count", type=int, default=5,
                   help="Maximum object count in training prompts")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--mixed_precision", type=str, default="bf16",
                   choices=["no", "fp16", "bf16"])
    p.add_argument("--gradient_checkpointing", action="store_true", default=False,
                   help="Enable gradient checkpointing on DiT layers to reduce memory usage")

    # ── Prompt contrastive loss ───────────────────────────────────────────
    p.add_argument("--contrastive_weight", type=float, default=1.0,
                   help="Weight for InfoNCE prompt contrastive loss; 0 = disabled")
    p.add_argument("--temperature", type=float, default=0.07,
                   help="InfoNCE temperature")
    p.add_argument("--num_ctr_negatives", type=int, default=4,
                   help="Hard negatives per anchor (different-count variants; max 4 for 5 number words)")
    p.add_argument("--contrastive_batch_size", type=int, default=16,
                   help="Number of anchor prompts per contrastive update step")

    # ── GRPO loss ─────────────────────────────────────────────────────────
    p.add_argument("--clip_range", type=float, default=1e-5,
                   help="PPO clip range ε — flow_grpo fast uses 1e-5")
    p.add_argument("--adv_clip_max", type=float, default=5.0,
                   help="Clip advantages to [-adv_clip_max, adv_clip_max]")
    p.add_argument("--global_std", action="store_true", default=True,
                   help="Use batch-level std for advantage normalisation")

    # ── Logging ───────────────────────────────────────────────────────────
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--vis_every", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="z-image-counting-grpo")
    p.add_argument("--wandb_run", type=str, default="")

    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
