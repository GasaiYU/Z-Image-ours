"""
Token-Aware Layer Router Training Script
=========================================
Trains a lightweight DynamicTokenRouter on top of a frozen LLM (Qwen) using
Token-level Triplet Contrastive Loss.

Goal: Force the router to route attribute tokens (colors, numbers, textures...)
      to shallower LLM layers, while keeping noun tokens in deep layers.

Usage:
    python train_router.py \
        --model_dir /path/to/z-image/model \
        --triplet_dir data/train_triplets \
        --output_dir checkpoints/router \
        --tasks color counting texture shape spatial \
        --epochs 10 --lr 3e-4 --batch_size 32 --margin 0.3
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Use the same loading path as test_token_routing.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils import ensure_model_weights, load_from_local_dir
import random


# =============================================================================
# 1. DynamicTokenRouter 网络结构
# =============================================================================
class DynamicTokenRouter(nn.Module):
    """
    A lightweight MLP that takes the deep LLM features of a token as input,
    and outputs per-layer routing weights (Softmax over all LLM layers).
    
    The final fused embedding is a weighted sum of all LLM layers' features.
    Only this MLP is trained; the LLM itself is frozen.
    
    Params: ~1M (hidden_size=3584, mid_dim=256, num_layers=32)
    """
    def __init__(self, hidden_size: int, num_layers: int, mid_dim: int = 256):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.router_mlp = nn.Sequential(
            nn.Linear(hidden_size, mid_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(mid_dim, num_layers),
        )

        # Smart initialization: default to deep layer (last layer = index -1)
        # This ensures at step 0, the router behaves identical to baseline Z-Image.
        nn.init.zeros_(self.router_mlp[-1].weight)
        nn.init.zeros_(self.router_mlp[-1].bias)
        self.router_mlp[-1].bias.data[-1] = 5.0   # strong prior towards deepest layer

    def forward(self, all_hidden_states: tuple, attention_mask: torch.Tensor = None):
        """
        Args:
            all_hidden_states: tuple of (num_layers+1) tensors, each [B, S, D].
                               Index 0 is the raw embedding layer; 1..num_layers are transformer layers.
            attention_mask: [B, S] bool mask (True = valid token).
        Returns:
            fused_embeds: [B, S, D]   weighted sum across layers
            routing_weights: [B, S, num_layers]   softmax weights (for visualization / aux loss)
        """
        # Stack transformer output layers (skip embedding layer at index 0)
        # stacked: [B, S, num_layers, D]
        layers = all_hidden_states[1 : self.num_layers + 1]
        stacked = torch.stack(layers, dim=2)                         # [B, S, L, D]

        # Use deep features as the routing decision signal (detach: no gradient into LLM)
        decision_feat = all_hidden_states[-2].detach()               # [B, S, D]

        # Compute per-token routing logits → weights
        routing_logits   = self.router_mlp(decision_feat)            # [B, S, L]
        routing_weights  = F.softmax(routing_logits, dim=-1)         # [B, S, L]

        # Weighted sum: fused_embeds[b, s, d] = sum_l w[b,s,l] * stacked[b,s,l,d]
        fused_embeds = (stacked * routing_weights.unsqueeze(-1)).sum(dim=2)  # [B, S, D]

        # Zero out pad tokens
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(fused_embeds.dtype)
            fused_embeds = fused_embeds * mask

        return fused_embeds, routing_weights


# =============================================================================
# 2. Dataset: 读取三元组 JSONL 并 Tokenize
# =============================================================================
class TripletDataset(Dataset):
    """
    Reads (anchor, positive, negative) triplets from JSONL files.
    Returns tokenized tensors and the index of the target_word in each sequence.
    """

    def __init__(
        self,
        triplet_files: list,
        tokenizer,
        max_length: int = 128,
        use_chat_template: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_chat_template = use_chat_template
        self.data = []

        for fpath in triplet_files:
            if not os.path.exists(fpath):
                print(f"[Dataset] Warning: file not found {fpath}")
                continue
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if all(k in item for k in ('anchor', 'positive', 'negative', 'target_word')):
                            self.data.append(item)
                    except json.JSONDecodeError:
                        continue

        print(f"[Dataset] Loaded {len(self.data)} triplets from {len(triplet_files)} files.")

    def _format(self, text: str) -> str:
        """Apply chat template consistent with test_token_routing.py inference."""
        if self.use_chat_template:
            messages = [{"role": "user", "content": text}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,   # matches test_token_routing.py exactly
            )
        return text

    def _tokenize(self, text: str):
        """Return input_ids and attention_mask tensors."""
        enc = self.tokenizer(
            self._format(text),
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt',
        )
        return enc.input_ids.squeeze(0), enc.attention_mask.squeeze(0)

    def _find_target_idx(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, target_word: str) -> int:
        """
        Find the first token index that (approximately) encodes `target_word`.
        Returns -1 if not found (this sample will be skipped in the loss).
        """
        valid_ids = input_ids[attention_mask.bool()].tolist()
        for idx, tid in enumerate(valid_ids):
            token_str = self.tokenizer.decode([tid], skip_special_tokens=True).lower().strip()
            # Allow partial match: 'red' may be tokenized as 'Ġred' etc.
            if target_word.lower() in token_str or token_str in target_word.lower():
                return idx
        return -1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        target_word = item['target_word']

        a_ids, a_mask = self._tokenize(item['anchor'])
        p_ids, p_mask = self._tokenize(item['positive'])
        n_ids, n_mask = self._tokenize(item['negative'])

        # Target token index in each sequence
        a_tidx = self._find_target_idx(a_ids, a_mask, target_word)
        p_tidx = self._find_target_idx(p_ids, p_mask, target_word)
        n_tidx = self._find_target_idx(n_ids, n_mask, target_word)

        return {
            'a_ids':   a_ids,   'a_mask':  a_mask,   'a_tidx':  a_tidx,
            'p_ids':   p_ids,   'p_mask':  p_mask,   'p_tidx':  p_tidx,
            'n_ids':   n_ids,   'n_mask':  n_mask,   'n_tidx':  n_tidx,
            'target_word': target_word,
            'task': item.get('task', 'unknown'),
        }


def collate_fn(batch):
    """Custom collate: stack tensors, keep Python scalars as lists."""
    return {
        'a_ids':  torch.stack([b['a_ids']  for b in batch]),
        'a_mask': torch.stack([b['a_mask'] for b in batch]),
        'a_tidx': [b['a_tidx'] for b in batch],
        'p_ids':  torch.stack([b['p_ids']  for b in batch]),
        'p_mask': torch.stack([b['p_mask'] for b in batch]),
        'p_tidx': [b['p_tidx'] for b in batch],
        'n_ids':  torch.stack([b['n_ids']  for b in batch]),
        'n_mask': torch.stack([b['n_mask'] for b in batch]),
        'n_tidx': [b['n_tidx'] for b in batch],
        'target_word': [b['target_word'] for b in batch],
        'task':   [b['task'] for b in batch],
    }


class HalfTaskBatchSampler(Sampler):
    """
    每个 batch 中：
      - 前 50%（half）来自随机选定的同一个 task（轮换采样）
      - 后 50% 来自其他所有 task 的混合

    这样保证 counting batch 里 two/three/four 必然同框，互为有效负样本，
    同时不损失跨 task 的多样性。
    """
    def __init__(self, dataset: TripletDataset, batch_size: int, shuffle: bool = True):
        self.batch_size = batch_size
        self.half = batch_size // 2
        self.shuffle = shuffle

        # 按 task 分组索引
        task_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, item in enumerate(dataset.data):
            task_to_indices[item['task']].append(idx)

        self.tasks = list(task_to_indices.keys())
        self.task_indices = task_to_indices   # {"counting": [...], "color": [...], ...}

        # 所有索引的扁平列表（用于采 other 50%）
        self.all_indices = list(range(len(dataset)))

        # 每个 epoch 需要多少个 batch（以数据集总量 / batch_size 为准）
        self.n_batches = len(self.all_indices) // batch_size

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        # 每个 epoch 开始时打乱各组
        task_pools: dict[str, list[int]] = {}
        for task, indices in self.task_indices.items():
            pool = indices.copy()
            if self.shuffle:
                random.shuffle(pool)
            task_pools[task] = pool

        other_pool = self.all_indices.copy()
        if self.shuffle:
            random.shuffle(other_pool)

        # 轮换 task（每个 batch 换一个 task 作为"主角"）
        task_cycle = self.tasks.copy()
        if self.shuffle:
            random.shuffle(task_cycle)

        other_cursor = 0
        for batch_idx in range(self.n_batches):
            focal_task = task_cycle[batch_idx % len(task_cycle)]
            focal_pool = task_pools[focal_task]

            # 从 focal_task 采 half 个（循环复用，数量不足时重新打乱）
            focal_samples = []
            while len(focal_samples) < self.half:
                if len(focal_pool) == 0:
                    focal_pool = self.task_indices[focal_task].copy()
                    if self.shuffle:
                        random.shuffle(focal_pool)
                    task_pools[focal_task] = focal_pool
                focal_samples.append(focal_pool.pop())

            # 从其余数据采 half 个
            other_samples = []
            while len(other_samples) < self.half:
                if other_cursor >= len(other_pool):
                    other_pool = self.all_indices.copy()
                    if self.shuffle:
                        random.shuffle(other_pool)
                    other_cursor = 0
                other_samples.append(other_pool[other_cursor])
                other_cursor += 1

            batch = focal_samples + other_samples
            if self.shuffle:
                random.shuffle(batch)
            yield batch


# =============================================================================
# 3. Loss Functions
# =============================================================================
def triplet_margin_loss(ea: torch.Tensor, ep: torch.Tensor, en: torch.Tensor, margin: float = 0.3):
    """
    Triplet Margin Loss (cosine similarity version).
    Pushes sim(anchor, positive) > sim(anchor, negative) + margin.

    Args:
        ea, ep, en: [N, D] L2-normalized feature vectors
    """
    sim_pos = F.cosine_similarity(ea, ep, dim=-1)   # [N]
    sim_neg = F.cosine_similarity(ea, en, dim=-1)   # [N]
    loss = F.relu(sim_neg - sim_pos + margin)        # [N]
    return loss.mean()


def supcon_loss(
    ea: torch.Tensor,
    ep: torch.Tensor,
    target_words: list,
    temperature: float = 0.07,
):
    """
    Supervised Contrastive Loss (SupCon, NeurIPS 2020).

    Fixes the "false negative" problem of vanilla InfoNCE:
    - Vanilla InfoNCE treats ALL other in-batch samples as negatives.
    - But if two samples share the same target_word (e.g. both are "red"),
      their features SHOULD be similar — pushing them apart is wrong.

    SupCon fix:
    - Positives: same target_word  (both diagonal AND same-class off-diagonal)
    - Negatives: different target_word ONLY

    For each anchor ea[i]:
        L_i = -1/|P(i)| * sum_{p in P(i)} log [
            exp(sim(ea_i, ep_p) / tau) /
            sum_{j: target_words[j] != target_words[i]} exp(sim(ea_i, ep_j) / tau)
        ]

    Args:
        ea:           [N, D]  L2-normalized anchor features
        ep:           [N, D]  L2-normalized positive features
        target_words: list of N strings, e.g. ["red", "two", "red", "wooden"]
        temperature:  scalar, typical 0.07~0.2
    """
    N = ea.shape[0]
    device = ea.device

    # Full similarity matrix [N, N]
    sim_matrix = torch.matmul(ea, ep.T) / temperature   # [N, N]

    # Build same-class mask: same_mask[i, j] = True if target_words[i] == target_words[j]
    same_mask = torch.zeros(N, N, dtype=torch.bool, device=device)
    for i in range(N):
        for j in range(N):
            if target_words[i] == target_words[j]:
                same_mask[i, j] = True

    # Positive mask: same class AND not self  →  P(i)
    pos_mask = same_mask.clone()
    pos_mask.fill_diagonal_(False)                       # remove self-pair

    # Negative mask: different class  →  used in denominator
    # (self-pair is also excluded from denominator by convention)
    neg_mask = ~same_mask                                # [N, N]
    neg_mask.fill_diagonal_(False)

    # For samples that have NO in-batch positive (unique target_word in this batch),
    # fall back to the diagonal-only InfoNCE (standard behaviour).
    has_pos = pos_mask.any(dim=1)                        # [N] bool

    loss_total = ea.new_tensor(0.0)
    n_valid = 0

    for i in range(N):
        # Denominator: diagonal (self-positive) + all negatives
        denom_mask = neg_mask[i].clone()
        denom_mask[i] = True                             # always include self positive ep[i]
        log_denom = torch.logsumexp(sim_matrix[i][denom_mask], dim=0)

        if has_pos[i]:
            # Multiple positives: average log-likelihood over all same-class ep[j]
            pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
            log_probs = sim_matrix[i][pos_indices] - log_denom   # [|P(i)|]
            loss_i = -log_probs.mean()
        else:
            # No same-class partner in this batch: treat diagonal as the only positive
            loss_i = -(sim_matrix[i, i] - log_denom)

        loss_total = loss_total + loss_i
        n_valid += 1

    return loss_total / max(n_valid, 1)


def noun_regularization_loss(
    fused_embeds: torch.Tensor,
    deep_embeds: torch.Tensor,
    noun_mask: torch.Tensor,
):
    """
    Force noun token features to stay close to the original deep features.
    Prevents the router from also routing noun tokens to shallow layers.

    Args:
        fused_embeds: [B, S, D]
        deep_embeds:  [B, S, D]
        noun_mask:    [B, S] bool, True for noun positions
    """
    if noun_mask.sum() == 0:
        return fused_embeds.new_tensor(0.0)
    diff = (fused_embeds - deep_embeds)[noun_mask]   # [N_nouns, D]
    return (diff ** 2).mean()


# =============================================================================
# 4. Helper: extract hidden states from frozen LLM
# =============================================================================
@torch.no_grad()
def encode_batch(text_encoder, input_ids, attention_mask):
    """Run frozen LLM and return all hidden states (tuple of tensors)."""
    outputs = text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    return outputs.hidden_states   # tuple: (embed_layer, layer1, ..., layerN)


def extract_token_features(fused_embeds: torch.Tensor, token_indices: list) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract feature vectors at specified token positions.
    Skips samples where index is -1 (target word not found).

    Returns:
        features: [N_valid, D]  L2-normalized
        valid_mask: [B] bool
    """
    features = []
    valid_mask = []
    B = fused_embeds.shape[0]
    for i in range(B):
        tidx = token_indices[i]
        if tidx < 0 or tidx >= fused_embeds.shape[1]:
            valid_mask.append(False)
            features.append(fused_embeds.new_zeros(fused_embeds.shape[-1]))
        else:
            feat = fused_embeds[i, tidx, :]
            features.append(F.normalize(feat, dim=-1))
            valid_mask.append(True)
    features = torch.stack(features, dim=0)               # [B, D]
    valid_mask = torch.tensor(valid_mask, device=fused_embeds.device)
    return features, valid_mask


# =============================================================================
# 5. Training Loop
# =============================================================================
def train(args):
    device = torch.device(args.device)

    # ---- Load text_encoder + tokenizer (same as test_token_routing.py) ----
    # load_from_local_dir also loads transformer/vae/scheduler which we don't need,
    # but it correctly handles tokenizer path fallback (tokenizer/ or text_encoder/).
    print("[Init] Loading text encoder and tokenizer via load_from_local_dir...")
    components = load_from_local_dir(
        args.model_dir,
        device=str(device),
        dtype=torch.bfloat16,
        verbose=True,
    )
    text_encoder = components["text_encoder"]
    tokenizer    = components["tokenizer"]

    # Free the components we don't need to save GPU memory
    del components["transformer"], components["vae"], components["scheduler"]
    import gc; gc.collect()
    torch.cuda.empty_cache()

    # Freeze all LLM parameters
    for p in text_encoder.parameters():
        p.requires_grad_(False)
    text_encoder.eval()

    # Probe hidden_size and num_layers with a dummy forward pass
    with torch.no_grad():
        dummy_ids = torch.zeros(1, 4, dtype=torch.long, device=device)
        dummy_mask = torch.ones(1, 4, dtype=torch.long, device=device)
        dummy_out = text_encoder(input_ids=dummy_ids, attention_mask=dummy_mask, output_hidden_states=True)
    hidden_size = dummy_out.hidden_states[0].shape[-1]
    num_layers = len(dummy_out.hidden_states) - 1   # subtract embedding layer
    print(f"[Init] LLM: hidden_size={hidden_size}, num_transformer_layers={num_layers}")
    del dummy_out

    # ---- Build router ----
    router = DynamicTokenRouter(hidden_size=hidden_size, num_layers=num_layers, mid_dim=args.mid_dim).to(device)
    n_params = sum(p.numel() for p in router.parameters())
    print(f"[Init] Router params: {n_params:,}  (~{n_params/1e6:.2f}M)")

    # ---- Dataset ----
    triplet_files = [
        os.path.join(args.triplet_dir, f"{task}_triplets.jsonl")
        for task in args.tasks
    ]
    dataset = TripletDataset(
        triplet_files=triplet_files,
        tokenizer=tokenizer,
        max_length=args.max_length,
        use_chat_template=args.use_chat_template,
    )
    batch_sampler = HalfTaskBatchSampler(dataset, batch_size=args.batch_size, shuffle=True)
    loader = DataLoader(
        dataset, batch_sampler=batch_sampler,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )

    # ---- Optimizer & Scheduler ----
    optimizer = AdamW(router.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = len(loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr * 0.01)

    # ---- Output directory ----
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Training ----
    global_step = 0
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        router.train()
        epoch_loss_contrastive = 0.0
        epoch_loss_reg = 0.0
        n_batches = 0
        skipped = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            # Move ids/masks to device
            a_ids  = batch['a_ids'].to(device)
            a_mask = batch['a_mask'].to(device)
            p_ids  = batch['p_ids'].to(device)
            p_mask = batch['p_mask'].to(device)
            n_ids  = batch['n_ids'].to(device)
            n_mask = batch['n_mask'].to(device)

            # ---- Step 1: Run frozen LLM on all three ----
            hs_a = encode_batch(text_encoder, a_ids, a_mask)
            hs_p = encode_batch(text_encoder, p_ids, p_mask)
            hs_n = encode_batch(text_encoder, n_ids, n_mask)

            # Deep features for regularization (before routing, no grad)
            deep_a = hs_a[-2].detach()   # [B, S, D]

            # ---- Step 2: Route through DynamicTokenRouter ----
            # For contrastive loss: we need to enable gradients here
            # (router parameters receive gradient from this path)
            fused_a, rw_a = router(hs_a, attention_mask=a_mask)
            fused_p, _    = router(hs_p, attention_mask=p_mask)
            fused_n, _    = router(hs_n, attention_mask=n_mask)

            # ---- Step 3: Extract target token features ----
            ea, vm_a = extract_token_features(fused_a, batch['a_tidx'])
            ep, vm_p = extract_token_features(fused_p, batch['p_tidx'])
            en, vm_n = extract_token_features(fused_n, batch['n_tidx'])

            if args.loss_type == 'triplet':
                valid = vm_a & vm_p & vm_n
            else:
                # supcon only needs anchor + positive
                valid = vm_a & vm_p

            if valid.sum() < 2:
                skipped += a_ids.shape[0]
                continue

            valid_idx = valid.nonzero(as_tuple=True)[0].tolist()
            ea = ea[valid]
            ep = ep[valid]
            en = en[valid] if args.loss_type == 'triplet' else None
            batch_target_words = [batch['target_word'][i] for i in valid_idx]

            # ---- Step 4a: Contrastive Loss ----
            if args.loss_type == 'triplet':
                loss_contrastive = triplet_margin_loss(ea, ep, en, margin=args.margin)
            else:
                # SupCon: same target_word pairs become extra positives, not false negatives
                loss_contrastive = supcon_loss(ea, ep, batch_target_words, temperature=args.temperature)

            # ---- Step 4b: Noun Regularization Loss (optional) ----
            # Mark all non-target positions as "noun-like" (conservative: just use valid token positions)
            loss_reg = fused_a.new_tensor(0.0)
            if args.lambda_reg > 0:
                # Build noun_mask: valid tokens that are NOT the target token
                B, S, D = fused_a.shape
                noun_mask = a_mask.bool().clone()   # [B, S]
                # Remove target token positions from noun_mask
                for i, tidx in enumerate(batch['a_tidx']):
                    if 0 <= tidx < S:
                        noun_mask[i, tidx] = False
                loss_reg = noun_regularization_loss(fused_a, deep_a, noun_mask)

            loss = loss_contrastive + args.lambda_reg * loss_reg

            # ---- Step 5: Backward ----
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss_contrastive += loss_contrastive.item()
            epoch_loss_reg         += loss_reg.item() if args.lambda_reg > 0 else 0.0
            n_batches += 1
            global_step += 1

            pbar.set_postfix({
                'L_contra': f'{loss_contrastive.item():.4f}',
                'L_reg':    f'{loss_reg.item():.4f}' if args.lambda_reg > 0 else '—',
                'lr':       f'{scheduler.get_last_lr()[0]:.2e}',
                'skipped':  skipped,
            })

            # Save checkpoint every N steps
            if global_step % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, f"router_step{global_step}.pt")
                torch.save({
                    'step': global_step,
                    'router_state_dict': router.state_dict(),
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'mid_dim': args.mid_dim,
                }, ckpt_path)
                print(f"\n[Checkpoint] Saved -> {ckpt_path}")

        # ---- End of epoch stats ----
        avg_contra = epoch_loss_contrastive / max(n_batches, 1)
        avg_reg    = epoch_loss_reg          / max(n_batches, 1)
        print(f"\n[Epoch {epoch}] avg_contrastive={avg_contra:.4f}  avg_reg={avg_reg:.4f}  skipped={skipped}")

        # Save best checkpoint
        if avg_contra < best_loss:
            best_loss = avg_contra
            best_path = os.path.join(args.output_dir, "router_best.pt")
            torch.save({
                'epoch': epoch,
                'step': global_step,
                'router_state_dict': router.state_dict(),
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'mid_dim': args.mid_dim,
                'best_loss': best_loss,
            }, best_path)
            print(f"[Best] New best loss={best_loss:.4f} -> {best_path}")

    # ---- Save final checkpoint ----
    final_path = os.path.join(args.output_dir, "router_final.pt")
    torch.save({
        'epoch': args.epochs,
        'step': global_step,
        'router_state_dict': router.state_dict(),
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'mid_dim': args.mid_dim,
    }, final_path)
    print(f"\n[Done] Final checkpoint saved -> {final_path}")


# =============================================================================
# 6. Argument Parsing
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train DynamicTokenRouter with Triplet Contrastive Loss")

    # Paths
    parser.add_argument('--model_dir',   type=str, default='ckpts/Z-Image-Turbo',
                        help='Path to Z-Image model directory, same as used in test_token_routing.py '
                             '(e.g. ckpts/Z-Image-Turbo). Loaded via load_from_local_dir.')
    parser.add_argument('--triplet_dir', type=str, default='data/train_triplets',
                        help='Directory containing *_triplets.jsonl files')
    parser.add_argument('--output_dir',  type=str, default='checkpoints/router',
                        help='Directory to save router checkpoints')

    # Task selection
    parser.add_argument('--tasks', nargs='+',
                        default=['color', 'counting', 'texture', 'shape', 'spatial'],
                        help='Which task triplets to use (matches *_triplets.jsonl filenames)')

    # Model
    parser.add_argument('--mid_dim', type=int, default=256,
                        help='Hidden dimension of Router MLP')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Max tokenizer sequence length')
    parser.add_argument('--use_chat_template', action='store_true',
                        help='Apply Qwen chat template during tokenization (consistent with inference)')

    # Loss
    parser.add_argument('--loss_type',   type=str,   default='supcon',
                        choices=['supcon', 'triplet'],
                        help='supcon: supervised contrastive, fixes false negatives (recommended); '
                             'triplet: explicit hard negative, simpler but weaker')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for InfoNCE loss (smaller = harder, typical: 0.07~0.2)')
    parser.add_argument('--margin',     type=float, default=0.3,
                        help='Margin for triplet loss (higher = harder constraint)')
    parser.add_argument('--lambda_reg', type=float, default=0.1,
                        help='Weight of noun regularization loss (0 to disable)')

    # Training
    parser.add_argument('--epochs',      type=int,   default=10)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=3e-4)
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--save_every',  type=int,   default=500,
                        help='Save checkpoint every N global steps')
    parser.add_argument('--device',      type=str,   default='cuda')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print("=" * 60)
    print("  DynamicTokenRouter Contrastive Training")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k:<20} = {v}")
    print("=" * 60)
    train(args)
