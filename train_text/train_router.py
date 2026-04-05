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
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# Use the same loading path as test_token_routing.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from utils import ensure_model_weights, load_from_local_dir
import random


# =============================================================================
# 0. Distributed helpers
# =============================================================================
def init_distributed():
    """Initialize DDP if launched via torchrun, otherwise return single-GPU defaults."""
    if 'RANK' not in os.environ:
        return 0, 1, 0   # rank, world_size, local_rank
    dist.init_process_group(backend='nccl')
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def all_gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-gather a tensor across all GPUs while keeping gradients on the local slice.

    Standard DDP contrastive trick (used in CLIP, MoCo v3, etc.):
      - torch.distributed.all_gather() has no grad → we restore the local slice
        with the original tensor that still has its computation graph attached.
      - Result: gradients flow only through the local portion, but the similarity
        matrix is built from the globally gathered features.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    world_size = dist.get_world_size()
    rank       = dist.get_rank()
    gathered   = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous())
    gathered[rank] = tensor          # restore grad-carrying local slice
    return torch.cat(gathered, dim=0)


def all_gather_strings(str_list: list) -> list:
    """All-gather a list of strings across GPUs (no grad needed)."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return str_list
    world_size = dist.get_world_size()
    gathered   = [None] * world_size
    dist.all_gather_object(gathered, str_list)
    return [s for sublist in gathered for s in sublist]


# =============================================================================
# 1. DynamicTokenRouter 网络结构
# =============================================================================
class DynamicTokenRouter(nn.Module):
    """
    Lightweight MLP: takes h_1 (shallowest transformer layer) as input and
    outputs routing weights over a restricted mid-layer range [route_start, route_end).

    Design rationale
    ----------------
    Input — h_1 only:
      h_1 carries the strongest *lexical identity* signal.  For counting words
      (two/three/four…), h_1 similarities are ~0.84, far more discriminative
      than deeper layers (0.93~0.99).

    Output — layers [route_start, route_end):
      Restricting to mid-layers (default 10–20) avoids two failure modes:
        • Layer 1 collapse: layer 1 is most discriminative but too raw for DiT
          (DiT was trained on hidden_states[-2], i.e. a deep layer).
        • Deep-layer saturation: layers 25+ are nearly identical for counting
          words (cos_sim ~0.99) and carry no discriminative signal.
      Layers 10–20 have good discriminability (~0.90–0.95) while remaining
      semantically rich enough for the DiT to interpret correctly.
      This physical constraint replaces entropy regularisation entirely.

    Initialisation — deep-biased within the range:
      bias[-1] = +5 → softmax ≈ 1 on hidden_states[route_end-1] (deepest in
      range). At t=0, fused ≈ hidden_states[route_end-1] for every token.
      SupCon shifts routing only for target-token positions; all other tokens
      receive no gradient and stay at the deep-within-range initialisation.
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
        self.num_layers  = num_layers
        self.hidden_size = hidden_size

        # Clamp range to valid transformer layer indices [1, num_layers-1]
        self.route_start = max(1, min(route_start, num_layers - 2))
        self.route_end   = max(self.route_start + 1, min(route_end, num_layers))
        self.n_route     = self.route_end - self.route_start  # e.g. 11 for [10,21)

        in_dim = hidden_size
        self.router_mlp = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.LayerNorm(mid_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(mid_dim, mid_dim // 2),
            nn.SiLU(),
            nn.Linear(mid_dim // 2, self.n_route),
        )

        # Deep-biased init within the routing range:
        #   bias[-1] = +5  →  weight ≈ 1 on hidden_states[route_end-1]
        nn.init.zeros_(self.router_mlp[-1].bias)
        self.router_mlp[-1].bias.data[-1] = 5.0
        nn.init.normal_(self.router_mlp[-1].weight, std=0.01)

    def forward(self, all_hidden_states: tuple, attention_mask: torch.Tensor = None):
        """
        Args:
            all_hidden_states : tuple of (num_layers+1) tensors, each [B, S, D].
                                [0]=embedding, [1..num_layers]=transformer layers.
            attention_mask    : [B, S] bool/int mask (True = valid token).
        Returns:
            fused_embeds    : [B, S, D]           weighted sum over routing layers
            routing_weights : [B, S, n_route]     softmax weights
            h1              : [B, S, D]            raw h_1 features
        """
        h1 = all_hidden_states[1].float()                    # [B, S, D]
        decision_feat   = F.normalize(h1.detach(), dim=-1)  # [B, S, D]

        routing_logits  = self.router_mlp(decision_feat)    # [B, S, n_route]
        routing_weights = F.softmax(routing_logits, dim=-1) # [B, S, n_route]

        # Weighted sum over hidden_states[route_start : route_end]
        route_layers = all_hidden_states[self.route_start : self.route_end]
        stacked = torch.stack([l.float() for l in route_layers], dim=2)  # [B,S,n_route,D]
        rw = routing_weights.to(stacked.dtype).unsqueeze(-1)              # [B,S,n_route,1]
        fused_embeds = (stacked * rw).sum(dim=2)                          # [B, S, D]

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(fused_embeds.dtype)
            fused_embeds = fused_embeds * mask

        return fused_embeds, routing_weights, h1


# =============================================================================
# 1b. RouterTextEncoder — E2E wrapper for inference
# =============================================================================
class RouterTextEncoder(nn.Module):
    """
    Wraps a frozen LLM text encoder + a trained DynamicTokenRouter so that
    the pipeline can call text_encoder(input_ids, ...) and transparently
    receive router-fused embeddings in hidden_states[-2].

    Usage:
        router = DynamicTokenRouter(...)
        router.load_state_dict(ckpt["router_state_dict"])
        components["text_encoder"] = RouterTextEncoder(
            components["text_encoder"], router
        )
        # Now just call generate() normally — no monkey-patching needed.
    """
    def __init__(self, base_encoder: nn.Module, router: "DynamicTokenRouter"):
        super().__init__()
        self.base   = base_encoder
        self.router = router

    def forward(self, input_ids, attention_mask=None, output_hidden_states=True, **kwargs):
        outputs = self.base(
            input_ids, attention_mask=attention_mask,
            output_hidden_states=True, **kwargs
        )
        with torch.no_grad() if not self.training else torch.enable_grad():
            fused, _, _ = self.router(outputs.hidden_states, attention_mask)
        new_hs = list(outputs.hidden_states)
        new_hs[-2] = fused
        outputs.hidden_states = tuple(new_hs)
        return outputs


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

    DDP 用法：每个 rank 传入不同的 seed（如 base_seed + rank），
    确保各 GPU 处理不同的 batch（之后 all_gather 汇总做 contrastive loss）。
    """
    def __init__(self, dataset: TripletDataset, batch_size: int, shuffle: bool = True, seed: int = 42):
        self.batch_size = batch_size
        self.half = batch_size // 2
        self.shuffle = shuffle
        self.seed = seed

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
        # 每个 epoch 用固定 seed 初始化 RNG，不同 rank 因 seed 不同而产生不同 batch
        rng = random.Random(self.seed)
        self.seed += 1   # 每次迭代递增，确保不同 epoch 也不同

        task_pools: dict[str, list[int]] = {}
        for task, indices in self.task_indices.items():
            pool = indices.copy()
            if self.shuffle:
                rng.shuffle(pool)
            task_pools[task] = pool

        other_pool = self.all_indices.copy()
        if self.shuffle:
            rng.shuffle(other_pool)

        # 轮换 task（每个 batch 换一个 task 作为"主角"）
        task_cycle = self.tasks.copy()
        if self.shuffle:
            rng.shuffle(task_cycle)

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
                        rng.shuffle(focal_pool)
                    task_pools[focal_task] = focal_pool
                focal_samples.append(focal_pool.pop())

            # 从其余数据采 half 个
            other_samples = []
            while len(other_samples) < self.half:
                if other_cursor >= len(other_pool):
                    other_pool = self.all_indices.copy()
                    if self.shuffle:
                        rng.shuffle(other_pool)
                    other_cursor = 0
                other_samples.append(other_pool[other_cursor])
                other_cursor += 1

            batch = focal_samples + other_samples
            if self.shuffle:
                rng.shuffle(batch)
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

    # Build same-class mask vectorized: hash each string → int64 tensor → broadcast compare
    # hash() is deterministic within one process; collisions are astronomically rare for short words
    word_ids = torch.tensor([hash(w) for w in target_words], dtype=torch.int64, device=device)
    same_mask = (word_ids.unsqueeze(0) == word_ids.unsqueeze(1))   # [N, N]  O(N) not O(N²)

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


def routing_entropy_loss(routing_weights: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Entropy regularisation on the routing weight distribution.

    Maximising entropy encourages the router to spread weight across layers
    rather than collapsing to a single layer (the deep-layer prior from init).
    This is especially important for tokens whose deep features are similar
    (e.g. "two" vs "three") — it keeps the routing distribution explorable
    so that the contrastive loss can pull them apart.

    Loss = -mean_entropy = mean[ sum_l w_l * log(w_l) ]
    Minimising this loss maximises entropy (more uniform → lower loss_entropy).

    Args:
        routing_weights: [B, S, L]  softmax weights (already normalised).
        attention_mask:  [B, S]     bool / int mask; only valid tokens count.
    Returns:
        scalar entropy loss (>= 0).
    """
    # entropy per token: -sum_l w * log(w),  shape [B, S]
    entropy = -(routing_weights.float() * (routing_weights.float() + 1e-8).log()).sum(dim=-1)

    if attention_mask is not None:
        mask    = attention_mask.bool().float()          # [B, S]
        entropy = (entropy * mask).sum() / mask.sum().clamp(min=1)
    else:
        entropy = entropy.mean()

    # We want to maximise entropy → minimise negative entropy
    return -entropy


def layer_discriminability_loss(
    routing_weights: torch.Tensor,
    all_hs_a: tuple,
    all_hs_p: tuple,
    token_indices_a: list,
    token_indices_p: list,
    valid_mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Layer Discriminability Loss (L_disc) for all-layer routing.

    For each valid (anchor, positive) token pair, compute how discriminative
    each routing layer l ∈ hidden_states[1 : num_layers] is:

        disc_l = cos_sim( h_l^anchor[target], h_l^positive[target] )

    Lower disc_l  →  same word has more diverse context-dependent representations
                  →  layer l carries more fine-grained information
                  →  routing should give it higher weight.

    Target distribution: softmax(-disc / tau)  (lower sim = higher weight).
    Loss: KL(target || routing_weights) — direct gradient to routing_weights.
    """
    S       = routing_weights.shape[1]
    n_route = routing_weights.shape[2]
    # routing covers all_hs[1 : n_route+1] = all_hs[1:-1]
    route_layers_a = all_hs_a[1 : n_route + 1]
    route_layers_p = all_hs_p[1 : n_route + 1]

    valid_idx = valid_mask.nonzero(as_tuple=True)[0]
    if valid_idx.numel() == 0:
        return routing_weights.new_tensor(0.0)

    vi_list, at_list, pt_list = [], [], []
    for i in valid_idx.tolist():
        a_t, p_t = token_indices_a[i], token_indices_p[i]
        if 0 <= a_t < S and 0 <= p_t < S:
            vi_list.append(i); at_list.append(a_t); pt_list.append(p_t)

    if not vi_list:
        return routing_weights.new_tensor(0.0)

    # Vectorised: build [N_valid, n_route, D] tensors
    ha_layers = torch.stack(
        [torch.stack([route_layers_a[l][i, at].float()
                      for l in range(n_route)])
         for i, at in zip(vi_list, at_list)]
    )  # [N, n_route, D]
    hp_layers = torch.stack(
        [torch.stack([route_layers_p[l][i, pt].float()
                      for l in range(n_route)])
         for i, pt in zip(vi_list, pt_list)]
    )  # [N, n_route, D]

    ha_n = F.normalize(ha_layers, dim=-1)
    hp_n = F.normalize(hp_layers, dim=-1)
    disc_scores = (ha_n * hp_n).sum(-1)               # [N, n_route]

    target  = F.softmax(-disc_scores / temperature, dim=-1).detach()
    rw_tok  = torch.stack([routing_weights[i, at] for i, at in zip(vi_list, at_list)])

    return F.kl_div((rw_tok + 1e-8).log(), target, reduction='batchmean')


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
    # ---- Distributed init ----
    rank, world_size, local_rank = init_distributed()
    is_main = (rank == 0)
    device = torch.device(f'cuda:{local_rank}')

    # ---- WandB (rank 0 only) ----
    use_wandb = args.use_wandb and is_main and _WANDB_AVAILABLE
    if args.use_wandb and not _WANDB_AVAILABLE:
        print("[WandB] wandb not installed, skipping. Run: pip install wandb")
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run or None,
            config=vars(args),
        )
        print(f"[WandB] Run: {wandb.run.url}")

    # ---- Load text_encoder + tokenizer (same as test_token_routing.py) ----
    if is_main:
        print("[Init] Loading text encoder and tokenizer via load_from_local_dir...")
    components = load_from_local_dir(
        args.model_dir,
        device=str(device),
        dtype=torch.bfloat16,
        verbose=is_main,
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
        dummy_ids  = torch.zeros(1, 4, dtype=torch.long, device=device)
        dummy_mask = torch.ones(1,  4, dtype=torch.long, device=device)
        dummy_out  = text_encoder(input_ids=dummy_ids, attention_mask=dummy_mask, output_hidden_states=True)
    hidden_size = dummy_out.hidden_states[0].shape[-1]
    num_layers  = len(dummy_out.hidden_states) - 1   # subtract embedding layer
    if is_main:
        print(f"[Init] LLM: hidden_size={hidden_size}, num_transformer_layers={num_layers}")
    del dummy_out

    # ---- Build router ----
    router = DynamicTokenRouter(
        hidden_size=hidden_size, num_layers=num_layers, mid_dim=args.mid_dim,
        route_start=args.route_start, route_end=args.route_end,
    ).to(device)
    if world_size > 1:
        router = DDP(router, device_ids=[local_rank])
    n_params = sum(p.numel() for p in router.parameters())
    if is_main:
        print(f"[Init] Router params: {n_params:,}  (~{n_params/1e6:.2f}M)")
        print(f"[Init] world_size={world_size}")

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
    # Each rank uses a different seed → different batches per GPU
    # After forward, features are all-gathered across ranks for a larger effective batch
    batch_sampler = HalfTaskBatchSampler(
        dataset, batch_size=args.batch_size, shuffle=True, seed=args.seed + rank
    )
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
        epoch_loss_disc        = 0.0
        epoch_loss_entropy     = 0.0
        n_batches = 0
        skipped   = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not is_main)
        for batch in pbar:
            a_ids  = batch['a_ids'].to(device)
            a_mask = batch['a_mask'].to(device)
            p_ids  = batch['p_ids'].to(device)
            p_mask = batch['p_mask'].to(device)
            n_ids  = batch['n_ids'].to(device)
            n_mask = batch['n_mask'].to(device)

            # ---- Step 1: Run frozen LLM ----
            hs_a = encode_batch(text_encoder, a_ids, a_mask)
            hs_p = encode_batch(text_encoder, p_ids, p_mask)
            hs_n = encode_batch(text_encoder, n_ids, n_mask)

            # ---- Step 2: Route through DynamicTokenRouter ----
            fused_a, rw_a, _ = router(hs_a, attention_mask=a_mask)
            fused_p, _,    _ = router(hs_p, attention_mask=p_mask)
            fused_n, _,    _ = router(hs_n, attention_mask=n_mask)

            # ---- Step 3: Extract target token features ----
            ea, vm_a = extract_token_features(fused_a, batch['a_tidx'])
            ep, vm_p = extract_token_features(fused_p, batch['p_tidx'])
            en, vm_n = extract_token_features(fused_n, batch['n_tidx'])

            if args.loss_type == 'triplet':
                valid = vm_a & vm_p & vm_n
            else:
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
            # Gradient ONLY flows through target token positions (ea, ep).
            # All other token positions receive zero gradient → their routing
            # weights stay at the deep-biased initialisation → fused ≈ deep.
            # No noun_reg needed; gradient sparsity handles it by design.
            if args.loss_type == 'triplet':
                loss_contrastive = triplet_margin_loss(ea, ep, en, margin=args.margin)
            else:
                ea_global    = all_gather_with_grad(ea)
                ep_global    = all_gather_with_grad(ep)
                words_global = all_gather_strings(batch_target_words)
                if args.temperature_warmup_steps > 0 and global_step < args.temperature_warmup_steps:
                    t = global_step / args.temperature_warmup_steps
                    current_temp = args.temperature_init + t * (args.temperature - args.temperature_init)
                else:
                    current_temp = args.temperature
                loss_contrastive = supcon_loss(ea_global, ep_global, words_global,
                                               temperature=current_temp)

            # ---- Step 4b: Layer Discriminability Loss ----
            loss_disc = fused_a.new_tensor(0.0)
            if args.lambda_disc > 0:
                disc_valid = vm_a & vm_p
                loss_disc = layer_discriminability_loss(
                    rw_a, hs_a, hs_p,
                    batch['a_tidx'], batch['p_tidx'],
                    disc_valid,
                    temperature=args.disc_temperature,
                )

            # ---- Step 4c: Entropy Regularisation (anti-collapse) ----
            # Only penalise target token positions — avoids fighting the
            # deep-bias on non-attribute tokens.
            loss_entropy = fused_a.new_tensor(0.0)
            if args.lambda_entropy > 0:
                vi_list, at_list = [], []
                for i, (vm, tidx) in enumerate(zip(vm_a.tolist(), batch['a_tidx'])):
                    if vm and 0 <= tidx < rw_a.shape[1]:
                        vi_list.append(i)
                        at_list.append(tidx)
                if vi_list:
                    rw_tok = torch.stack([rw_a[bi, ti] for bi, ti in zip(vi_list, at_list)])
                    loss_entropy = routing_entropy_loss(rw_tok.unsqueeze(1))

            loss = loss_contrastive + args.lambda_disc * loss_disc + args.lambda_entropy * loss_entropy

            # ---- Step 5: Backward ----
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss_contrastive += loss_contrastive.item()
            epoch_loss_disc        += loss_disc.item() if args.lambda_disc > 0 else 0.0
            epoch_loss_entropy     += loss_entropy.item() if args.lambda_entropy > 0 else 0.0
            n_batches  += 1
            global_step += 1

            if is_main:
                pbar.set_postfix({
                    'L_contra': f'{loss_contrastive.item():.4f}',
                    'L_disc':   f'{loss_disc.item():.4f}' if args.lambda_disc > 0 else '—',
                    'L_ent':    f'{loss_entropy.item():.4f}' if args.lambda_entropy > 0 else '—',
                    'lr':       f'{scheduler.get_last_lr()[0]:.2e}',
                    'skipped':  skipped,
                })
                if use_wandb:
                    wandb.log({
                        'train/loss_contrastive': loss_contrastive.item(),
                        'train/loss_disc':        loss_disc.item() if args.lambda_disc > 0 else 0.0,
                        'train/loss_entropy':     loss_entropy.item() if args.lambda_entropy > 0 else 0.0,
                        'train/loss_total':       loss.item(),
                        'train/lr':               scheduler.get_last_lr()[0],
                        'train/skipped':          skipped,
                    }, step=global_step)

            # Save checkpoint every N steps (rank 0 only)
            if is_main and global_step % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, f"router_step{global_step}.pt")
                raw_state = router.module.state_dict() if world_size > 1 else router.state_dict()
                torch.save({
                    'step': global_step,
                    'router_state_dict': raw_state,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'mid_dim': args.mid_dim,
                    'route_start': args.route_start,
                    'route_end': args.route_end,
                }, ckpt_path)
                print(f"\n[Checkpoint] Saved -> {ckpt_path}")

        # ---- End of epoch stats (rank 0 only) ----
        if is_main:
            avg_contra  = epoch_loss_contrastive / max(n_batches, 1)
            avg_disc    = epoch_loss_disc         / max(n_batches, 1)
            avg_entropy = epoch_loss_entropy      / max(n_batches, 1)
            print(f"\n[Epoch {epoch}] avg_contrastive={avg_contra:.4f}  "
                  f"avg_disc={avg_disc:.4f}  avg_entropy={avg_entropy:.4f}  skipped={skipped}")
            if use_wandb:
                wandb.log({
                    'epoch/loss_contrastive': avg_contra,
                    'epoch/loss_disc':        avg_disc,
                    'epoch/loss_entropy':     avg_entropy,
                    'epoch/epoch':            epoch,
                }, step=global_step)

            # Save best checkpoint
            if avg_contra < best_loss:
                best_loss = avg_contra
                best_path = os.path.join(args.output_dir, "router_best.pt")
                raw_state = router.module.state_dict() if world_size > 1 else router.state_dict()
                torch.save({
                    'epoch': epoch,
                    'step': global_step,
                    'router_state_dict': raw_state,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'mid_dim': args.mid_dim,
                    'route_start': args.route_start,
                    'route_end': args.route_end,
                    'best_loss': best_loss,
                }, best_path)
                print(f"[Best] New best loss={best_loss:.4f} -> {best_path}")

    # ---- Save final checkpoint (rank 0 only) ----
    if is_main:
        final_path = os.path.join(args.output_dir, "router_final.pt")
        raw_state  = router.module.state_dict() if world_size > 1 else router.state_dict()
        torch.save({
            'epoch': args.epochs,
            'step': global_step,
            'router_state_dict': raw_state,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'mid_dim': args.mid_dim,
            'route_start': args.route_start,
            'route_end': args.route_end,
        }, final_path)
        print(f"\n[Done] Final checkpoint saved -> {final_path}")
        if use_wandb:
            wandb.finish()


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
    parser.add_argument('--mid_dim', type=int, default=1024,
                        help='Hidden dimension of Router MLP (default raised to 1024 to handle '
                             '4×hidden_size=14336 input without over-compressing)')
    parser.add_argument('--route_start', type=int, default=10,
                        help='First transformer layer index included in routing range (inclusive)')
    parser.add_argument('--route_end', type=int, default=21,
                        help='Last transformer layer index included in routing range (exclusive)')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Max tokenizer sequence length')
    parser.add_argument('--use_chat_template', action='store_true',
                        help='Apply Qwen chat template during tokenization (consistent with inference)')

    # Loss
    parser.add_argument('--loss_type',   type=str,   default='supcon',
                        choices=['supcon', 'triplet'])
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Target SupCon temperature (final value after warmup)')
    parser.add_argument('--temperature_init', type=float, default=0.2,
                        help='Initial temperature for warmup')
    parser.add_argument('--temperature_warmup_steps', type=int, default=200,
                        help='Steps to anneal temperature_init → temperature (0 = off)')
    parser.add_argument('--margin',      type=float, default=0.3,
                        help='Margin for triplet loss')
    parser.add_argument('--lambda_disc', type=float, default=0.1,
                        help='Weight of layer discriminability loss L_disc. '
                             'Directly supervises routing weights to concentrate on '
                             'the most discriminative layer. (0 to disable)')
    parser.add_argument('--disc_temperature', type=float, default=1.0,
                        help='Softmax temperature for disc→target distribution '
                             '(lower = sharper target; default 1.0)')
    parser.add_argument('--lambda_entropy', type=float, default=0.01,
                        help='Weight of entropy regularisation on routing weights. '
                             'Prevents routing from collapsing to a single layer '
                             '(e.g. layer 1). Only applied to target token positions. '
                             '(0 to disable, recommended 0.005–0.05)')

    # Training
    parser.add_argument('--epochs',      type=int,   default=10)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=3e-4)
    parser.add_argument('--num_workers', type=int,   default=4)
    parser.add_argument('--save_every',  type=int,   default=500,
                        help='Save checkpoint every N global steps')
    parser.add_argument('--seed',        type=int,   default=42,
                        help='Base random seed; each DDP rank uses seed+rank for data diversity')

    # WandB
    parser.add_argument('--use_wandb',      action='store_true',
                        help='Enable Weights & Biases logging (rank 0 only)')
    parser.add_argument('--wandb_project',  type=str, default='z-image-router',
                        help='WandB project name')
    parser.add_argument('--wandb_run',      type=str, default='',
                        help='WandB run name (auto-generated if empty)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Only rank 0 prints the config header (DDP may not be init yet, check env)
    if int(os.environ.get('RANK', 0)) == 0:
        print("=" * 60)
        print("  DynamicTokenRouter Contrastive Training")
        print("=" * 60)
        for k, v in vars(args).items():
            print(f"  {k:<20} = {v}")
        print("=" * 60)
    train(args)
