"""
Random p-n verification for refiner embeddings.

For each trial:
  - sample B triplets: (anchor, positive, negative)
  - encode pre-refiner and post-refiner sentence embeddings
  - compute p-n = sim(anchor, positive) - sim(anchor, negative)
  - report per-trial and overall statistics
"""

import argparse
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from utils import load_from_local_dir  # noqa: E402

def run_context_refiner(
    transformer: torch.nn.Module,
    token_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Project to transformer dim and run context_refiner, returning all stage outputs."""
    model = transformer.module if hasattr(transformer, "module") else transformer
    bsz, seq_len, _ = token_hidden.shape
    device = token_hidden.device
    dtype = next(model.parameters()).dtype
    attn_mask = attention_mask.bool()
    stage_outputs: dict[str, torch.Tensor] = {}

    cap_feats = model.cap_embedder(token_hidden.to(dtype))
    cap_feats = cap_feats.clone()
    cap_feats[~attn_mask] = model.cap_pad_token.to(dtype)
    stage_outputs["cap_embedder"] = cap_feats

    pos_ids = torch.zeros((bsz, seq_len, 3), dtype=torch.int32, device=device)
    pos_ids[:, :, 0] = torch.arange(1, seq_len + 1, dtype=torch.int32, device=device).unsqueeze(0).expand(bsz, -1)
    cap_freqs = model.rope_embedder(pos_ids.view(-1, 3)).view(bsz, seq_len, -1)

    refined = cap_feats
    for i, layer in enumerate(model.context_refiner):
        refined = layer(refined, attn_mask, cap_freqs)
        stage_outputs[f"context_refiner_{i}"] = refined
    return refined, stage_outputs


CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789"
NUMBER_WORDS = [
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
]
OBJECT_WORDS = [
    "apple", "banana", "backpack", "bicycle", "bird",
    "book", "bottle", "camera", "chair", "cup",
    "dog", "elephant", "flower", "hammer", "laptop",
    "pencil", "phone", "shoe", "table", "tree",
]


def rand_gibberish(min_len: int = 6, max_len: int = 14) -> str:
    n = random.randint(min_len, max_len)
    return "".join(random.choice(CHARSET) for _ in range(n))


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
    seen = set()
    out = []
    for c in candidates:
        k = c.strip().lower()
        if k and k not in seen:
            out.append(c)
            seen.add(k)
    return out


def build_gibberish_triplet() -> tuple[str, str, str, str, str]:
    """
    Build a noisy/gibberish triplet for stress testing anisotropy:
      anchor:   "xxxdasda qwe12"
      positive: "a photo of xxxdasda qwe12"      (keeps anchor core)
      negative: "a photo of asd98zx qqq77"       (different gibberish core)
    """
    anchor_word = rand_gibberish()
    neg_word = rand_gibberish()
    core_a = f"{anchor_word} {rand_gibberish()}"
    core_n = f"{neg_word} {rand_gibberish()}{rand_gibberish()}{rand_gibberish()}"
    anchor = core_a
    positive = f"{core_a}"
    negative = f"{core_n}"
    return anchor, positive, negative, anchor_word, neg_word


def build_counting_triplet() -> tuple[str, str, str, str, str]:
    count = random.choice(NUMBER_WORDS)
    neg_count = random.choice([w for w in NUMBER_WORDS if w != count])
    obj = random.choice(OBJECT_WORDS)
    anchor = f"{count} {obj}"

    pos_candidates = [v for v in make_anchor_variants(anchor) if v.strip().lower() != anchor.lower()]
    positive = random.choice(pos_candidates) if pos_candidates else anchor

    neg_base = anchor.replace(count, neg_count, 1)
    neg_candidates = make_anchor_variants(neg_base)
    negative = random.choice(neg_candidates) if neg_candidates else neg_base

    return anchor, positive, negative, count, neg_count


def build_triplet(mode: str) -> tuple[str, str, str, str, str]:
    if mode == "gibberish":
        return build_gibberish_triplet()
    return build_counting_triplet()


def pool_content(token_hidden: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, special_ids: set[int]) -> torch.Tensor:
    is_special = torch.zeros_like(input_ids, dtype=torch.bool)
    for sid in special_ids:
        is_special |= (input_ids == sid)
    content_mask = attention_mask.bool() & ~is_special
    m = content_mask.unsqueeze(-1).float()
    pooled = (token_hidden.float() * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
    return F.normalize(pooled, dim=-1)


def mean_std_min_max(x: torch.Tensor) -> tuple[float, float, float, float]:
    return float(x.mean()), float(x.std()), float(x.min()), float(x.max())


def build_frozen_random_projection(in_dim: int, out_dim: int, seed: int, device: torch.device) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    proj = torch.randn(in_dim, out_dim, generator=gen, dtype=torch.float32) / (out_dim ** 0.5)
    return proj.to(device=device)


def random_project_and_normalize(x: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float() @ proj, dim=-1)


def mean_center_and_normalize(x: torch.Tensor) -> torch.Tensor:
    """Subtract batch mean and then apply L2 normalization."""
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    return F.normalize(x, dim=-1)


def mean_center_std_and_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Subtract batch mean, divide by batch std, then L2 normalize."""
    x = x.float()
    x = x - x.mean(dim=0, keepdim=True)
    x = x / (x.std(dim=0, keepdim=True, unbiased=False) + eps)
    return F.normalize(x, dim=-1)


def update_stage_metrics(
    metrics: dict[str, dict[str, list[torch.Tensor]]],
    stage_name: str,
    stage_hidden: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    special_ids: set[int],
    bsz: int,
) -> None:
    vec = pool_content(stage_hidden, input_ids, attention_mask, special_ids)
    ea, ep, en = vec[:bsz], vec[bsz : 2 * bsz], vec[2 * bsz :]
    sim_ap = (ea * ep).sum(dim=-1)
    sim_an = (ea * en).sum(dim=-1)
    pn = sim_ap - sim_an

    if stage_name not in metrics:
        metrics[stage_name] = {"sim_ap": [], "sim_an": [], "pn": []}
    metrics[stage_name]["sim_ap"].append(sim_ap.detach().cpu())
    metrics[stage_name]["sim_an"].append(sim_an.detach().cpu())
    metrics[stage_name]["pn"].append(pn.detach().cpu())


def find_target_idx(tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor, target_word: str) -> int:
    valid_ids = input_ids[attention_mask.bool()].tolist()
    tw = target_word.lower().strip()
    for idx, tid in enumerate(valid_ids):
        token_str = tokenizer.decode([tid], skip_special_tokens=True).lower().strip()
        if tw in token_str or token_str in tw:
            return idx
    return -1


def extract_token_features(hidden: torch.Tensor, indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    feats = []
    valid = []
    seq_len = hidden.shape[1]
    for i, tidx in enumerate(indices):
        if 0 <= tidx < seq_len:
            feats.append(F.normalize(hidden[i, tidx, :].float(), dim=-1))
            valid.append(True)
        else:
            feats.append(torch.zeros(hidden.shape[-1], device=hidden.device, dtype=torch.float32))
            valid.append(False)
    return torch.stack(feats, dim=0), torch.tensor(valid, device=hidden.device, dtype=torch.bool)


def select_source_hidden(hidden_states: tuple[torch.Tensor, ...], args: argparse.Namespace) -> tuple[torch.Tensor, str]:
    if args.llm_input_mode == "single":
        idx = args.llm_input_layer_idx
        return hidden_states[idx], f"single:{idx}"
    start = args.llm_layer_start
    end = args.llm_layer_end
    if end < start:
        raise ValueError(f"Invalid range: llm_layer_end({end}) < llm_layer_start({start})")
    picked = hidden_states[start : end + 1]
    if len(picked) == 0:
        raise ValueError(f"Empty layer range: [{start}, {end}]")
    stacked = torch.stack([h.float() for h in picked], dim=0)
    return stacked.mean(dim=0).to(hidden_states[start].dtype), f"avg:{start}-{end}"


def main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    components = load_from_local_dir(
        args.model_dir,
        device="cpu",
        dtype=torch.bfloat16,
        verbose=True,
    )
    transformer = components["transformer"].to(device).eval()
    text_encoder = components["text_encoder"].to(device).eval()
    tokenizer = components["tokenizer"]

    for p in transformer.parameters():
        p.requires_grad_(False)
    for p in text_encoder.parameters():
        p.requires_grad_(False)

    special_ids = set(tokenizer.all_special_ids)
    all_post = []
    all_src = []
    all_post_l2 = []
    all_src_l2 = []
    all_post_ap = []
    all_post_an = []
    all_src_ap = []
    all_src_an = []
    all_post_tok = []
    all_post_tok_ap = []
    all_post_tok_an = []
    all_src_tok = []
    all_src_tok_ap = []
    all_src_tok_an = []
    all_post_mc = []
    all_post_mc_ap = []
    all_post_mc_an = []
    all_post_zs = []
    all_post_zs_ap = []
    all_post_zs_an = []
    all_post_tok_mc = []
    all_post_tok_mc_ap = []
    all_post_tok_mc_an = []
    all_post_tok_zs = []
    all_post_tok_zs_ap = []
    all_post_tok_zs_an = []
    all_post_rand = []
    all_post_rand_ap = []
    all_post_rand_an = []
    all_post_tok_rand = []
    all_post_tok_rand_ap = []
    all_post_tok_rand_an = []
    stage_metrics: dict[str, dict[str, list[torch.Tensor]]] = {}
    rand_proj = None
    source_desc = ""

    for t in range(args.trials):
        triplets = [build_triplet(args.prompt_mode) for _ in range(args.batch_size)]
        anchors = [x[0] for x in triplets]
        positives = [x[1] for x in triplets]
        negatives = [x[2] for x in triplets]
        anchor_words = [x[3] for x in triplets]
        neg_words = [x[4] for x in triplets]

        texts = anchors + positives + negatives
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)

        with torch.no_grad():
            out = text_encoder(input_ids=input_ids, attention_mask=attention_mask.bool(), output_hidden_states=True)
            source_tok, source_desc = select_source_hidden(out.hidden_states, args)
            post_tok, stage_outputs = run_context_refiner(transformer, source_tok, attention_mask)

            source_vec = pool_content(source_tok, input_ids, attention_mask, special_ids)
            post_vec = pool_content(post_tok, input_ids, attention_mask, special_ids)

        b = args.batch_size
        if rand_proj is None:
            rand_proj = build_frozen_random_projection(post_vec.shape[-1], args.proj_dim, args.seed + 2026, device)
        ea_src, ep_src, en_src = source_vec[:b], source_vec[b:2 * b], source_vec[2 * b:]
        ea_post, ep_post, en_post = post_vec[:b], post_vec[b:2 * b], post_vec[2 * b:]

        combined_post = torch.cat([ea_post, ep_post, en_post], dim=0)
        combined_post_mc = mean_center_and_normalize(combined_post)
        ea_post_mc = combined_post_mc[:b]
        ep_post_mc = combined_post_mc[b:2 * b]
        en_post_mc = combined_post_mc[2 * b:]
        combined_post_zs = mean_center_std_and_normalize(combined_post)
        ea_post_zs = combined_post_zs[:b]
        ep_post_zs = combined_post_zs[b:2 * b]
        en_post_zs = combined_post_zs[2 * b:]

        ea_post_rand = random_project_and_normalize(ea_post, rand_proj)
        ep_post_rand = random_project_and_normalize(ep_post, rand_proj)
        en_post_rand = random_project_and_normalize(en_post, rand_proj)

        sim_ap_src = (ea_src * ep_src).sum(dim=-1)
        sim_an_src = (ea_src * en_src).sum(dim=-1)
        pn_src = sim_ap_src - sim_an_src
        sim_ap_post = (ea_post * ep_post).sum(dim=-1)
        sim_an_post = (ea_post * en_post).sum(dim=-1)
        pn_post = sim_ap_post - sim_an_post

        sim_ap_post_mc = (ea_post_mc * ep_post_mc).sum(dim=-1)
        sim_an_post_mc = (ea_post_mc * en_post_mc).sum(dim=-1)
        pn_post_mc = sim_ap_post_mc - sim_an_post_mc
        sim_ap_post_zs = (ea_post_zs * ep_post_zs).sum(dim=-1)
        sim_an_post_zs = (ea_post_zs * en_post_zs).sum(dim=-1)
        pn_post_zs = sim_ap_post_zs - sim_an_post_zs

        sim_ap_post_rand = (ea_post_rand * ep_post_rand).sum(dim=-1)
        sim_an_post_rand = (ea_post_rand * en_post_rand).sum(dim=-1)
        pn_post_rand = sim_ap_post_rand - sim_an_post_rand

        # Euclidean distance margin: >0 means positive is closer than negative.
        dist_ap_src = torch.norm(ea_src - ep_src, p=2, dim=-1)
        dist_an_src = torch.norm(ea_src - en_src, p=2, dim=-1)
        l2_margin_src = dist_an_src - dist_ap_src
        dist_ap_post = torch.norm(ea_post - ep_post, p=2, dim=-1)
        dist_an_post = torch.norm(ea_post - en_post, p=2, dim=-1)
        l2_margin_post = dist_an_post - dist_ap_post

        # Token-level p-n: use anchor target token vs positive same token / negative token.
        a_ids = input_ids[:b]
        p_ids = input_ids[b:2 * b]
        n_ids = input_ids[2 * b:]
        a_mask = attention_mask[:b]
        p_mask = attention_mask[b:2 * b]
        n_mask = attention_mask[2 * b:]

        a_tidx = [find_target_idx(tokenizer, a_ids[i], a_mask[i], anchor_words[i]) for i in range(b)]
        p_tidx = [find_target_idx(tokenizer, p_ids[i], p_mask[i], anchor_words[i]) for i in range(b)]
        n_tidx = [find_target_idx(tokenizer, n_ids[i], n_mask[i], neg_words[i]) for i in range(b)]

        a_src_tok, vm_a_src = extract_token_features(source_tok[:b], a_tidx)
        p_src_tok, vm_p_src = extract_token_features(source_tok[b:2 * b], p_tidx)
        n_src_tok, vm_n_src = extract_token_features(source_tok[2 * b:], n_tidx)
        a_post_tok, vm_a_post = extract_token_features(post_tok[:b], a_tidx)
        p_post_tok, vm_p_post = extract_token_features(post_tok[b:2 * b], p_tidx)
        n_post_tok, vm_n_post = extract_token_features(post_tok[2 * b:], n_tidx)

        valid_src = vm_a_src & vm_p_src & vm_n_src
        valid_post = vm_a_post & vm_p_post & vm_n_post

        tok_sim_ap_src = (a_src_tok[valid_src] * p_src_tok[valid_src]).sum(dim=-1)
        tok_sim_an_src = (a_src_tok[valid_src] * n_src_tok[valid_src]).sum(dim=-1)
        tok_sim_ap_post = (a_post_tok[valid_post] * p_post_tok[valid_post]).sum(dim=-1)
        tok_sim_an_post = (a_post_tok[valid_post] * n_post_tok[valid_post]).sum(dim=-1)

        tok_pn_src = tok_sim_ap_src - tok_sim_an_src
        tok_pn_post = tok_sim_ap_post - tok_sim_an_post

        combined_post_tok = torch.cat(
            [a_post_tok[valid_post], p_post_tok[valid_post], n_post_tok[valid_post]],
            dim=0,
        )
        if combined_post_tok.numel() > 0:
            combined_post_tok_mc = mean_center_and_normalize(combined_post_tok)
            nv = int(valid_post.sum().item())
            a_post_tok_mc = combined_post_tok_mc[:nv]
            p_post_tok_mc = combined_post_tok_mc[nv : 2 * nv]
            n_post_tok_mc = combined_post_tok_mc[2 * nv :]
            tok_sim_ap_post_mc = (a_post_tok_mc * p_post_tok_mc).sum(dim=-1)
            tok_sim_an_post_mc = (a_post_tok_mc * n_post_tok_mc).sum(dim=-1)
            tok_pn_post_mc = tok_sim_ap_post_mc - tok_sim_an_post_mc

            combined_post_tok_zs = mean_center_std_and_normalize(combined_post_tok)
            a_post_tok_zs = combined_post_tok_zs[:nv]
            p_post_tok_zs = combined_post_tok_zs[nv : 2 * nv]
            n_post_tok_zs = combined_post_tok_zs[2 * nv :]
            tok_sim_ap_post_zs = (a_post_tok_zs * p_post_tok_zs).sum(dim=-1)
            tok_sim_an_post_zs = (a_post_tok_zs * n_post_tok_zs).sum(dim=-1)
            tok_pn_post_zs = tok_sim_ap_post_zs - tok_sim_an_post_zs
        else:
            tok_sim_ap_post_mc = torch.empty(0, device=device)
            tok_sim_an_post_mc = torch.empty(0, device=device)
            tok_pn_post_mc = torch.empty(0, device=device)
            tok_sim_ap_post_zs = torch.empty(0, device=device)
            tok_sim_an_post_zs = torch.empty(0, device=device)
            tok_pn_post_zs = torch.empty(0, device=device)

        a_post_tok_rand = random_project_and_normalize(a_post_tok[valid_post], rand_proj)
        p_post_tok_rand = random_project_and_normalize(p_post_tok[valid_post], rand_proj)
        n_post_tok_rand = random_project_and_normalize(n_post_tok[valid_post], rand_proj)
        tok_sim_ap_post_rand = (a_post_tok_rand * p_post_tok_rand).sum(dim=-1)
        tok_sim_an_post_rand = (a_post_tok_rand * n_post_tok_rand).sum(dim=-1)
        tok_pn_post_rand = tok_sim_ap_post_rand - tok_sim_an_post_rand

        all_src.append(pn_src.cpu())
        all_post.append(pn_post.cpu())
        all_src_l2.append(l2_margin_src.cpu())
        all_post_l2.append(l2_margin_post.cpu())
        all_src_ap.append(sim_ap_src.cpu())
        all_src_an.append(sim_an_src.cpu())
        all_post_ap.append(sim_ap_post.cpu())
        all_post_an.append(sim_an_post.cpu())
        all_post_mc.append(pn_post_mc.cpu())
        all_post_mc_ap.append(sim_ap_post_mc.cpu())
        all_post_mc_an.append(sim_an_post_mc.cpu())
        all_post_zs.append(pn_post_zs.cpu())
        all_post_zs_ap.append(sim_ap_post_zs.cpu())
        all_post_zs_an.append(sim_an_post_zs.cpu())
        all_post_rand.append(pn_post_rand.cpu())
        all_post_rand_ap.append(sim_ap_post_rand.cpu())
        all_post_rand_an.append(sim_an_post_rand.cpu())
        if tok_pn_src.numel() > 0:
            all_src_tok.append(tok_pn_src.cpu())
            all_src_tok_ap.append(tok_sim_ap_src.cpu())
            all_src_tok_an.append(tok_sim_an_src.cpu())
        if tok_pn_post.numel() > 0:
            all_post_tok.append(tok_pn_post.cpu())
            all_post_tok_ap.append(tok_sim_ap_post.cpu())
            all_post_tok_an.append(tok_sim_an_post.cpu())
            all_post_tok_mc.append(tok_pn_post_mc.cpu())
            all_post_tok_mc_ap.append(tok_sim_ap_post_mc.cpu())
            all_post_tok_mc_an.append(tok_sim_an_post_mc.cpu())
            all_post_tok_zs.append(tok_pn_post_zs.cpu())
            all_post_tok_zs_ap.append(tok_sim_ap_post_zs.cpu())
            all_post_tok_zs_an.append(tok_sim_an_post_zs.cpu())
            all_post_tok_rand.append(tok_pn_post_rand.cpu())
            all_post_tok_rand_ap.append(tok_sim_ap_post_rand.cpu())
            all_post_tok_rand_an.append(tok_sim_an_post_rand.cpu())

        # Stage-wise sentence-level similarity stats: context_refiner block outputs only.
        for k, v in stage_outputs.items():
            if k.startswith("context_refiner_"):
                update_stage_metrics(stage_metrics, k, v, input_ids, attention_mask, special_ids, b)

        src_stats = mean_std_min_max(pn_src)
        post_stats = mean_std_min_max(pn_post)
        post_mc_stats = mean_std_min_max(pn_post_mc)
        post_zs_stats = mean_std_min_max(pn_post_zs)
        post_rand_stats = mean_std_min_max(pn_post_rand)
        src_l2_stats = mean_std_min_max(l2_margin_src)
        post_l2_stats = mean_std_min_max(l2_margin_post)
        print(
            f"[trial {t+1:02d}/{args.trials}] "
            f"src_p-n mean={src_stats[0]:+.6f} std={src_stats[1]:.6f} min={src_stats[2]:+.6f} max={src_stats[3]:+.6f} | "
            f"post_p-n mean={post_stats[0]:+.6f} std={post_stats[1]:.6f} min={post_stats[2]:+.6f} max={post_stats[3]:+.6f} | "
            f"mc_post_p-n mean={post_mc_stats[0]:+.6f} | "
            f"zs_post_p-n mean={post_zs_stats[0]:+.6f} | "
            f"rand_post_p-n mean={post_rand_stats[0]:+.6f} | "
            f"src_l2_margin mean={src_l2_stats[0]:+.6f} post_l2_margin mean={post_l2_stats[0]:+.6f} | "
            f"token_valid src={int(valid_src.sum())}/{b} post={int(valid_post.sum())}/{b}"
        )

    all_src_t = torch.cat(all_src, dim=0)
    all_post_t = torch.cat(all_post, dim=0)
    all_src_l2_t = torch.cat(all_src_l2, dim=0)
    all_post_l2_t = torch.cat(all_post_l2, dim=0)
    all_src_ap_t = torch.cat(all_src_ap, dim=0)
    all_src_an_t = torch.cat(all_src_an, dim=0)
    all_post_ap_t = torch.cat(all_post_ap, dim=0)
    all_post_an_t = torch.cat(all_post_an, dim=0)
    all_post_mc_t = torch.cat(all_post_mc, dim=0)
    all_post_mc_ap_t = torch.cat(all_post_mc_ap, dim=0)
    all_post_mc_an_t = torch.cat(all_post_mc_an, dim=0)
    all_post_zs_t = torch.cat(all_post_zs, dim=0)
    all_post_zs_ap_t = torch.cat(all_post_zs_ap, dim=0)
    all_post_zs_an_t = torch.cat(all_post_zs_an, dim=0)
    all_post_rand_t = torch.cat(all_post_rand, dim=0)
    all_post_rand_ap_t = torch.cat(all_post_rand_ap, dim=0)
    all_post_rand_an_t = torch.cat(all_post_rand_an, dim=0)
    src_all = mean_std_min_max(all_src_t)
    post_all = mean_std_min_max(all_post_t)

    print("\n=== Overall ===")
    print(f"[Source LLM] {source_desc}")
    print(
        f"SRC  p-n mean={src_all[0]:+.6f} std={src_all[1]:.6f} min={src_all[2]:+.6f} max={src_all[3]:+.6f} "
        f"pos_rate={(all_src_t > 0).float().mean().item() * 100:.2f}%"
    )
    print(
        f"POST p-n mean={post_all[0]:+.6f} std={post_all[1]:.6f} min={post_all[2]:+.6f} max={post_all[3]:+.6f} "
        f"pos_rate={(all_post_t > 0).float().mean().item() * 100:.2f}%"
    )
    print(
        f"SRC  l2_margin mean={all_src_l2_t.mean().item():+.6f} std={all_src_l2_t.std().item():.6f} "
        f"min={all_src_l2_t.min().item():+.6f} max={all_src_l2_t.max().item():+.6f} "
        f"pos_rate={(all_src_l2_t > 0).float().mean().item() * 100:.2f}%"
    )
    print(
        f"POST l2_margin mean={all_post_l2_t.mean().item():+.6f} std={all_post_l2_t.std().item():.6f} "
        f"min={all_post_l2_t.min().item():+.6f} max={all_post_l2_t.max().item():+.6f} "
        f"pos_rate={(all_post_l2_t > 0).float().mean().item() * 100:.2f}%"
    )
    print(
        f"SRC  sim(a,p) mean={all_src_ap_t.mean().item():.6f}  sim(a,n) mean={all_src_an_t.mean().item():.6f}"
    )
    print(
        f"POST sim(a,p) mean={all_post_ap_t.mean().item():.6f}  sim(a,n) mean={all_post_an_t.mean().item():.6f}"
    )
    print("\n=== Mean Centered ===")
    print(
        f"POST mc_p-n mean={all_post_mc_t.mean().item():+.6f} std={all_post_mc_t.std().item():.6f} "
        f"min={all_post_mc_t.min().item():+.6f} max={all_post_mc_t.max().item():+.6f} "
        f"pos_rate={(all_post_mc_t > 0).float().mean().item() * 100:.2f}%"
    )
    print(
        f"POST mc_sim(a,p) mean={all_post_mc_ap_t.mean().item():.6f}  mc_sim(a,n) mean={all_post_mc_an_t.mean().item():.6f}"
    )
    print("\n=== Mean Centered + /std ===")
    print(
        f"POST zs_p-n mean={all_post_zs_t.mean().item():+.6f} std={all_post_zs_t.std().item():.6f} "
        f"min={all_post_zs_t.min().item():+.6f} max={all_post_zs_t.max().item():+.6f} "
        f"pos_rate={(all_post_zs_t > 0).float().mean().item() * 100:.2f}%"
    )
    print(
        f"POST zs_sim(a,p) mean={all_post_zs_ap_t.mean().item():.6f}  zs_sim(a,n) mean={all_post_zs_an_t.mean().item():.6f}"
    )
    print(f"\n=== Frozen random projection (dim={args.proj_dim}) ===")
    print(
        f"POST rand_p-n mean={all_post_rand_t.mean().item():+.6f} std={all_post_rand_t.std().item():.6f} "
        f"min={all_post_rand_t.min().item():+.6f} max={all_post_rand_t.max().item():+.6f} "
        f"pos_rate={(all_post_rand_t > 0).float().mean().item() * 100:.2f}%"
    )
    print(
        f"POST rand_sim(a,p) mean={all_post_rand_ap_t.mean().item():.6f}  rand_sim(a,n) mean={all_post_rand_an_t.mean().item():.6f}"
    )

    if all_src_tok and all_post_tok:
        src_tok_all = torch.cat(all_src_tok, dim=0)
        src_tok_ap_all = torch.cat(all_src_tok_ap, dim=0)
        src_tok_an_all = torch.cat(all_src_tok_an, dim=0)
        post_tok_all = torch.cat(all_post_tok, dim=0)
        post_tok_ap_all = torch.cat(all_post_tok_ap, dim=0)
        post_tok_an_all = torch.cat(all_post_tok_an, dim=0)
        post_tok_mc_all = torch.cat(all_post_tok_mc, dim=0)
        post_tok_mc_ap_all = torch.cat(all_post_tok_mc_ap, dim=0)
        post_tok_mc_an_all = torch.cat(all_post_tok_mc_an, dim=0)
        post_tok_zs_all = torch.cat(all_post_tok_zs, dim=0)
        post_tok_zs_ap_all = torch.cat(all_post_tok_zs_ap, dim=0)
        post_tok_zs_an_all = torch.cat(all_post_tok_zs_an, dim=0)
        post_tok_rand_all = torch.cat(all_post_tok_rand, dim=0)
        post_tok_rand_ap_all = torch.cat(all_post_tok_rand_ap, dim=0)
        post_tok_rand_an_all = torch.cat(all_post_tok_rand_an, dim=0)
        src_tok_stats = mean_std_min_max(src_tok_all)
        post_tok_stats = mean_std_min_max(post_tok_all)
        print("\n=== Token-level p-n (target token only) ===")
        print(
            f"SRC  token_sim(a,p) mean={src_tok_ap_all.mean().item():.6f}  token_sim(a,n) mean={src_tok_an_all.mean().item():.6f}"
        )
        print(
            f"POST token_sim(a,p) mean={post_tok_ap_all.mean().item():.6f}  token_sim(a,n) mean={post_tok_an_all.mean().item():.6f}"
        )
        print(
            f"SRC  token_p-n mean={src_tok_stats[0]:+.6f} std={src_tok_stats[1]:.6f} "
            f"min={src_tok_stats[2]:+.6f} max={src_tok_stats[3]:+.6f} "
            f"pos_rate={(src_tok_all > 0).float().mean().item() * 100:.2f}%"
        )
        print(
            f"POST token_p-n mean={post_tok_stats[0]:+.6f} std={post_tok_stats[1]:.6f} "
            f"min={post_tok_stats[2]:+.6f} max={post_tok_stats[3]:+.6f} "
            f"pos_rate={(post_tok_all > 0).float().mean().item() * 100:.2f}%"
        )
        print("\n=== Token-level Mean Centered ===")
        print(
            f"POST mc_token_sim(a,p) mean={post_tok_mc_ap_all.mean().item():.6f}  mc_token_sim(a,n) mean={post_tok_mc_an_all.mean().item():.6f}"
        )
        print(
            f"POST mc_token_p-n mean={post_tok_mc_all.mean().item():+.6f} std={post_tok_mc_all.std().item():.6f} "
            f"min={post_tok_mc_all.min().item():+.6f} max={post_tok_mc_all.max().item():+.6f} "
            f"pos_rate={(post_tok_mc_all > 0).float().mean().item() * 100:.2f}%"
        )
        print("\n=== Token-level Mean Centered + /std ===")
        print(
            f"POST zs_token_sim(a,p) mean={post_tok_zs_ap_all.mean().item():.6f}  zs_token_sim(a,n) mean={post_tok_zs_an_all.mean().item():.6f}"
        )
        print(
            f"POST zs_token_p-n mean={post_tok_zs_all.mean().item():+.6f} std={post_tok_zs_all.std().item():.6f} "
            f"min={post_tok_zs_all.min().item():+.6f} max={post_tok_zs_all.max().item():+.6f} "
            f"pos_rate={(post_tok_zs_all > 0).float().mean().item() * 100:.2f}%"
        )
        print(f"\n=== Token-level Frozen random projection (dim={args.proj_dim}) ===")
        print(
            f"POST rand_token_sim(a,p) mean={post_tok_rand_ap_all.mean().item():.6f}  rand_token_sim(a,n) mean={post_tok_rand_an_all.mean().item():.6f}"
        )
        print(
            f"POST rand_token_p-n mean={post_tok_rand_all.mean().item():+.6f} std={post_tok_rand_all.std().item():.6f} "
            f"min={post_tok_rand_all.min().item():+.6f} max={post_tok_rand_all.max().item():+.6f} "
            f"pos_rate={(post_tok_rand_all > 0).float().mean().item() * 100:.2f}%"
        )

    # Layer-wise similarity summary (context_refiner outputs only)
    print("\n=== Layer-wise sentence similarity (overall) ===")
    order = sorted(
        [k for k in stage_metrics.keys() if k.startswith("context_refiner_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    for name in order:
        if name not in stage_metrics:
            continue
        sim_ap = torch.cat(stage_metrics[name]["sim_ap"], dim=0)
        sim_an = torch.cat(stage_metrics[name]["sim_an"], dim=0)
        pn = torch.cat(stage_metrics[name]["pn"], dim=0)
        print(
            f"{name:<28} "
            f"sim(a,p)={sim_ap.mean().item():.6f}  "
            f"sim(a,n)={sim_an.mean().item():.6f}  "
            f"p-n={pn.mean().item():+.6f}  "
            f"pn_pos_rate={(pn > 0).float().mean().item() * 100:.2f}%"
        )

    print("first 10 POST p-n:", [round(float(x), 6) for x in all_post_t[:10]])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default="ckpts/Z-Image-Turbo")
    p.add_argument("--trials", type=int, default=10, help="How many random batches to test")
    p.add_argument("--batch_size", type=int, default=32, help="Triplets per trial")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--llm_input_layer_idx",
        type=int,
        default=1,
        help="Which text_encoder hidden_states layer is fed into refiner. "
             "0=embedding output, 1=first transformer layer, -2=penultimate layer.",
    )
    p.add_argument(
        "--llm_input_mode",
        type=str,
        default="single",
        choices=["single", "avg_range"],
        help="single: use one hidden layer; avg_range: average a layer range before refiner.",
    )
    p.add_argument("--llm_layer_start", type=int, default=10, help="Start layer idx (inclusive) for avg_range mode.")
    p.add_argument("--llm_layer_end", type=int, default=20, help="End layer idx (inclusive) for avg_range mode.")
    p.add_argument("--proj_dim", type=int, default=256, help="Output dim for frozen random projection")
    p.add_argument(
        "--prompt_mode",
        type=str,
        default="counting",
        choices=["counting", "gibberish"],
        help="Triplet construction mode: counting matches training-style templates.",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

