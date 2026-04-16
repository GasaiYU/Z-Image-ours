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


def rand_gibberish(min_len: int = 6, max_len: int = 14) -> str:
    n = random.randint(min_len, max_len)
    return "".join(random.choice(CHARSET) for _ in range(n))


def build_triplet() -> tuple[str, str, str, str, str]:
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
    positive = f"a photo of {core_a}"
    negative = f"a photo of {core_n}"
    return anchor, positive, negative, anchor_word, neg_word


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
    all_pre = []
    all_post = []
    all_pre_l2 = []
    all_post_l2 = []
    all_pre_ap = []
    all_pre_an = []
    all_post_ap = []
    all_post_an = []
    all_pre_tok = []
    all_post_tok = []
    stage_metrics: dict[str, dict[str, list[torch.Tensor]]] = {}

    for t in range(args.trials):
        triplets = [build_triplet() for _ in range(args.batch_size)]
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
            pre_tok = out.hidden_states[-2]
            post_tok, stage_outputs = run_context_refiner(transformer, pre_tok, attention_mask)

            pre_vec = pool_content(pre_tok, input_ids, attention_mask, special_ids)
            post_vec = pool_content(post_tok, input_ids, attention_mask, special_ids)

        b = args.batch_size
        ea_pre, ep_pre, en_pre = pre_vec[:b], pre_vec[b:2 * b], pre_vec[2 * b:]
        ea_post, ep_post, en_post = post_vec[:b], post_vec[b:2 * b], post_vec[2 * b:]

        sim_ap_pre = (ea_pre * ep_pre).sum(dim=-1)
        sim_an_pre = (ea_pre * en_pre).sum(dim=-1)
        sim_ap_post = (ea_post * ep_post).sum(dim=-1)
        sim_an_post = (ea_post * en_post).sum(dim=-1)
        pn_pre = sim_ap_pre - sim_an_pre
        pn_post = sim_ap_post - sim_an_post

        # Euclidean distance margin: >0 means positive is closer than negative.
        dist_ap_pre = torch.norm(ea_pre - ep_pre, p=2, dim=-1)
        dist_an_pre = torch.norm(ea_pre - en_pre, p=2, dim=-1)
        dist_ap_post = torch.norm(ea_post - ep_post, p=2, dim=-1)
        dist_an_post = torch.norm(ea_post - en_post, p=2, dim=-1)
        l2_margin_pre = dist_an_pre - dist_ap_pre
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

        a_pre_tok, vm_a_pre = extract_token_features(pre_tok[:b], a_tidx)
        p_pre_tok, vm_p_pre = extract_token_features(pre_tok[b:2 * b], p_tidx)
        n_pre_tok, vm_n_pre = extract_token_features(pre_tok[2 * b:], n_tidx)
        a_post_tok, vm_a_post = extract_token_features(post_tok[:b], a_tidx)
        p_post_tok, vm_p_post = extract_token_features(post_tok[b:2 * b], p_tidx)
        n_post_tok, vm_n_post = extract_token_features(post_tok[2 * b:], n_tidx)

        valid_pre = vm_a_pre & vm_p_pre & vm_n_pre
        valid_post = vm_a_post & vm_p_post & vm_n_post
        tok_pn_pre = (a_pre_tok[valid_pre] * p_pre_tok[valid_pre]).sum(dim=-1) - (a_pre_tok[valid_pre] * n_pre_tok[valid_pre]).sum(dim=-1)
        tok_pn_post = (a_post_tok[valid_post] * p_post_tok[valid_post]).sum(dim=-1) - (a_post_tok[valid_post] * n_post_tok[valid_post]).sum(dim=-1)

        all_pre.append(pn_pre.cpu())
        all_post.append(pn_post.cpu())
        all_pre_l2.append(l2_margin_pre.cpu())
        all_post_l2.append(l2_margin_post.cpu())
        all_pre_ap.append(sim_ap_pre.cpu())
        all_pre_an.append(sim_an_pre.cpu())
        all_post_ap.append(sim_ap_post.cpu())
        all_post_an.append(sim_an_post.cpu())
        if tok_pn_pre.numel() > 0:
            all_pre_tok.append(tok_pn_pre.cpu())
        if tok_pn_post.numel() > 0:
            all_post_tok.append(tok_pn_post.cpu())

        # Stage-wise sentence-level similarity stats:
        # text_encoder(-2), cap_embedder, and each context_refiner block output.
        update_stage_metrics(stage_metrics, "text_encoder_pre_refiner", pre_tok, input_ids, attention_mask, special_ids, b)
        update_stage_metrics(stage_metrics, "cap_embedder", stage_outputs["cap_embedder"], input_ids, attention_mask, special_ids, b)
        for k, v in stage_outputs.items():
            if k.startswith("context_refiner_"):
                update_stage_metrics(stage_metrics, k, v, input_ids, attention_mask, special_ids, b)

        pre_stats = mean_std_min_max(pn_pre)
        post_stats = mean_std_min_max(pn_post)
        pre_l2_stats = mean_std_min_max(l2_margin_pre)
        post_l2_stats = mean_std_min_max(l2_margin_post)
        print(
            f"[trial {t+1:02d}/{args.trials}] "
            f"pre_p-n mean={pre_stats[0]:+.6f} std={pre_stats[1]:.6f} min={pre_stats[2]:+.6f} max={pre_stats[3]:+.6f} | "
            f"post_p-n mean={post_stats[0]:+.6f} std={post_stats[1]:.6f} min={post_stats[2]:+.6f} max={post_stats[3]:+.6f} | "
            f"pre_l2_margin mean={pre_l2_stats[0]:+.6f} post_l2_margin mean={post_l2_stats[0]:+.6f} | "
            f"token_valid pre={int(valid_pre.sum())}/{b} post={int(valid_post.sum())}/{b}"
        )

    all_pre_t = torch.cat(all_pre, dim=0)
    all_post_t = torch.cat(all_post, dim=0)
    all_pre_l2_t = torch.cat(all_pre_l2, dim=0)
    all_post_l2_t = torch.cat(all_post_l2, dim=0)
    all_pre_ap_t = torch.cat(all_pre_ap, dim=0)
    all_pre_an_t = torch.cat(all_pre_an, dim=0)
    all_post_ap_t = torch.cat(all_post_ap, dim=0)
    all_post_an_t = torch.cat(all_post_an, dim=0)
    pre_all = mean_std_min_max(all_pre_t)
    post_all = mean_std_min_max(all_post_t)

    print("\n=== Overall ===")
    print(
        f"PRE  p-n mean={pre_all[0]:+.6f} std={pre_all[1]:.6f} min={pre_all[2]:+.6f} max={pre_all[3]:+.6f} "
        f"pos_rate={(all_pre_t > 0).float().mean().item() * 100:.2f}%"
    )
    print(
        f"POST p-n mean={post_all[0]:+.6f} std={post_all[1]:.6f} min={post_all[2]:+.6f} max={post_all[3]:+.6f} "
        f"pos_rate={(all_post_t > 0).float().mean().item() * 100:.2f}%"
    )
    print(
        f"PRE  l2_margin mean={all_pre_l2_t.mean().item():+.6f} std={all_pre_l2_t.std().item():.6f} "
        f"min={all_pre_l2_t.min().item():+.6f} max={all_pre_l2_t.max().item():+.6f} "
        f"pos_rate={(all_pre_l2_t > 0).float().mean().item() * 100:.2f}%"
    )
    print(
        f"POST l2_margin mean={all_post_l2_t.mean().item():+.6f} std={all_post_l2_t.std().item():.6f} "
        f"min={all_post_l2_t.min().item():+.6f} max={all_post_l2_t.max().item():+.6f} "
        f"pos_rate={(all_post_l2_t > 0).float().mean().item() * 100:.2f}%"
    )
    print(
        f"PRE  sim(a,p) mean={all_pre_ap_t.mean().item():.6f}  sim(a,n) mean={all_pre_an_t.mean().item():.6f}"
    )
    print(
        f"POST sim(a,p) mean={all_post_ap_t.mean().item():.6f}  sim(a,n) mean={all_post_an_t.mean().item():.6f}"
    )

    if all_pre_tok and all_post_tok:
        pre_tok_all = torch.cat(all_pre_tok, dim=0)
        post_tok_all = torch.cat(all_post_tok, dim=0)
        pre_tok_stats = mean_std_min_max(pre_tok_all)
        post_tok_stats = mean_std_min_max(post_tok_all)
        print("\n=== Token-level p-n (target token only) ===")
        print(
            f"PRE  token_p-n mean={pre_tok_stats[0]:+.6f} std={pre_tok_stats[1]:.6f} "
            f"min={pre_tok_stats[2]:+.6f} max={pre_tok_stats[3]:+.6f} "
            f"pos_rate={(pre_tok_all > 0).float().mean().item() * 100:.2f}%"
        )
        print(
            f"POST token_p-n mean={post_tok_stats[0]:+.6f} std={post_tok_stats[1]:.6f} "
            f"min={post_tok_stats[2]:+.6f} max={post_tok_stats[3]:+.6f} "
            f"pos_rate={(post_tok_all > 0).float().mean().item() * 100:.2f}%"
        )

    # Layer-wise similarity summary (including cap_embedder)
    print("\n=== Layer-wise sentence similarity (overall) ===")
    order = ["text_encoder_pre_refiner", "cap_embedder"] + sorted(
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

    print("\nfirst 10 PRE p-n :", [round(float(x), 6) for x in all_pre_t[:10]])
    print("first 10 POST p-n:", [round(float(x), 6) for x in all_post_t[:10]])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, default="ckpts/Z-Image-Turbo")
    p.add_argument("--trials", type=int, default=10, help="How many random batches to test")
    p.add_argument("--batch_size", type=int, default=32, help="Triplets per trial")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())

