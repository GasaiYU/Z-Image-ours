import argparse
import os
import random
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA

# Add src to path so we can import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from utils import ensure_model_weights, load_from_local_dir

# ---------------------------------------------------------------------------
# Quantity word bank
# ---------------------------------------------------------------------------
NUMERIC_QUANTITIES = [
    "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
]
VAGUE_QUANTITIES = [
    "several", "many", "few", "some", "multiple",
    "numerous", "a couple of", "a dozen",
]
ALL_QUANTITIES = NUMERIC_QUANTITIES + VAGUE_QUANTITIES


# ---------------------------------------------------------------------------
# Embedding extraction (unchanged from original)
# ---------------------------------------------------------------------------
def get_prompt_embeds(prompt, text_encoder, tokenizer, device, max_sequence_length=256):
    """Extract token embeddings for a given prompt using the Z-Image pipeline logic."""
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    text_inputs = tokenizer(
        [formatted_prompt],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device).bool()

    with torch.no_grad():
        prompt_embeds = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

    # In Z-Image, the DiT backbone's cap_embedder applies an RMSNorm to the text features
    # right after receiving them. We instantiate an equivalent RMSNorm here to apply it.
    hidden_size = prompt_embeds.shape[-1]
    
    class SimpleRMSNorm(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-5):
            super().__init__()
            self.eps = eps
            self.weight = torch.nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return output * self.weight
            
    # Apply the norm to match what the DiT actually sees
    norm_layer = SimpleRMSNorm(hidden_size).to(device, dtype=prompt_embeds.dtype)
    prompt_embeds = norm_layer(prompt_embeds)

    valid_embeds = prompt_embeds[0][prompt_masks[0]]
    valid_input_ids = text_input_ids[0][prompt_masks[0]]

    tokens = [tokenizer.decode([tid]) for tid in valid_input_ids]

    content_start_idx = 0
    content_end_idx = len(tokens)
    for i, token in enumerate(tokens):
        if 'user' in token:
            content_start_idx = i + 1
        elif '<|im_end|>' in token and i > content_start_idx:
            content_end_idx = i
            break

    if content_start_idx < content_end_idx and content_start_idx > 0:
        valid_embeds = valid_embeds[content_start_idx:content_end_idx]
        valid_input_ids = valid_input_ids[content_start_idx:content_end_idx]
        tokens = tokens[content_start_idx:content_end_idx]

    return valid_embeds.detach().cpu().float().numpy(), tokens, valid_input_ids


# ---------------------------------------------------------------------------
# Quantity swap utilities
# ---------------------------------------------------------------------------
def find_quantities_in_prompt(prompt, quantity_bank=None):
    """
    Find all quantity words present in the prompt (case-insensitive, whole word).
    Returns a list of matched quantity strings in the order they appear.
    """
    if quantity_bank is None:
        quantity_bank = ALL_QUANTITIES

    # Sort by length descending so multi-word entries (e.g. "a couple of") match first
    sorted_bank = sorted(quantity_bank, key=len, reverse=True)
    found = []
    text_lower = prompt.lower()
    for qty in sorted_bank:
        pattern = r'(?<!\w)' + re.escape(qty.lower()) + r'(?!\w)'
        if re.search(pattern, text_lower):
            found.append(qty)
    return found


def replace_quantity_in_prompt(prompt, original_qty, replacement):
    """Replace the first occurrence of original_qty (whole word) with replacement."""
    pattern = r'(?<!\w)' + re.escape(original_qty) + r'(?!\w)'
    return re.sub(pattern, replacement, prompt, count=1, flags=re.IGNORECASE)


def generate_quantity_variants(prompt, quantity_bank=None, num_variants=8, seed=42):
    """
    Detect quantity words in `prompt` and create `num_variants` new prompts by
    replacing each detected quantity with a randomly chosen alternative.

    Returns a list of (label, variant_prompt) tuples.
    The first entry is always the original: ("original", prompt).
    """
    if quantity_bank is None:
        quantity_bank = ALL_QUANTITIES

    rng = random.Random(seed)
    found = find_quantities_in_prompt(prompt, quantity_bank)

    if not found:
        print(f"  [WARNING] No quantity words found in: '{prompt}'")
        return [("original", prompt)]

    print(f"  Detected quantity words: {found}")

    variants = [("original", prompt)]
    bank_lower = {q.lower() for q in quantity_bank}

    # For reproducibility and variety, use the full bank minus found words as candidate pool
    found_lower = {f.lower() for f in found}
    candidates = [q for q in quantity_bank if q.lower() not in found_lower]

    for _ in range(num_variants):
        variant = prompt
        label_parts = []
        for orig_qty in found:
            replacement = rng.choice(candidates)
            variant = replace_quantity_in_prompt(variant, orig_qty, replacement)
            label_parts.append(f"{orig_qty}→{replacement}")
        variants.append((" | ".join(label_parts), variant))

    return variants


def mean_pool(embeds):
    """Mean-pool token embeddings into a single vector."""
    return embeds.mean(axis=0)


def cosine_similarity(a, b):
    """Cosine similarity between two 1-D vectors."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))


def find_quantity_token_indices(tokens, qty_word):
    """
    Find the token indices that correspond to `qty_word` in the token sequence.
    Handles subword tokenization: looks for a contiguous run of tokens whose
    concatenated text matches the quantity word.
    Returns the list of matching indices, or [] if not found.
    """
    qty_lower = qty_word.strip().lower()
    qty_chars = qty_lower.replace(' ', '')

    for start in range(len(tokens)):
        accumulated = ''
        for end in range(start, min(start + 4, len(tokens))):
            accumulated += tokens[end].strip().lower()
            if accumulated.replace(' ', '') == qty_chars:
                return list(range(start, end + 1))
            if len(accumulated) > len(qty_chars) + 2:
                break
    return []


# ---------------------------------------------------------------------------
# Quantity swap visualizations
# ---------------------------------------------------------------------------
def visualize_quantity_swap(
    variants_data,   # list of (label, prompt_text, embeds_array, tokens_list)
    found_quantities,
    out_dir="quantity_swap_visualizations",
):
    """
    Produce three figures:
      1. Prompt-level cosine similarity heatmap (mean-pooled embeddings).
      2. Bar chart: cosine similarity of each variant's quantity-token embedding
         vs. the original's quantity-token embedding.
      3. PCA projection of mean-pooled embeddings across all variants.
    """
    os.makedirs(out_dir, exist_ok=True)

    labels = [d[0] for d in variants_data]
    all_embeds = [d[2] for d in variants_data]   # list of (T, D) arrays
    all_tokens = [d[3] for d in variants_data]   # list of token lists

    mean_vecs = np.stack([mean_pool(e) for e in all_embeds])  # (N, D)
    n = len(labels)

    # ------------------------------------------------------------------
    # 1. Prompt-level cosine similarity heatmap
    # ------------------------------------------------------------------
    norms = mean_vecs / (np.linalg.norm(mean_vecs, axis=1, keepdims=True) + 1e-8)
    sim_matrix = norms @ norms.T  # (N, N)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.9), max(7, n * 0.8)))
    short_labels = [lb[:40] for lb in labels]   # truncate for readability
    mask_diag = np.eye(n, dtype=bool)
    vmin = sim_matrix[~mask_diag].min() - 0.01 if n > 1 else 0.9
    sns.heatmap(
        sim_matrix, annot=True, fmt=".3f", cmap="coolwarm",
        xticklabels=short_labels, yticklabels=short_labels,
        vmin=vmin, vmax=1.0, ax=ax,
        linewidths=0.5, linecolor='gray',
    )
    ax.set_title("Prompt-Level Cosine Similarity\n(mean-pooled embeddings, quantity words swapped)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    # Highlight the original row/col
    ax.add_patch(plt.Rectangle((0, 0), n, 1, fill=False, edgecolor='lime', lw=3))
    ax.add_patch(plt.Rectangle((0, 0), 1, n, fill=False, edgecolor='lime', lw=3))
    plt.tight_layout()
    out_path = os.path.join(out_dir, "prompt_similarity_heatmap.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

    # ------------------------------------------------------------------
    # 2. Bar chart: quantity-token embedding similarity to original
    # ------------------------------------------------------------------
    orig_tokens = all_tokens[0]
    orig_embeds = all_embeds[0]

    # Collect (original) quantity-token mean embedding
    qty_token_sims = []
    qty_labels_found = []
    any_found = False

    for qty in found_quantities:
        orig_idxs = find_quantity_token_indices(orig_tokens, qty)
        if not orig_idxs:
            continue
        orig_qty_vec = orig_embeds[orig_idxs].mean(axis=0)

        sims_for_qty = []
        bar_labels = []
        for variant_label, _, v_embeds, v_tokens in variants_data:
            # Determine the replaced quantity word in this variant
            # (grab whatever replaced `qty` by scanning the label string)
            replacement_word = qty  # fallback = same
            for part in variant_label.split(' | '):
                if f'{qty}→' in part:
                    replacement_word = part.split('→')[1]
                    break
            v_idxs = find_quantity_token_indices(v_tokens, replacement_word)
            if not v_idxs:
                # fall back: find original word in case it wasn't replaced
                v_idxs = find_quantity_token_indices(v_tokens, qty)
            if v_idxs:
                v_qty_vec = v_embeds[v_idxs].mean(axis=0)
                sims_for_qty.append(cosine_similarity(orig_qty_vec, v_qty_vec))
            else:
                sims_for_qty.append(0.0)
            bar_labels.append(variant_label[:35])

        qty_token_sims.append((qty, sims_for_qty, bar_labels))
        any_found = True

    if any_found:
        num_plots = len(qty_token_sims)
        fig, axes = plt.subplots(num_plots, 1, figsize=(max(10, n * 0.8), 5 * num_plots), squeeze=False)
        for plot_i, (qty, sims, bar_labels) in enumerate(qty_token_sims):
            ax = axes[plot_i][0]
            colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(sims))]
            bars = ax.bar(range(len(sims)), sims, color=colors, edgecolor='black', linewidth=0.5)
            ax.set_xticks(range(len(bar_labels)))
            ax.set_xticklabels(bar_labels, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel("Cosine Similarity to Original")
            ax.set_title(f"Quantity Token '{qty}' Embedding Similarity vs. Original")
            ax.set_ylim(max(0, min(sims) - 0.05), 1.05)
            ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(sims[0], color='green', linestyle=':', alpha=0.7, label='original self-sim')
            for bar, sim_val in zip(bars, sims):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                        f"{sim_val:.3f}", ha='center', va='bottom', fontsize=7)
            ax.legend(fontsize=8)

        plt.tight_layout()
        out_path = os.path.join(out_dir, "quantity_token_sim_bar.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved: {out_path}")
    else:
        print("  [WARNING] Could not locate quantity tokens in token sequences; skipping bar chart.")

    # ------------------------------------------------------------------
    # 3. PCA of mean-pooled embeddings
    # ------------------------------------------------------------------
    if n >= 3:
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(mean_vecs)

        fig, ax = plt.subplots(figsize=(11, 9))
        cmap = plt.get_cmap('tab10')

        for i, (label, _, _, _) in enumerate(variants_data):
            color = cmap(0) if i == 0 else cmap((i % 9) + 1)
            marker = '*' if i == 0 else 'o'
            size = 200 if i == 0 else 80
            ax.scatter(coords_2d[i, 0], coords_2d[i, 1], c=[color],
                       marker=marker, s=size, zorder=3, edgecolors='black', linewidths=0.5)
            ax.annotate(label[:35], (coords_2d[i, 0], coords_2d[i, 1]),
                        fontsize=8, xytext=(5, 5), textcoords='offset points',
                        color='darkred' if i == 0 else 'black',
                        fontweight='bold' if i == 0 else 'normal')

        ax.set_title("PCA of Mean-Pooled Prompt Embeddings (Quantity Swap Variants)")
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(out_dir, "pca_quantity_variants.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved: {out_path}")

    print(f"\nAll quantity-swap visualizations saved to: {out_dir}/")


# ---------------------------------------------------------------------------
# Original single-prompt visualizations (unchanged)
# ---------------------------------------------------------------------------
def visualize_embeddings(embeds, tokens, out_dir="output_visualizations",
                         highlight_words=None, only_show_highlighted=False,
                         phrase_to_extract=None):
    """Generate and save various visualizations for the embeddings."""
    os.makedirs(out_dir, exist_ok=True)

    if highlight_words is None:
        highlight_words = []

    if phrase_to_extract:
        phrase_words = phrase_to_extract.lower().split()
        start_idx = -1
        end_idx = -1
        for i in range(len(tokens) - len(phrase_words) + 1):
            match = True
            for j, word in enumerate(phrase_words):
                clean_token = tokens[i + j].strip().replace('\n', '\\n').lower()
                if word not in clean_token and clean_token not in word:
                    match = False
                    break
            if match:
                start_idx = i
                end_idx = i + len(phrase_words)
                break
        if start_idx != -1:
            print(f"Found phrase '{phrase_to_extract}' at token indices {start_idx} to {end_idx-1}")
            embeds = embeds[start_idx:end_idx]
            tokens = tokens[start_idx:end_idx]
        else:
            print(f"Warning: Could not find exact phrase '{phrase_to_extract}' in tokens. Showing all tokens.")

    elif only_show_highlighted and highlight_words:
        filtered_embeds = []
        filtered_tokens = []
        for i, token in enumerate(tokens):
            clean_token = token.strip().replace('\n', '\\n')
            if any(hw.lower() in clean_token.lower() for hw in highlight_words):
                filtered_embeds.append(embeds[i])
                filtered_tokens.append(token)
        if not filtered_tokens:
            print("Warning: No highlighted words found in the prompt. Showing all tokens.")
        else:
            embeds = np.array(filtered_embeds)
            tokens = filtered_tokens

    labels = []
    is_highlighted = []
    for token in tokens:
        clean_token = token.strip().replace('\n', '\\n')
        if not clean_token:
            clean_token = "[SPACE]"
        highlight = any(hw.lower() in clean_token.lower() for hw in highlight_words)
        is_highlighted.append(highlight)
        labels.append(f"*{clean_token}*" if highlight else clean_token)

    # 1. Heatmap of the embeddings
    plt.figure(figsize=(12, max(6, len(tokens) * 0.4)))
    dim_step = max(1, embeds.shape[1] // 100)
    clean_embeds = embeds[:, 1:]
    ax = sns.heatmap(clean_embeds[:, ::dim_step], cmap="viridis", yticklabels=labels)
    for i, tick_label in enumerate(ax.get_yticklabels()):
        if is_highlighted[i]:
            tick_label.set_color('red')
            tick_label.set_fontweight('bold')
    plt.title("Token Embeddings Heatmap (Subsampled, Dim 0 Removed)")
    plt.xlabel("Hidden Dimensions")
    plt.ylabel("Tokens")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "embeds_heatmap.png"))
    plt.close()

    # 2. Token-to-Token Cosine Similarity
    norms = np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8
    norm_embeds = embeds / norms
    similarity_matrix = np.dot(norm_embeds, norm_embeds.T)
    plt.figure(figsize=(max(8, len(tokens) * 0.5), max(8, len(tokens) * 0.5)))
    ax = sns.heatmap(similarity_matrix, cmap="coolwarm", xticklabels=labels, yticklabels=labels,
                     annot=len(tokens) <= 15, fmt=".2f")
    for i, tick_label in enumerate(ax.get_xticklabels()):
        if is_highlighted[i]:
            tick_label.set_color('red')
            tick_label.set_fontweight('bold')
    for i, tick_label in enumerate(ax.get_yticklabels()):
        if is_highlighted[i]:
            tick_label.set_color('red')
            tick_label.set_fontweight('bold')
    plt.title("Token-to-Token Cosine Similarity")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "token_similarity.png"))
    plt.close()

    # 3. PCA 2D Projection
    if len(tokens) > 2:
        pca = PCA(n_components=2)
        embeds_2d = pca.fit_transform(embeds)
        plt.figure(figsize=(12, 10))
        normal_idx = [i for i, h in enumerate(is_highlighted) if not h]
        high_idx = [i for i, h in enumerate(is_highlighted) if h]
        if normal_idx:
            plt.scatter(embeds_2d[normal_idx, 0], embeds_2d[normal_idx, 1],
                        alpha=0.6, s=100, c='blue', label='Normal Tokens')
        if high_idx:
            plt.scatter(embeds_2d[high_idx, 0], embeds_2d[high_idx, 1],
                        alpha=0.9, s=150, c='red', marker='*', label='Attribute Tokens')
            plt.legend()
        for i, label in enumerate(labels):
            color = 'red' if is_highlighted[i] else 'black'
            weight = 'bold' if is_highlighted[i] else 'normal'
            plt.annotate(label, (embeds_2d[i, 0], embeds_2d[i, 1]),
                         fontsize=12 if is_highlighted[i] else 9,
                         color=color, fontweight=weight, alpha=0.8,
                         xytext=(5, 5), textcoords='offset points')
        plt.title("PCA Projection of Token Embeddings")
        plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "pca_projection.png"))
        plt.close()

    print(f"Visualizations successfully saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize input prompt embeddings for Z-Image")
    parser.add_argument("--prompt", type=str,
                        default="Two cats sitting on a red couch in the living room",
                        help="Prompt to visualize (or base prompt for quantity swap mode)")
    parser.add_argument("--out_dir", type=str, default="embed_visualizations",
                        help="Output directory for plots")
    parser.add_argument("--max_seq_len", type=int, default=256,
                        help="Max sequence length for the tokenizer")

    # Original single-prompt visualization args
    parser.add_argument("--highlight_words", type=str, nargs='+', default=[],
                        help="Words to highlight in the visualizations")
    parser.add_argument("--only_show_highlighted", action="store_true",
                        help="Only show highlighted words in visualizations")
    parser.add_argument("--extract_phrase", type=str, default=None,
                        help="Extract and visualize only a specific contiguous phrase")

    # Quantity swap mode
    parser.add_argument("--quantity_swap", action="store_true",
                        help="Enable quantity-swap mode: randomly replace quantity words in the "
                             "prompt and compare resulting embeddings")
    parser.add_argument("--num_variants", type=int, default=8,
                        help="Number of quantity-swap variants to generate (default: 8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for variant generation (default: 42)")
    parser.add_argument("--quantity_bank", type=str, nargs='+', default=None,
                        help="Custom list of quantity words to use as the swap bank. "
                             "Defaults to the built-in numeric + vague quantity bank.")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        device = "mps"
    print(f"Using device: {device}")

    model_path = ensure_model_weights("ckpts/Z-Image-Turbo", verify=False)
    print("Loading models (this might take a moment)...")
    components = load_from_local_dir(model_path, device=device, dtype=torch.bfloat16)
    text_encoder = components["text_encoder"]
    tokenizer = components["tokenizer"]

    # ------------------------------------------------------------------
    # Quantity swap mode
    # ------------------------------------------------------------------
    if args.quantity_swap:
        qty_bank = args.quantity_bank if args.quantity_bank else ALL_QUANTITIES
        print(f"\n[Quantity Swap Mode]")
        print(f"Base prompt : '{args.prompt}'")
        print(f"Quantity bank ({len(qty_bank)} words): {qty_bank}")
        print(f"Generating {args.num_variants} variants (seed={args.seed})...\n")

        found_qtys = find_quantities_in_prompt(args.prompt, qty_bank)
        variants = generate_quantity_variants(
            args.prompt, qty_bank, num_variants=args.num_variants, seed=args.seed
        )

        print(f"Generated {len(variants)} prompts (1 original + {len(variants)-1} variants):\n")
        variants_data = []
        for i, (label, variant_prompt) in enumerate(variants):
            tag = "ORIGINAL" if i == 0 else f"variant {i}"
            print(f"  [{tag}] {label}")
            print(f"           -> '{variant_prompt}'")
            embeds, tokens, _ = get_prompt_embeds(
                variant_prompt, text_encoder, tokenizer, device, args.max_seq_len
            )
            variants_data.append((label, variant_prompt, embeds, tokens))

        print(f"\nGenerating visualizations...")
        visualize_quantity_swap(variants_data, found_qtys, out_dir=args.out_dir)

    # ------------------------------------------------------------------
    # Original single-prompt visualization mode
    # ------------------------------------------------------------------
    else:
        print(f"\nEncoding prompt: '{args.prompt}'")
        embeds, tokens, valid_input_ids = get_prompt_embeds(
            args.prompt, text_encoder, tokenizer, device, args.max_seq_len
        )
        print(f"Extracted embeddings shape: {embeds.shape}")
        print(f"Number of valid tokens (excluding padding and special tokens): {len(tokens)}")
        print("\nToken sequence:")
        for i, token in enumerate(tokens):
            clean_token = token.replace('\n', '\\n')
            print(f"  [{i:3d}] ID: {valid_input_ids[i].item():5} | Token: '{clean_token}'")
        print("\nGenerating visualizations...")
        visualize_embeddings(embeds, tokens, args.out_dir,
                             args.highlight_words, args.only_show_highlighted,
                             args.extract_phrase)


if __name__ == "__main__":
    main()
