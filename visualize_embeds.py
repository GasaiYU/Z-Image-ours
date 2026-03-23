import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.decomposition import PCA

# Add src to path so we can import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from utils import ensure_model_weights, load_from_local_dir

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

    # Filter out padding tokens using the attention mask
    valid_embeds = prompt_embeds[0][prompt_masks[0]]
    valid_input_ids = text_input_ids[0][prompt_masks[0]]
    
    # Decode tokens for visualization labels
    tokens = [tokenizer.decode([tid]) for tid in valid_input_ids]
    
    # Filter out special tokens (like <|im_start|>, user, assistant, <|im_end|>)
    # We only want the actual prompt content
    content_start_idx = 0
    content_end_idx = len(tokens)
    
    # Find the actual prompt content between user and <|im_end|>
    for i, token in enumerate(tokens):
        if 'user' in token:
            content_start_idx = i + 1
        elif '<|im_end|>' in token and i > content_start_idx:
            content_end_idx = i
            break
            
    # If we couldn't find the exact markers, just use the whole thing
    # Otherwise, slice the arrays to only include the content
    if content_start_idx < content_end_idx and content_start_idx > 0:
        valid_embeds = valid_embeds[content_start_idx:content_end_idx]
        valid_input_ids = valid_input_ids[content_start_idx:content_end_idx]
        tokens = tokens[content_start_idx:content_end_idx]
    
    return valid_embeds.cpu().float().numpy(), tokens, valid_input_ids

def visualize_embeddings(embeds, tokens, out_dir="output_visualizations", highlight_words=None, only_show_highlighted=False, phrase_to_extract=None):
    """Generate and save various visualizations for the embeddings."""
    os.makedirs(out_dir, exist_ok=True)
    
    if highlight_words is None:
        highlight_words = []
        
    # If phrase_to_extract is provided, find the contiguous sequence of tokens
    if phrase_to_extract:
        phrase_words = phrase_to_extract.lower().split()
        start_idx = -1
        end_idx = -1
        
        # Simple sliding window to find the phrase
        for i in range(len(tokens) - len(phrase_words) + 1):
            match = True
            for j, word in enumerate(phrase_words):
                clean_token = tokens[i+j].strip().replace('\n', '\\n').lower()
                # Check if the token contains the word or vice versa (handling subwords)
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
            
    # If only_show_highlighted is True (and we didn't just extract a phrase), filter the embeds and tokens
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
            
    # Create highlighted labels
    labels = []
    is_highlighted = []
    for token in tokens:
        clean_token = token.strip().replace('\n', '\\n')
        if not clean_token:
            clean_token = "[SPACE]"
            
        highlight = any(hw.lower() in clean_token.lower() for hw in highlight_words)
        is_highlighted.append(highlight)
        if highlight:
            labels.append(f"*{clean_token}*")
        else:
            labels.append(clean_token)
            
    # 1. Heatmap of the embeddings (subsampled if too large)
    plt.figure(figsize=(12, max(6, len(tokens) * 0.4)))
    # Subsample hidden dimensions for better visualization if it's huge (e.g., 4096 -> 100)
    dim_step = max(1, embeds.shape[1] // 100)
    
    # Remove the first dimension (outlier) for better visualization of the rest
    clean_embeds = embeds[:, 1:]
    
    ax = sns.heatmap(clean_embeds[:, ::dim_step], cmap="viridis", yticklabels=labels)
    
    # Color highlighted y-tick labels red
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
    # Add small epsilon to avoid division by zero
    norms = np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-8
    norm_embeds = embeds / norms
    similarity_matrix = np.dot(norm_embeds, norm_embeds.T)
    
    plt.figure(figsize=(max(8, len(tokens) * 0.5), max(8, len(tokens) * 0.5)))
    ax = sns.heatmap(similarity_matrix, cmap="coolwarm", xticklabels=labels, yticklabels=labels, annot=len(tokens) <= 15, fmt=".2f")
    
    # Color highlighted tick labels red
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
        
        # Plot normal points and highlighted points separately for different colors
        normal_idx = [i for i, h in enumerate(is_highlighted) if not h]
        high_idx = [i for i, h in enumerate(is_highlighted) if h]
        
        if normal_idx:
            plt.scatter(embeds_2d[normal_idx, 0], embeds_2d[normal_idx, 1], alpha=0.6, s=100, c='blue', label='Normal Tokens')
        if high_idx:
            plt.scatter(embeds_2d[high_idx, 0], embeds_2d[high_idx, 1], alpha=0.9, s=150, c='red', marker='*', label='Attribute Tokens')
            plt.legend()
        
        for i, label in enumerate(labels):
            color = 'red' if is_highlighted[i] else 'black'
            weight = 'bold' if is_highlighted[i] else 'normal'
            plt.annotate(label, (embeds_2d[i, 0], embeds_2d[i, 1]), 
                         fontsize=12 if is_highlighted[i] else 9, 
                         color=color, fontweight=weight, alpha=0.8, xytext=(5, 5), textcoords='offset points')
            
        plt.title("PCA Projection of Token Embeddings")
        plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "pca_projection.png"))
        plt.close()
        
    print(f"Visualizations successfully saved to {out_dir}/")

def main():
    parser = argparse.ArgumentParser(description="Visualize input prompt embeddings for Z-Image")
    parser.add_argument("--prompt", type=str, default="A beautiful sunset over the mountains, cinematic lighting, 8k resolution", help="Prompt to visualize")
    parser.add_argument("--out_dir", type=str, default="embed_visualizations", help="Output directory for plots")
    parser.add_argument("--max_seq_len", type=int, default=256, help="Max sequence length for the tokenizer")
    parser.add_argument("--highlight_words", type=str, nargs='+', default=[], help="Words to highlight in the visualizations (e.g. red two female Asian)")
    parser.add_argument("--only_show_highlighted", action="store_true", help="If set, only the highlighted words will be shown in the visualizations")
    parser.add_argument("--extract_phrase", type=str, default=None, help="Extract and visualize only a specific contiguous phrase (e.g. 'a yellow umbrella')")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        device = "mps"
    
    print(f"Using device: {device}")
    
    # Load models
    model_path = ensure_model_weights("ckpts/Z-Image-Turbo", verify=False)
    
    print("Loading models (this might take a moment)...")
    # We use load_from_local_dir to ensure we load the exact same way as the pipeline
    components = load_from_local_dir(model_path, device=device, dtype=torch.bfloat16)
    text_encoder = components["text_encoder"]
    tokenizer = components["tokenizer"]
    
    print(f"\nEncoding prompt: '{args.prompt}'")
    embeds, tokens, valid_input_ids = get_prompt_embeds(args.prompt, text_encoder, tokenizer, device, args.max_seq_len)
    
    print(f"Extracted embeddings shape: {embeds.shape}")
    print(f"Number of valid tokens (excluding padding and special tokens): {len(tokens)}")
    
    print("\nToken sequence:")
    for i, token in enumerate(tokens):
        # Clean up newlines for printing
        clean_token = token.replace('\n', '\\n')
        print(f"  [{i:3d}] ID: {valid_input_ids[i].item():5} | Token: '{clean_token}'")
    
    print("\nGenerating visualizations...")
    visualize_embeddings(embeds, tokens, args.out_dir, args.highlight_words, args.only_show_highlighted, args.extract_phrase)

if __name__ == "__main__":
    main()
