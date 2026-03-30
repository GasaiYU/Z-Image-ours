import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

# Add src to path so we can import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from utils import ensure_model_weights, load_from_local_dir

def get_all_layers_embeds(prompt, text_encoder, tokenizer, device, max_sequence_length=256):
    """
    Extract token embeddings from ALL layers for a given prompt.
    Returns:
        all_layers_embeds: List of numpy arrays, one for each layer. Shape of each: (num_valid_tokens, hidden_dim)
        tokens: List of string tokens
    """
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
        # output_hidden_states=True returns a tuple of all layers' hidden states
        outputs = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        )
        hidden_states_tuple = outputs.hidden_states

    # In Z-Image, the DiT backbone applies an RMSNorm to the text features
    # right after receiving them (in the cap_embedder).
    # We instantiate an equivalent RMSNorm here to apply to all extracted layers.
    # The cap_feat_dim is typically the hidden size of the text encoder.
    hidden_size = hidden_states_tuple[0].shape[-1]
    
    # Define a simple RMSNorm equivalent to the one in Z-Image
    class SimpleRMSNorm(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-5):
            super().__init__()
            self.eps = eps
            self.weight = torch.nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return output * self.weight
            
    norm_layer = SimpleRMSNorm(hidden_size).to(device, dtype=hidden_states_tuple[0].dtype)

    # Process tokens to find the actual content boundaries
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

    # Extract valid content embeddings for EVERY layer
    all_layers_embeds = []
    for layer_idx, hidden_state in enumerate(hidden_states_tuple):
        # Apply the Z-Image style RMSNorm
        hidden_state = norm_layer(hidden_state)
        
        valid_embeds = hidden_state[0][prompt_masks[0]]
        if content_start_idx < content_end_idx and content_start_idx > 0:
            valid_embeds = valid_embeds[content_start_idx:content_end_idx]
        all_layers_embeds.append(valid_embeds.cpu().float().numpy())
        
    print(f"  [Info] Applied Z-Image cap_embedder RMSNorm to all extracted layers.")

    if content_start_idx < content_end_idx and content_start_idx > 0:
        tokens = tokens[content_start_idx:content_end_idx]

    return all_layers_embeds, tokens

def find_token_index(tokens, target_word):
    """Find the index of a target word in the token list."""
    target_lower = target_word.lower().strip()
    for i, token in enumerate(tokens):
        clean_token = token.lower().strip().replace(' ', '')
        if target_lower in clean_token or clean_token in target_lower:
            return i
    return -1

def cosine_similarity(a, b):
    """Calculate cosine similarity between two 1D vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def analyze_attribute_leakage(all_layers_embeds, tokens, attr_word, target_noun, unrelated_noun, out_dir):
    """
    Analyze how the similarity between an attribute and nouns changes across layers.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    attr_idx = find_token_index(tokens, attr_word)
    target_idx = find_token_index(tokens, target_noun)
    unrelated_idx = find_token_index(tokens, unrelated_noun)

    print(f"\n--- Token Indices ---")
    print(f"Attribute '{attr_word}': index {attr_idx} (Token: '{tokens[attr_idx] if attr_idx != -1 else 'NOT FOUND'}')")
    print(f"Target Noun '{target_noun}': index {target_idx} (Token: '{tokens[target_idx] if target_idx != -1 else 'NOT FOUND'}')")
    print(f"Unrelated Noun '{unrelated_noun}': index {unrelated_idx} (Token: '{tokens[unrelated_idx] if unrelated_idx != -1 else 'NOT FOUND'}')")

    if -1 in [attr_idx, target_idx, unrelated_idx]:
        print("\n[ERROR] Could not find one or more target words in the tokenized prompt. Please check your prompt and target words.")
        print("Available tokens:", [t.strip() for t in tokens])
        return

    num_layers = len(all_layers_embeds)
    sim_attr_target = []
    sim_attr_unrelated = []
    sim_target_unrelated = []

    for layer_idx in range(num_layers):
        layer_embeds = all_layers_embeds[layer_idx]
        
        vec_attr = layer_embeds[attr_idx]
        vec_target = layer_embeds[target_idx]
        vec_unrelated = layer_embeds[unrelated_idx]

        sim_attr_target.append(cosine_similarity(vec_attr, vec_target))
        sim_attr_unrelated.append(cosine_similarity(vec_attr, vec_unrelated))
        sim_target_unrelated.append(cosine_similarity(vec_target, vec_unrelated))

    # Plotting
    plt.figure(figsize=(10, 6))
    layers = range(num_layers)
    
    plt.plot(layers, sim_attr_target, label=f"'{attr_word}' <-> '{target_noun}' (Correct Binding)", color='green', linewidth=2, marker='o', markersize=4)
    plt.plot(layers, sim_attr_unrelated, label=f"'{attr_word}' <-> '{unrelated_noun}' (Attribute Leakage)", color='red', linewidth=2, marker='x', markersize=4)
    # plt.plot(layers, sim_target_unrelated, label=f"'{target_noun}' <-> '{unrelated_noun}'", color='gray', linestyle='--', alpha=0.5)

    plt.title(f"Attribute Leakage Across LLM Layers\nPrompt: \"{''.join(tokens).replace(' ', ' ')}\"")
    plt.xlabel("LLM Layer Depth (0 = Embedding, N = Final Layer)")
    plt.ylabel("Cosine Similarity")
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Highlight the leakage gap
    plt.fill_between(layers, sim_attr_target, sim_attr_unrelated, color='yellow', alpha=0.1, label='Semantic Gap')
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, f"attribute_leakage_{attr_word}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    
    print(f"\nVisualization saved to {save_path}")
    
    # Print analysis
    print("\n--- Analysis Results ---")
    print(f"Layer 0 (Shallow) Similarity Gap (Correct - Leakage): {sim_attr_target[0] - sim_attr_unrelated[0]:.4f}")
    print(f"Layer {num_layers-1} (Deep) Similarity Gap (Correct - Leakage): {sim_attr_target[-1] - sim_attr_unrelated[-1]:.4f}")
    
    if sim_attr_unrelated[-1] > sim_attr_unrelated[0]:
        print(f"\n[CONCLUSION] Attribute Leakage OBSERVED: The similarity between the attribute '{attr_word}' and the unrelated noun '{unrelated_noun}' increased from {sim_attr_unrelated[0]:.4f} in shallow layers to {sim_attr_unrelated[-1]:.4f} in deep layers.")
    else:
        print("\n[CONCLUSION] No significant attribute leakage observed for these specific tokens.")

def main():
    parser = argparse.ArgumentParser(description="Verify Attribute Leakage across LLM layers")
    parser.add_argument("--prompt", type=str, default="A red apple and a blue cup", help="Prompt to analyze")
    parser.add_argument("--attr", type=str, default="red", help="The attribute word (e.g., 'red')")
    parser.add_argument("--target", type=str, default="apple", help="The noun the attribute belongs to (e.g., 'apple')")
    parser.add_argument("--unrelated", type=str, default="cup", help="The unrelated noun (e.g., 'cup')")
    parser.add_argument("--out_dir", type=str, default="leakage_analysis", help="Output directory")
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
    
    print(f"\nEncoding prompt: '{args.prompt}'")
    all_layers_embeds, tokens = get_all_layers_embeds(args.prompt, text_encoder, tokenizer, device)
    
    print(f"Extracted features from {len(all_layers_embeds)} layers.")
    print("Token sequence:", [t.strip() for t in tokens])
    
    analyze_attribute_leakage(all_layers_embeds, tokens, args.attr, args.target, args.unrelated, args.out_dir)

if __name__ == "__main__":
    main()
