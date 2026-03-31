import argparse
import os
import sys
import re

import torch
from PIL import Image

# Add src to path so we can import Z-Image modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from utils import ensure_model_weights, load_from_local_dir
from zimage.pipeline import generate

# Word banks to identify tokens that should use shallow features
QUANTITY_BANK = [
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"
]
COLOR_BANK = [
    "red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white",
    "gray", "grey", "cyan", "magenta"
]
TARGET_WORDS = set(QUANTITY_BANK + COLOR_BANK)

def find_target_token_indices(tokens, target_words):
    """Find indices of tokens that match any word in the target_words set."""
    indices = []
    for i, token in enumerate(tokens):
        clean_token = token.lower().strip().replace(' ', '')
        # Check if the token is exactly the target word or contains it as a whole word
        for word in target_words:
            if clean_token == word or re.search(r'\b' + re.escape(word) + r'\b', clean_token):
                indices.append(i)
                break
    return indices

def get_mixed_prompt_embeds(prompt, text_encoder, tokenizer, device, shallow_layer_idx=4, max_sequence_length=256):
    """
    Extract token embeddings, replacing specific tokens (colors/quantities) 
    with features from a shallow layer, while keeping the rest from the deep layer.
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
        outputs = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        )
        hidden_states_tuple = outputs.hidden_states

    # Get baseline (deep) features: Layer -2
    deep_embeds = hidden_states_tuple[-2].clone()
    
    # Get shallow features: e.g., Layer 4
    shallow_embeds = hidden_states_tuple[shallow_layer_idx].clone()

    # Find token boundaries to locate the actual words
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

    # Extract just the prompt content tokens
    content_tokens = tokens[content_start_idx:content_end_idx]
    
    # Find which tokens are colors or quantities
    target_indices_in_content = find_target_token_indices(content_tokens, TARGET_WORDS)
    
    # Map back to the full sequence index
    target_indices_in_full = [idx + content_start_idx for idx in target_indices_in_content]
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Found {len(target_indices_in_full)} target tokens to replace with shallow (Layer {shallow_layer_idx}) features:")
    for idx, full_idx in zip(target_indices_in_content, target_indices_in_full):
        print(f"  - '{content_tokens[idx].strip()}' (Index: {full_idx})")

    # --- THE FRANKENSTEIN SPLICING ---
    # We create a new mixed embedding tensor
    mixed_embeds = deep_embeds.clone()
    
    # Replace the specific token features with shallow ones
    # IMPORTANT: We apply RMSNorm to both to ensure scale matching before splicing
    hidden_size = deep_embeds.shape[-1]
    
    class SimpleRMSNorm(torch.nn.Module):
        def __init__(self, dim: int, eps: float = 1e-5):
            super().__init__()
            self.eps = eps
            self.weight = torch.nn.Parameter(torch.ones(dim))
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
            return output * self.weight
            
    norm_layer = SimpleRMSNorm(hidden_size).to(device, dtype=deep_embeds.dtype)
    
    # Normalize both before splicing to ensure they are in the same numerical scale
    normed_deep = norm_layer(deep_embeds)
    normed_shallow = norm_layer(shallow_embeds)
    
    # Start with normed deep features
    final_mixed_embeds = normed_deep.clone()
    
    # Splice in the normed shallow features for specific tokens
    for idx in target_indices_in_full:
        final_mixed_embeds[0, idx, :] = normed_shallow[0, idx, :]
        
    # We also need to return the baseline (just normed deep) for comparison
    return final_mixed_embeds, normed_deep, text_input_ids, prompt_masks

def custom_generate(
    transformer, vae, text_encoder, tokenizer, scheduler, 
    prompt, mixed_embeds, text_input_ids, prompt_masks,
    device, seed=42
):
    """
    A modified version of the pipeline generate function that accepts pre-computed embeds.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Extract valid embeds based on mask
    prompt_embeds_list = [mixed_embeds[0][prompt_masks[0]]]
    
    # Handle negative prompt (empty string)
    neg_messages = [{"role": "user", "content": ""}]
    neg_formatted = tokenizer.apply_chat_template(
        neg_messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    neg_inputs = tokenizer(
        [neg_formatted], padding="max_length", max_length=256, truncation=True, return_tensors="pt"
    )
    neg_input_ids = neg_inputs.input_ids.to(device)
    neg_masks = neg_inputs.attention_mask.to(device).bool()
    
    with torch.no_grad():
        neg_embeds = text_encoder(
            input_ids=neg_input_ids,
            attention_mask=neg_masks,
            output_hidden_states=True,
        ).hidden_states[-2]
        
        # Apply the same norm to negative embeds
        hidden_size = neg_embeds.shape[-1]
        class SimpleRMSNorm(torch.nn.Module):
            def __init__(self, dim: int, eps: float = 1e-5):
                super().__init__()
                self.eps = eps
                self.weight = torch.nn.Parameter(torch.ones(dim))
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                return output * self.weight
        norm_layer = SimpleRMSNorm(hidden_size).to(device, dtype=neg_embeds.dtype)
        neg_embeds = norm_layer(neg_embeds)
        
    negative_prompt_embeds_list = [neg_embeds[0][neg_masks[0]]]

    # Call the rest of the generation logic using a hack:
    # We temporarily mock the text_encoder to return our pre-computed embeds
    # This avoids rewriting the entire 200-line generate function
    class MockTextEncoder:
        def __call__(self, input_ids, attention_mask, **kwargs):
            class Output:
                pass
            out = Output()
            # If it's the positive prompt, return our mixed embeds
            if input_ids.shape == text_input_ids.shape and torch.all(input_ids == text_input_ids):
                out.hidden_states = [None] * 32
                out.hidden_states[-2] = mixed_embeds
            else:
                out.hidden_states = [None] * 32
                out.hidden_states[-2] = neg_embeds
            return out

    # We need to bypass the internal norm in generate if we already normed it
    # But since we can't easily modify the internal generate without copying it,
    # we'll just pass our embeds to the original generate function
    
    # Actually, it's safer to just copy the core latent loop here to ensure it works
    # with our custom embeds.
    
    # For simplicity in this test script, we will just use the standard generate
    # but we monkey-patch the text encoder temporarily
    original_forward = text_encoder.forward
    
    def patched_forward(input_ids, attention_mask, **kwargs):
        class Output:
            pass
        out = Output()
        if input_ids.shape == text_input_ids.shape and torch.all(input_ids == text_input_ids):
            out.hidden_states = [None] * 32
            out.hidden_states[-2] = mixed_embeds
        else:
            # For negative prompt, we just run the original
            real_out = original_forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            out.hidden_states = [None] * 32
            # Apply norm to negative too so scales match
            out.hidden_states[-2] = norm_layer(real_out.hidden_states[-2])
        return out
        
    text_encoder.forward = patched_forward
    
    try:
        image = generate(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            prompt=prompt,
            num_inference_steps=8, # Match Z-Image-Turbo defaults
            guidance_scale=0.0,    # Match Z-Image-Turbo defaults
            generator=generator
        )
    finally:
        # Restore original forward
        text_encoder.forward = original_forward
        
    return image[0]

def main():
    parser = argparse.ArgumentParser(description="Test Training-Free Token-wise Routing")
    parser.add_argument("--prompt", type=str, default="A red apple and a blue cup", help="Prompt to test")
    parser.add_argument("--shallow_layer", type=int, default=4, help="Which shallow layer to use for attributes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--out_dir", type=str, default="routing_test_results", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        device = "mps"
    
    print(f"Using device: {device}")
    
    model_path = ensure_model_weights("ckpts/Z-Image-Turbo", verify=False)
    print("Loading models (this might take a moment)...")
    components = load_from_local_dir(model_path, device=device, dtype=torch.bfloat16)
    
    print(f"\nPreparing embeddings for prompt: '{args.prompt}'")
    mixed_embeds, baseline_embeds, input_ids, masks = get_mixed_prompt_embeds(
        args.prompt, 
        components["text_encoder"], 
        components["tokenizer"], 
        device,
        shallow_layer_idx=args.shallow_layer
    )
    
    print(f"\n1. Generating BASELINE image (All Deep Features)...")
    img_baseline = custom_generate(
        components["transformer"], components["vae"], components["text_encoder"],
        components["tokenizer"], components["scheduler"],
        args.prompt, baseline_embeds, input_ids, masks, device, seed=args.seed
    )
    base_path = os.path.join(args.out_dir, "baseline_deep_only.png")
    img_baseline.save(base_path)
    print(f"Saved baseline to {base_path}")
    
    print(f"\n2. Generating OURS image (Token-wise Mixed Features)...")
    img_ours = custom_generate(
        components["transformer"], components["vae"], components["text_encoder"],
        components["tokenizer"], components["scheduler"],
        args.prompt, mixed_embeds, input_ids, masks, device, seed=args.seed
    )
    ours_path = os.path.join(args.out_dir, f"ours_mixed_layer{args.shallow_layer}.png")
    img_ours.save(ours_path)
    print(f"Saved ours to {ours_path}")
    
    print("\nDone! Please compare the two images to see if attribute leakage/counting is improved.")

if __name__ == "__main__":
    main()
