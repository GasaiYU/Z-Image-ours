import argparse
import os
import sys
import re

import torch
from PIL import Image, ImageDraw, ImageFont
from pytorch_lightning import seed_everything

# Add src to path so we can import Z-Image modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from utils import ensure_model_weights, load_from_local_dir, set_attention_backend
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

def get_mixed_prompt_embeds(prompt, text_encoder, tokenizer, device, shallow_layer_idx=4, max_sequence_length=512, alpha=0.3):
    """
    Extract token embeddings, blending specific tokens (colors/quantities) 
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
    
    # Soft Blending: Combine deep and shallow features for specific tokens
    # alpha controls how much shallow feature to inject (e.g., 0.3 means 30% shallow, 70% deep)
    print(f"  [Debug] Applying Soft Blending with alpha={alpha}")
    for idx in target_indices_in_full:
        mixed_embeds[0, idx, :] = (1 - alpha) * deep_embeds[0, idx, :] + alpha * shallow_embeds[0, idx, :]
        
    return mixed_embeds, deep_embeds, text_input_ids, prompt_masks

def custom_generate(
    transformer, vae, text_encoder, tokenizer, scheduler, 
    prompt, mixed_embeds, device, seed=42
):
    """
    A modified version of the pipeline generate function that accepts pre-computed embeds.
    """
    # Match zimage_generate.py seed logic exactly
    from pytorch_lightning import seed_everything
    seed_everything(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    
    original_forward = text_encoder.forward
    
    def patched_forward(input_ids, attention_mask, **kwargs):
        class Output:
            pass
        out = Output()
        # Unconditionally return our mixed embeds to guarantee it works
        # (Since guidance_scale=0.0, this is only called once for the positive prompt)
        print("  [Debug] Monkey patch hit! Injecting custom embeddings.")
        out.hidden_states = [None] * 32
        out.hidden_states[-2] = mixed_embeds
        return out
        
    text_encoder.forward = patched_forward
    
    try:
        # Match zimage_generate.py arguments exactly
        image = generate(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            prompt=[prompt], # Pass as list to match batched format
            height=1024,     # Explicitly set to 1024
            width=1024,      # Explicitly set to 1024
            num_inference_steps=8, 
            guidance_scale=0.0,    
            generator=generator
        )
    finally:
        # Restore original forward
        text_encoder.forward = original_forward
        
    return image[0]

def main():
    parser = argparse.ArgumentParser(description="Test Training-Free Token-wise Routing")
    parser.add_argument("--prompt", type=str, default="A red apple and a blue cup", help="Prompt to test")
    parser.add_argument("--shallow_layer", type=int, default=12, help="Which shallow layer to use for attributes")
    parser.add_argument("--alpha", type=float, default=0.3, help="Soft blending alpha (0.0=all deep, 1.0=all shallow)")
    parser.add_argument("--num_seeds", type=int, default=4, help="Number of different random seeds to test")
    parser.add_argument("--start_seed", type=int, default=42, help="Starting random seed")
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
    mixed_embeds, baseline_embeds, _, _ = get_mixed_prompt_embeds(
        args.prompt, 
        components["text_encoder"], 
        components["tokenizer"], 
        device,
        shallow_layer_idx=args.shallow_layer,
        alpha=args.alpha
    )
    
    baseline_images = []
    ours_images = []

    for i in range(args.num_seeds):
        current_seed = args.start_seed + i
        print(f"\n=== Testing Seed {current_seed} ({i+1}/{args.num_seeds}) ===")
        
        print(f"1. Generating BASELINE image (All Deep Features)...")
        img_baseline = custom_generate(
            components["transformer"], components["vae"], components["text_encoder"],
            components["tokenizer"], components["scheduler"],
            args.prompt, baseline_embeds, device, seed=current_seed
        )
        baseline_images.append(img_baseline)
        
        print(f"2. Generating OURS image (Token-wise Mixed Features)...")
        img_ours = custom_generate(
            components["transformer"], components["vae"], components["text_encoder"],
            components["tokenizer"], components["scheduler"],
            args.prompt, mixed_embeds, device, seed=current_seed
        )
        ours_images.append(img_ours)

    # Build comparison grid:
    # Row 0: label row
    # Row 1: Baseline images (one per seed)
    # Row 2: Ours images     (one per seed)
    n = len(baseline_images)
    img_w, img_h = baseline_images[0].size
    label_h = 60
    pad = 8
    total_w = n * img_w + (n + 1) * pad
    total_h = 2 * img_h + 3 * label_h + (2 + 1) * pad  # 2 rows of images + 3 label rows (top + per row)

    grid = Image.new("RGB", (total_w, total_h), color=(240, 240, 240))
    draw = ImageDraw.Draw(grid)

    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except Exception:
        font_large = ImageFont.load_default()
        font_small = font_large

    # Top title
    title = f'Prompt: "{args.prompt}"'
    draw.text((pad, pad), title, fill=(30, 30, 30), font=font_large)

    row_labels = ["Baseline (Deep Layer Only)", f"Ours (Layer {args.shallow_layer}, alpha {args.alpha})"]
    for row_idx, (images, label) in enumerate(zip([baseline_images, ours_images], row_labels)):
        row_y_label = label_h + pad + row_idx * (img_h + label_h + pad)
        row_y_img   = row_y_label + label_h

        # Row label
        draw.rectangle([pad, row_y_label, total_w - pad, row_y_label + label_h - 4], fill=(200, 220, 255) if row_idx == 0 else (200, 255, 210))
        draw.text((pad * 2, row_y_label + 10), label, fill=(20, 20, 20), font=font_small)

        for col_idx, img in enumerate(images):
            x = pad + col_idx * (img_w + pad)
            grid.paste(img, (x, row_y_img))
            # Seed label on each image
            seed_label = f"seed={args.start_seed + col_idx}"
            draw.text((x + 6, row_y_img + 6), seed_label, fill=(255, 255, 255), font=font_small)

    grid_path = os.path.join(args.out_dir, f"comparison_layer{args.shallow_layer}_alpha{args.alpha}.png")
    grid.save(grid_path)
    print(f"\nSaved comparison grid to: {grid_path}")
    print("Done!")

if __name__ == "__main__":
    main()
