"""Generate Geneval images with token-wise all-layer decay fusion."""

import argparse
import json
import os
import re
import sys

import torch
import numpy as np
from PIL import Image
from tqdm import trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from pytorch_lightning import seed_everything

# Add root directory to sys.path to import zimage and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from utils import ensure_model_weights, load_from_local_dir, set_attention_backend
from zimage import generate

torch.set_grad_enabled(False)


QUANTITY_BANK = [
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
]
COLOR_BANK = [
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "orange",
    "pink",
    "brown",
    "black",
    "white",
    "gray",
    "grey",
    "cyan",
    "magenta",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("metadata_file", type=str, help="JSONL file containing lines of metadata for each prompt")
    parser.add_argument("--outdir", type=str, nargs="?", default="outputs", help="dir to write results to")
    parser.add_argument("--n_samples", type=int, default=4, help="number of samples")
    parser.add_argument("--steps", type=int, default=8, help="number of inference steps")
    parser.add_argument("--H", type=int, default=1024, help="image height")
    parser.add_argument("--W", type=int, default=1024, help="image width")
    parser.add_argument("--scale", type=float, default=0.0, help="CFG scale")
    parser.add_argument("--negative_prompt", type=str, default=None,
                        help="Negative prompt for CFG (only used when --scale > 1.0)")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--skip_grid", action="store_true", help="skip saving grid")
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="only run prompts with these tags (e.g. counting position color_attr colors)",
    )
    parser.add_argument(
        "--target_type",
        type=str,
        choices=["quantity", "color", "both"],
        default="both",
        help="which token types use decay fusion",
    )
    parser.add_argument(
        "--decay_rate",
        type=float,
        default=0.1,
        help="exponential decay rate for all-layer fusion",
    )
    parser.add_argument(
        "--route_start",
        type=int,
        default=10,
        help="first transformer layer index (inclusive) for decay fusion range",
    )
    parser.add_argument(
        "--route_end",
        type=int,
        default=21,
        help="last transformer layer index (exclusive) for decay fusion range",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=512,
        help="tokenization length used for prompt encoding",
    )
    return parser.parse_args()


def find_target_token_indices(tokens, target_words):
    indices = []
    for i, token in enumerate(tokens):
        clean_token = token.lower().strip().replace(" ", "")
        for word in target_words:
            if clean_token == word or re.search(r"\b" + re.escape(word) + r"\b", clean_token):
                indices.append(i)
                break
    return indices


def get_target_words(target_type):
    if target_type == "quantity":
        return set(QUANTITY_BANK)
    if target_type == "color":
        return set(COLOR_BANK)
    return set(QUANTITY_BANK + COLOR_BANK)


def build_decay_mixed_embeds(prompts, text_encoder, tokenizer, device, decay_rate,
                             max_sequence_length, target_words, route_start=10, route_end=21):
    messages_batch = [[{"role": "user", "content": p}] for p in prompts]
    formatted_prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        for m in messages_batch
    ]

    text_inputs = tokenizer(
        formatted_prompts,
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

    deep_embeds = hidden_states_tuple[-2].clone()
    mixed_embeds = deep_embeds.clone()

    total_hs = len(hidden_states_tuple)
    rs  = max(1, route_start)
    re_ = max(rs + 1, min(route_end, total_hs - 1))
    layers_to_fuse = hidden_states_tuple[rs:re_]
    num_layers = len(layers_to_fuse)
    weights = torch.exp(-decay_rate * torch.arange(num_layers, device=device, dtype=deep_embeds.dtype))
    weights = weights / weights.sum()
    print(f"[Decay] layers [{rs}, {re_})  n={num_layers}  "
          f"w[0]={weights[0]:.4f}  w[-1]={weights[-1]:.4f}")

    for b_idx in range(text_input_ids.shape[0]):
        valid_input_ids = text_input_ids[b_idx][prompt_masks[b_idx]]
        tokens = [tokenizer.decode([tid]) for tid in valid_input_ids]

        content_start_idx = 0
        content_end_idx = len(tokens)
        for i, token in enumerate(tokens):
            if "user" in token:
                content_start_idx = i + 1
            elif "<|im_end|>" in token and i > content_start_idx:
                content_end_idx = i
                break

        content_tokens = tokens[content_start_idx:content_end_idx]
        target_indices_in_content = find_target_token_indices(content_tokens, target_words)
        target_indices_in_full = [idx + content_start_idx for idx in target_indices_in_content]

        for idx in target_indices_in_full:
            token_fused = torch.zeros_like(deep_embeds[b_idx, idx, :])
            for i, layer_embeds in enumerate(layers_to_fuse):
                token_fused += weights[i] * layer_embeds[b_idx, idx, :]
            mixed_embeds[b_idx, idx, :] = token_fused

    return mixed_embeds, text_input_ids, prompt_masks


def generate_with_decay(components, prompts, opt, device, generator, target_words):
    text_encoder = components["text_encoder"]
    mixed_embeds, expected_ids, _ = build_decay_mixed_embeds(
        prompts,
        text_encoder,
        components["tokenizer"],
        device,
        opt.decay_rate,
        opt.max_sequence_length,
        target_words,
        route_start=opt.route_start,
        route_end=opt.route_end,
    )

    original_forward = text_encoder.forward

    def patched_forward(input_ids, attention_mask, **kwargs):
        class Output:
            pass

        out = Output()
        if input_ids.shape == expected_ids.shape and torch.equal(input_ids, expected_ids):
            out.hidden_states = [None] * 32
            out.hidden_states[-2] = mixed_embeds
            return out
        return original_forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    text_encoder.forward = patched_forward
    try:
        images = generate(
            prompt=prompts,
            **components,
            height=opt.H,
            width=opt.W,
            num_inference_steps=opt.steps,
            guidance_scale=opt.scale,
            generator=generator,
            max_sequence_length=opt.max_sequence_length,
        )
    finally:
        text_encoder.forward = original_forward
    return images


def main(opt):
    with open(opt.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    if opt.tags is not None:
        metadatas = [m for m in metadatas if m.get("tag") in opt.tags]
        print(f"Filtered prompts to {len(metadatas)} items matching tags: {opt.tags}")

    if torch.cuda.is_available():
        device = "cuda"
    else:
        try:
            import torch_xla.core.xla_model as xm

            device = xm.xla_device()
        except (ImportError, RuntimeError):
            if torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
    print(f"Chosen device: {device}")

    model_path = ensure_model_weights("ckpts/Z-Image-Turbo", verify=False)
    components = load_from_local_dir(model_path, device=device, dtype=torch.bfloat16, compile=False)
    attn_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")
    set_attention_backend(attn_backend)
    print(f"Chosen attention backend: {attn_backend}")

    target_words = get_target_words(opt.target_type)
    print(f"Using decay fusion. target_type={opt.target_type}, decay_rate={opt.decay_rate}")

    for index, metadata in enumerate(metadatas):
        seed_everything(opt.seed)

        outpath = os.path.join(opt.outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        prompt = metadata["prompt"]
        n_rows = batch_size = opt.batch_size
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        sample_count = 0
        with torch.no_grad():
            all_samples = []
            for _ in trange((opt.n_samples + batch_size - 1) // batch_size, desc="Sampling"):
                current_batch_size = min(batch_size, opt.n_samples - sample_count)
                prompts = [prompt] * current_batch_size
                generator = torch.Generator(device).manual_seed(opt.seed + sample_count)

                images = generate_with_decay(components, prompts, opt, device, generator, target_words)

                for sample in images:
                    sample.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                    sample_count += 1
                    if not opt.skip_grid:
                        all_samples.append(ToTensor()(sample))

            if not opt.skip_grid and len(all_samples) > 0:
                grid = torch.stack(all_samples, 0)
                grid = make_grid(grid, nrow=n_rows)
                grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
                grid = Image.fromarray(grid.astype(np.uint8))
                grid.save(os.path.join(outpath, "grid.png"))
                del grid

    print("Done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
