"""Generate GenEval images using a uniform average of all LLM hidden layers.

This is a simple *training-free* baseline:
  - hidden_states[0]      : token embedding (skipped)
  - hidden_states[1..-2]  : all transformer layers  (averaged)
  - hidden_states[-1]     : final layer (not used by the DiT pipeline)

The averaged embedding replaces hidden_states[-2] (the layer the DiT reads).
This tests whether spreading attention uniformly across layers, rather than
concentrating on the deepest layer, improves counting performance.

Usage:
    python benchmarks/geneval/generation/zimage_generate_layer_avg.py \\
        benchmarks/geneval/prompts/evaluation_metadata.jsonl \\
        --outdir outputs/geneval_layer_avg_seed42 \\
        --tags counting \\
        --seed 42

    # average only a contiguous range of layers (0-indexed from layer 1):
    python ... --layer_start 0 --layer_end 16   # shallow half
    python ... --layer_start 16                  # deep half
"""

import argparse
import json
import os
import sys

import torch
import numpy as np
from PIL import Image
from tqdm import trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from pytorch_lightning import seed_everything

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from utils import ensure_model_weights, load_from_local_dir, set_attention_backend
from zimage import generate

torch.set_grad_enabled(False)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="GenEval generation: uniform average of all LLM layers"
    )
    parser.add_argument(
        "metadata_file", type=str,
        help="JSONL file with one metadata record per line",
    )
    parser.add_argument("--outdir",      type=str,   default="outputs")
    parser.add_argument("--n_samples",   type=int,   default=4)
    parser.add_argument("--steps",       type=int,   default=8)
    parser.add_argument("--H",           type=int,   default=1024)
    parser.add_argument("--W",           type=int,   default=1024)
    parser.add_argument("--scale",       type=float, default=0.0)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--batch_size",  type=int,   default=1)
    parser.add_argument("--skip_grid",   action="store_true")
    parser.add_argument(
        "--tags", type=str, nargs="+", default=None,
        help="Only process prompts whose tag is in this list (e.g. counting)",
    )
    parser.add_argument(
        "--max_sequence_length", type=int, default=512,
    )
    parser.add_argument(
        "--model_dir", type=str, default="ckpts/Z-Image-Turbo",
    )
    # Layer range to average (indices into the transformer layers, i.e.
    # hidden_states[1+layer_start : 1+layer_end]).
    # Defaults: layer_start=0, layer_end=-1 → average ALL transformer layers.
    parser.add_argument(
        "--layer_start", type=int, default=0,
        help="First transformer layer index to include (0 = shallowest, i.e. hidden_states[1])",
    )
    parser.add_argument(
        "--layer_end", type=int, default=-1,
        help="Last transformer layer index (exclusive, -1 = all layers up to hidden_states[-2])",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core: build layer-averaged prompt embeddings
# ---------------------------------------------------------------------------
def build_avg_embeds(
    prompts,
    text_encoder,
    tokenizer,
    device,
    max_sequence_length: int,
    layer_start: int = 0,
    layer_end: int = -1,
):
    """
    Run the text encoder, then uniformly average the chosen transformer layers.

    Hidden-state indexing:
        hidden_states[0]        : token embedding  (excluded)
        hidden_states[1 .. N-1] : transformer layers 0 .. N-2  (N-1 total)
        hidden_states[N]        : final layer (excluded, pipeline uses hs[-2]=hs[N-1])

    We average hidden_states[1+layer_start : 1+layer_end_incl+1].
    The result replaces hidden_states[-2].
    """
    messages_batch = [[{"role": "user", "content": p}] for p in prompts]
    formatted = [
        tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        for m in messages_batch
    ]

    text_inputs = tokenizer(
        formatted,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)

    with torch.no_grad():
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        all_hs = outputs.hidden_states   # tuple, length = num_layers + 1

    # Transformer layers available for averaging: hidden_states[1 .. -2] inclusive
    # i.e. hidden_states[1], ..., hidden_states[num_layers-1]
    n_transformer = len(all_hs) - 2   # excludes embedding (0) and final (last)

    if layer_end < 0 or layer_end > n_transformer:
        layer_end = n_transformer

    layer_start = max(0, min(layer_start, n_transformer - 1))
    layer_end   = max(layer_start + 1, layer_end)

    # Slice: hidden_states[1+layer_start .. 1+layer_end]
    selected = all_hs[1 + layer_start : 1 + layer_end]   # list of [B, S, D]

    stacked = torch.stack([h.float() for h in selected], dim=0)  # [L, B, S, D]
    avg_embeds = stacked.mean(dim=0).to(all_hs[-2].dtype)         # [B, S, D]

    n_total_hs = len(all_hs)
    print(f"  [LayerAvg] averaging layers {layer_start}..{layer_end-1} "
          f"({layer_end - layer_start}/{n_transformer} transformer layers)", flush=True)

    return avg_embeds, input_ids, attention_mask, n_total_hs


# ---------------------------------------------------------------------------
# Generation with patched text encoder
# ---------------------------------------------------------------------------
def generate_with_avg(components, prompts, opt, device, sample_offset: int = 0):
    text_encoder = components["text_encoder"]

    avg_embeds, expected_ids, _, n_total_hs = build_avg_embeds(
        prompts,
        text_encoder,
        components["tokenizer"],
        device,
        opt.max_sequence_length,
        layer_start=opt.layer_start,
        layer_end=opt.layer_end,
    )

    original_forward = text_encoder.forward

    def patched_forward(input_ids, attention_mask=None, **kwargs):
        class _Out:
            pass

        if input_ids.shape == expected_ids.shape and torch.equal(input_ids, expected_ids):
            out = _Out()
            mock_hs = [None] * n_total_hs
            mock_hs[-2] = avg_embeds
            out.hidden_states = mock_hs
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
            generator=torch.Generator(device).manual_seed(opt.seed + sample_offset),
            max_sequence_length=opt.max_sequence_length,
        )
    finally:
        text_encoder.forward = original_forward

    return images


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(opt):
    with open(opt.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    if opt.tags is not None:
        metadatas = [m for m in metadatas if m.get("tag") in opt.tags]
        print(f"[Filter] {len(metadatas)} prompts match tags: {opt.tags}")

    # ---- Device ----
    if torch.cuda.is_available():
        device = "cuda"
    else:
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
        except (ImportError, RuntimeError):
            device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[Device] {device}")

    # ---- Load Z-Image ----
    model_path = ensure_model_weights(opt.model_dir, verify=False)
    components = load_from_local_dir(model_path, device=device, dtype=torch.bfloat16, compile=False)
    attn_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")
    set_attention_backend(attn_backend)

    layer_tag = f"l{opt.layer_start}-{opt.layer_end}"
    print(f"[LayerAvg] layer range: [{opt.layer_start}, {opt.layer_end})")

    # ---- Generation loop ----
    os.makedirs(opt.outdir, exist_ok=True)

    for index, metadata in enumerate(metadatas):
        seed_everything(opt.seed)

        outpath = os.path.join(opt.outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        prompt     = metadata["prompt"]
        batch_size = opt.batch_size
        print(f"[{index + 1:>3}/{len(metadatas)}] '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)

        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        sample_count = 0
        all_samples  = []

        for _ in trange(
            (opt.n_samples + batch_size - 1) // batch_size,
            desc="Sampling",
            leave=False,
        ):
            current_bs = min(batch_size, opt.n_samples - sample_count)
            prompts    = [prompt] * current_bs

            images = generate_with_avg(components, prompts, opt, device, sample_count)

            for img in images:
                img.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                sample_count += 1
                if not opt.skip_grid:
                    all_samples.append(ToTensor()(img))

        if not opt.skip_grid and all_samples:
            grid = torch.stack(all_samples)
            grid = make_grid(grid, nrow=batch_size)
            grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
            Image.fromarray(grid.astype(np.uint8)).save(
                os.path.join(outpath, "grid.png")
            )
            del grid
        del all_samples

    print("\n[Done] All prompts processed.")


if __name__ == "__main__":
    main(parse_args())
