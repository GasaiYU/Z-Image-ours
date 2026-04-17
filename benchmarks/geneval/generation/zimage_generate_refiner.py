"""
GenEval generation script for the fine-tuned Z-Image model (context_refiner checkpoint).

Loads a base Z-Image-Turbo model, then hot-swaps the transformer weights from
a checkpoint saved by train_counting_contrastive_diffusion.py, and generates
images in the same format expected by GenEval evaluation.

Usage example:
    python zimage_generate_refiner.py \
        prompts/evaluation_metadata.jsonl \
        --checkpoint checkpoints/counting_text_refiner/transformer_refiner_step1000.pt \
        --outdir outputs_refiner \
        --n_samples 4 \
        --steps 8 \
        --seed 42
"""

import argparse
import json
import os
import sys

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor
from pytorch_lightning import seed_everything

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import AttentionBackend, ensure_model_weights, load_from_local_dir, set_attention_backend
from zimage import generate

torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate GenEval images with a fine-tuned context_refiner checkpoint"
    )
    parser.add_argument(
        "metadata_file",
        type=str,
        help="JSONL file containing lines of metadata for each prompt",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a .pt checkpoint saved by train_counting_contrastive_diffusion.py "
             "(must contain 'transformer_state_dict')",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="ckpts/Z-Image-Turbo",
        help="Base Z-Image-Turbo model directory (same as used during training)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs",
        help="Directory to write results to",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="Number of samples per prompt",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Number of inference steps (default 8 for Z-Image-Turbo)",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=1024,
        help="Image height in pixels",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=1024,
        help="Image width in pixels",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.0,
        help="Unconditional guidance scale (default 0.0 for Z-Image-Turbo)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducible sampling",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of samples generated per forward pass",
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="Skip saving the image grid",
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="Only run prompts with these tags (e.g. counting position color_attr)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Use strict=True when loading checkpoint state_dict (default: True)",
    )
    parser.add_argument(
        "--no_strict",
        dest="strict",
        action="store_false",
        help="Use strict=False when loading checkpoint state_dict",
    )
    return parser.parse_args()


def load_refiner_checkpoint(checkpoint_path: str, transformer, device: str, strict: bool = True):
    """
    Load transformer weights from a training checkpoint.

    The checkpoint format saved by train_counting_contrastive_diffusion.py is:
        {
            "step": int,
            "epoch": int,
            "transformer_state_dict": OrderedDict,
            "args": dict,
        }
    """
    print(f"[Checkpoint] loading: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "transformer_state_dict" not in ckpt:
        raise KeyError(
            f"Key 'transformer_state_dict' not found in checkpoint. "
            f"Available keys: {list(ckpt.keys())}"
        )

    state_dict = ckpt["transformer_state_dict"]
    missing, unexpected = transformer.load_state_dict(state_dict, strict=strict)

    step = ckpt.get("step", "?")
    epoch = ckpt.get("epoch", "?")
    print(f"[Checkpoint] loaded step={step}, epoch={epoch}")
    if missing:
        print(f"[Checkpoint] missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[Checkpoint] unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # Log training args if available
    train_args = ckpt.get("args", {})
    if train_args:
        relevant = {
            k: train_args[k]
            for k in ("text_source_mode", "text_source_range_start", "text_source_range_end",
                      "refiner_lr", "epochs", "contrastive_weight", "diffusion_weight")
            if k in train_args
        }
        if relevant:
            print(f"[Checkpoint] training config: {relevant}")

    return transformer


def main(opt):
    # Load prompt metadata
    with open(opt.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    if opt.tags is not None:
        metadatas = [m for m in metadatas if m.get("tag") in opt.tags]
        print(f"Filtered prompts to {len(metadatas)} items matching tags: {opt.tags}")

    # Device selection
    if torch.cuda.is_available():
        device = "cuda"
    else:
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
        except (ImportError, RuntimeError):
            device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"Device: {device}")

    # Load base model
    model_path = opt.model_dir
    if not os.path.isdir(model_path):
        model_path = ensure_model_weights("ckpts/Z-Image-Turbo", verify=False)

    dtype = torch.bfloat16
    print(f"[Model] loading base from: {model_path}")
    components = load_from_local_dir(model_path, device=device, dtype=dtype, compile=False)

    # Hot-swap transformer weights with fine-tuned checkpoint
    transformer = components["transformer"]
    transformer = load_refiner_checkpoint(opt.checkpoint, transformer, device, strict=opt.strict)
    transformer.eval()
    components["transformer"] = transformer

    attn_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")
    set_attention_backend(attn_backend)
    print(f"Attention backend: {attn_backend}")

    # Generation loop — identical structure to zimage_generate.py
    for index, metadata in enumerate(tqdm(metadatas, desc="Prompts")):
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
            for _ in trange(
                (opt.n_samples + batch_size - 1) // batch_size,
                desc="Sampling",
                leave=False,
            ):
                current_batch_size = min(batch_size, opt.n_samples - sample_count)
                prompts = [prompt] * current_batch_size

                images = generate(
                    prompt=prompts,
                    **components,
                    height=opt.H,
                    width=opt.W,
                    num_inference_steps=opt.steps,
                    guidance_scale=opt.scale,
                    generator=torch.Generator(device).manual_seed(opt.seed + sample_count),
                )

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

        del all_samples

    print("Done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
