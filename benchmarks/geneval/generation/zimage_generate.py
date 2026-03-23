"""Adapted from diffusers_generate.py for Z-Image-Turbo"""

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

# Add root directory to sys.path to import zimage and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from utils import AttentionBackend, ensure_model_weights, load_from_local_dir, set_attention_backend
from zimage import generate

torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "metadata_file",
        type=str,
        help="JSONL file containing lines of metadata for each prompt"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="number of samples",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="number of inference steps (default 8 for Z-Image-Turbo)",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=1024,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=1024,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.0,
        help="unconditional guidance scale (default 0.0 for Z-Image-Turbo)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="how many samples can be produced simultaneously",
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="skip saving grid",
    )
    opt = parser.parse_args()
    return opt


def main(opt):
    # Load prompts
    with open(opt.metadata_file) as fp:
        metadatas = [json.loads(line) for line in fp]

    # Device selection priority: cuda -> tpu -> mps -> cpu
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

    # Load Z-Image-Turbo models
    model_path = ensure_model_weights("ckpts/Z-Image-Turbo", verify=False)
    dtype = torch.bfloat16
    compile_model = False  # default False for compatibility
    
    components = load_from_local_dir(model_path, device=device, dtype=dtype, compile=compile_model)
    
    attn_backend = os.environ.get("ZIMAGE_ATTENTION", "_native_flash")
    set_attention_backend(attn_backend)
    print(f"Chosen attention backend: {attn_backend}")

    for index, metadata in enumerate(metadatas):
        seed_everything(opt.seed)

        outpath = os.path.join(opt.outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        prompt = metadata['prompt']
        n_rows = batch_size = opt.batch_size
        print(f"Prompt ({index: >3}/{len(metadatas)}): '{prompt}'")

        sample_path = os.path.join(outpath, "samples")
        os.makedirs(sample_path, exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as fp:
            json.dump(metadata, fp)

        sample_count = 0

        with torch.no_grad():
            all_samples = list()
            for n in trange((opt.n_samples + batch_size - 1) // batch_size, desc="Sampling"):
                current_batch_size = min(batch_size, opt.n_samples - sample_count)
                
                # Create a list of prompts for batched generation
                prompts = [prompt] * current_batch_size

                # Generate images
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
                # additionally, save as grid
                grid = torch.stack(all_samples, 0)
                grid = make_grid(grid, nrow=n_rows)

                # to image
                grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                grid = Image.fromarray(grid.astype(np.uint8))
                grid.save(os.path.join(outpath, f'grid.png'))
                del grid
        del all_samples

    print("Done.")


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
