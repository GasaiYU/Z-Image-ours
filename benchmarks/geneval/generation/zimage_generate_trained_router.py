"""Generate GenEval images using the trained DynamicTokenRouter.

The trained router replaces the standard ``hidden_states[-2]`` text-encoder
output with a learned per-token weighted fusion across all LLM layers.

Unlike the heuristic decay method, the router uses a lightweight MLP to
*learn* per-token routing weights from triplet contrastive training:
  - attribute tokens (counting numbers, colors, …) are routed to shallower
    LLM layers where surface-form representations are stronger.
  - noun tokens are regularised to stay in the deep layer.

Usage (generation only):
    python benchmarks/geneval/generation/zimage_generate_router.py \\
        benchmarks/geneval/prompts/evaluation_metadata.jsonl \\
        --router_ckpt checkpoints/router/router_best.pt \\
        --outdir outputs/geneval_router_counting_seed_42 \\
        --tags counting \\
        --seed 42

Full pipeline (generation + evaluation), use the companion shell script:
    bash scripts/evaluate_geneval_router.sh checkpoints/router/router_best.pt 42
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
# Path setup – repo root first so "utils" and "zimage" are importable
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
_TRAIN_TEXT_DIR = os.path.join(_REPO_ROOT, "train_text")
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, _TRAIN_TEXT_DIR)

from utils import ensure_model_weights, load_from_local_dir, set_attention_backend
from zimage import generate
from train_router import DynamicTokenRouter   # noqa: E402 (path set above)

torch.set_grad_enabled(False)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="GenEval image generation with trained DynamicTokenRouter"
    )
    parser.add_argument(
        "metadata_file", type=str,
        help="JSONL file containing one metadata record per line",
    )
    parser.add_argument(
        "--router_ckpt", type=str, required=True,
        help="Path to trained router checkpoint (.pt)",
    )
    parser.add_argument(
        "--outdir", type=str, default="outputs",
        help="Directory to write generated images",
    )
    parser.add_argument("--n_samples",  type=int,   default=4,    help="Samples per prompt")
    parser.add_argument("--steps",      type=int,   default=8,    help="Diffusion inference steps")
    parser.add_argument("--H",          type=int,   default=1024, help="Image height")
    parser.add_argument("--W",          type=int,   default=1024, help="Image width")
    parser.add_argument("--scale",      type=float, default=0.0,  help="CFG guidance scale")
    parser.add_argument("--seed",       type=int,   default=42,   help="Random seed")
    parser.add_argument("--batch_size", type=int,   default=1,    help="Images per forward pass")
    parser.add_argument("--skip_grid",  action="store_true",      help="Do not save image grid")
    parser.add_argument(
        "--tags", type=str, nargs="+", default=None,
        help="Only process prompts matching these tags (e.g. counting)",
    )
    parser.add_argument(
        "--max_sequence_length", type=int, default=512,
        help="Max tokenizer sequence length (must match training)",
    )
    parser.add_argument(
        "--model_dir", type=str, default="ckpts/Z-Image-Turbo",
        help="Z-Image model directory (passed to ensure_model_weights / load_from_local_dir)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Router loading
# ---------------------------------------------------------------------------
def load_router(ckpt_path: str, device) -> DynamicTokenRouter:
    """Load a trained DynamicTokenRouter checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    hidden_size = ckpt["hidden_size"]
    num_layers  = ckpt["num_layers"]
    mid_dim     = ckpt["mid_dim"]

    router = DynamicTokenRouter(hidden_size=hidden_size, num_layers=num_layers, mid_dim=mid_dim)

    # Handle checkpoints saved from DDP (keys may have "module." prefix)
    state_dict = ckpt["router_state_dict"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    router.load_state_dict(state_dict)
    router.to(device)
    router.eval()

    print(f"[Router] Loaded  : {ckpt_path}")
    print(f"[Router] Config  : hidden_size={hidden_size}, num_layers={num_layers}, mid_dim={mid_dim}")
    if "epoch" in ckpt:
        print(f"[Router] Trained : epoch={ckpt['epoch']}, step={ckpt.get('step', '?')}, "
              f"best_loss={ckpt.get('best_loss', 'N/A')}")

    return router


# ---------------------------------------------------------------------------
# Core: build router-fused prompt embeddings
# ---------------------------------------------------------------------------
def build_router_embeds(
    prompts,
    text_encoder,
    tokenizer,
    router: DynamicTokenRouter,
    device,
    max_sequence_length: int,
):
    """
    Tokenize *prompts*, run the frozen text encoder to collect all hidden
    states, then pass them through the trained router to produce per-token
    fused embeddings that replace ``hidden_states[-2]``.

    Returns
    -------
    fused_embeds  : [B, S, D] bfloat16  – router output (replaces hs[-2])
    input_ids     : [B, S]   int64      – ids used inside the pipeline call
    attention_mask: [B, S]   int64
    n_total_hs    : int                 – total hidden-state list length,
                                          needed to build the mock list
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
        all_hidden_states = outputs.hidden_states   # tuple, len = num_layers + 1

        # Router: learned weighted sum across all transformer layers
        fused_embeds, _routing_weights = router(
            all_hidden_states, attention_mask=attention_mask.bool()
        )
        # fused_embeds: [B, S, D], same dtype as LLM output (bfloat16)

    n_total_hs = len(all_hidden_states)   # e.g. 33 for a 32-layer LLM
    return fused_embeds, input_ids, attention_mask, n_total_hs


# ---------------------------------------------------------------------------
# Generation with patched text encoder
# ---------------------------------------------------------------------------
def generate_with_router(components, prompts, opt, device, generator, router):
    """
    Generate images for *prompts* using the router's fused embeddings.

    The trick (same as zimage_generate_decay.py):
      1. Pre-compute ``fused_embeds`` before calling the pipeline.
      2. Monkey-patch ``text_encoder.forward`` to return a fake output whose
         ``hidden_states[-2]`` is our ``fused_embeds``.
      3. The pipeline calls the patched forward, gets the router embeddings,
         and generates the image.
      4. Restore the original forward.
    """
    text_encoder = components["text_encoder"]

    fused_embeds, expected_ids, _, n_total_hs = build_router_embeds(
        prompts,
        text_encoder,
        components["tokenizer"],
        router,
        device,
        opt.max_sequence_length,
    )

    original_forward = text_encoder.forward

    def patched_forward(input_ids, attention_mask=None, **kwargs):
        class _Out:
            pass

        # Only intercept the positive-prompt call that matches our pre-computed ids.
        # (guidance_scale=0.0 means CFG is off; the negative-prompt branch is never hit.)
        if input_ids.shape == expected_ids.shape and torch.equal(input_ids, expected_ids):
            out = _Out()
            # Build a minimal hidden_states list: only slot [-2] needs a real value.
            # Length matches the actual text encoder so that [-2] maps correctly.
            mock_hs = [None] * n_total_hs
            mock_hs[-2] = fused_embeds
            out.hidden_states = mock_hs
            return out

        # Fallback: any other call (e.g. negative prompt) goes through normally
        return original_forward(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(opt):
    # ---- Load metadata ----
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
    print(f"[Attention] backend = {attn_backend}")

    # ---- Load Router ----
    router = load_router(opt.router_ckpt, device=device)

    # ---- Generation loop ----
    os.makedirs(opt.outdir, exist_ok=True)

    for index, metadata in enumerate(metadatas):
        seed_everything(opt.seed)

        outpath    = os.path.join(opt.outdir, f"{index:0>5}")
        os.makedirs(outpath, exist_ok=True)

        prompt = metadata["prompt"]
        batch_size = opt.batch_size
        n_rows     = batch_size
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
            generator  = torch.Generator(device).manual_seed(opt.seed + sample_count)

            images = generate_with_router(
                components, prompts, opt, device, generator, router
            )

            for img in images:
                img.save(os.path.join(sample_path, f"{sample_count:05}.png"))
                sample_count += 1
                if not opt.skip_grid:
                    all_samples.append(ToTensor()(img))

        if not opt.skip_grid and all_samples:
            grid = torch.stack(all_samples)
            grid = make_grid(grid, nrow=n_rows)
            grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
            Image.fromarray(grid.astype(np.uint8)).save(
                os.path.join(outpath, "grid.png")
            )
            del grid
        del all_samples

    print("\n[Done] All prompts processed.")


if __name__ == "__main__":
    main(parse_args())
