from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import os
# Avoid hard dependency on torchvision ops - transformers can work without it
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
# Improve ROCm stability with allocator tuning (must be set before torch import)
os.environ.setdefault("PYTORCH_HIP_ALLOC_CONF", "garbage_collection_threshold:0.8,max_split_size_mb:128")

import torch
from diffusers import AutoPipelineForText2Image, StableDiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image

from .utils.paths import ensure_project_dirs, MODELS_DIR, OUTPUT_DIR
from .utils.devices import pick_device, pick_dtype, set_torch_defaults


def resolve_model_path(model_path: str) -> str:
    # use absolute path if given
    p = Path(model_path)
    if p.is_absolute() or p.exists():
        return str(p)

    # no absolute path: look in local model dir
    mp = (MODELS_DIR / p).resolve()
    if mp.exists():
        return str(mp)

    # otherwise assume HuggingFace hub id
    return model_path


def is_sdxl(model_id: str) -> bool:
    low = model_id.lower()
    return "sdxl" in low or "xl" in low or "stable-diffusion-xl" in low


def load_pipeline(model_id: str, torch_dtype, device: torch.device, disable_safety_checker: bool = False):
    """Load text-to-image pipeline with automatic model family detection (SD/SDXL).

    Uses AutoPipelineForText2Image to avoid SD vs SDXL mis-detection leading to
    attention/linear shape mismatches.
    """
    try:
        pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=torch_dtype, use_safetensors=True
        )
    except Exception:
        # Fallback to legacy heuristic if auto resolution fails
        if is_sdxl(model_id):
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_id, torch_dtype=torch_dtype, use_safetensors=True
            )
        else:
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch_dtype, use_safetensors=True
            )
    if disable_safety_checker and hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
    pipe.to(device)
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe


def _patch_unet_added_cond_if_needed(pipe) -> None:
    """Work around a diffusers UNet change where added_cond_kwargs can be None.
    Some SD 1.x/2.x model configs set addition_embed_type on UNet which expects
    added_cond_kwargs to be a dict. If it's None, diffusers may error. Clearing
    the flag avoids using that path for classic SD pipelines.
    """
    try:
        cfg = getattr(pipe.unet, "config", None)
        if cfg is not None and getattr(cfg, "addition_embed_type", None) is not None:
            cfg.addition_embed_type = None
    except Exception:
        # Best-effort patch; ignore if structure differs
        pass


def run_inference(
    model_path: str,
    prompt: str,
    negative_prompt: Optional[str],
    steps: int,
    guidance: float,
    width: int,
    height: int,
    seed: Optional[int],
    num_images: int,
    outdir: Path,
    lora_path: Optional[str] = None,
    disable_safety_checker: bool = False,
    precision: Optional[str] = None,
    device_override: Optional[str] = None,
) -> list[Path]:
    ensure_project_dirs()
    set_torch_defaults()

    if device_override is not None:
        dev = device_override.lower()
        if dev not in {"cpu", "cuda"}:
            raise ValueError("--device must be 'cpu' or 'cuda'")
        device = torch.device(dev)
    else:
        device = pick_device()
    # Precision handling: override default if requested
    if precision is not None:
        prec = precision.lower()
        if prec == "bf16":
            dtype = torch.bfloat16
        elif prec == "fp16":
            dtype = torch.float16
        elif prec == "fp32":
            dtype = torch.float32
        else:
            raise ValueError(f"Unsupported precision: {precision}")
    else:
        dtype = pick_dtype(training=False)

    model_id = resolve_model_path(model_path)
    pipe = load_pipeline(model_id, dtype, device, disable_safety_checker=disable_safety_checker)

    if lora_path:
        print(f"Loading LoRA weights from: {lora_path}")
        pipe.load_lora_weights(lora_path)

    # Apply compatibility patch for non-SDXL models
    if not is_sdxl(model_id):
        _patch_unet_added_cond_if_needed(pipe)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device.type).manual_seed(seed)

    outdir.mkdir(parents=True, exist_ok=True)
    saved = []
    for i in range(num_images):
        try:
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                width=width,
                height=height,
                generator=generator,
            )
        except TypeError as e:
            # Retry once with UNet patch if diffusers expects added_cond_kwargs
            if "added_cond_kwargs" in str(e):
                _patch_unet_added_cond_if_needed(pipe)
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    width=width,
                    height=height,
                    generator=generator,
                )
            else:
                raise
        image: Image.Image = result.images[0]
        out_path = outdir / f"img_{i:03d}.png"
        image.save(out_path)
        saved.append(out_path)
    return saved


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Text-to-image inference (SD/SDXL) with ROCm acceleration")
    p.add_argument("--model-path", required=True, help="Local path under models/ or HF hub id")
    p.add_argument("--lora-path", default=None, help="Path to LoRA weights (.safetensors file) to apply to the base model.")
    p.add_argument("--prompt", required=True)
    p.add_argument("--negative-prompt", default=None)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-images", type=int, default=1)
    p.add_argument("--output-dir", default=str((OUTPUT_DIR / "inference").resolve()))
    p.add_argument("--disable-safety-checker", action="store_true", help="Disable NSFW safety checker")
    p.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default=None, help="Override precision (default picks best for device)")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Force device override (use 'cpu' to rule out ROCm issues)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outdir = Path(args.output_dir)
    saved = run_inference(
        model_path=args.model_path,
        lora_path=args.lora_path,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        steps=args.steps,
        guidance=args.guidance,
        width=args.width,
        height=args.height,
        seed=args.seed,
        num_images=args.num_images,
        outdir=outdir,
        disable_safety_checker=args.disable_safety_checker,
        precision=args.precision,
        device_override=args.device,
    )
    print("Saved:")
    for p in saved:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
