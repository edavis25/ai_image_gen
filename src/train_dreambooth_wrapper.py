from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import requests

from .utils.paths import ensure_project_dirs, PACKAGES_DIR, OUTPUT_DIR, MODELS_DIR
from .utils.devices import pick_dtype
import torch
import json
from .download_model import download as download_hf_repo

# Public sources for the DreamBooth LoRA training scripts (SD and SDXL variants).
# Try GitHub raw first (no auth required), then HF datasets as a fallback.
TRAIN_SCRIPT_URLS_SD = [
    "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/dreambooth/train_dreambooth_lora.py",
    "https://huggingface.co/datasets/diffusers/training_scripts/resolve/main/dreambooth/train_dreambooth_lora.py",
]
TRAIN_SCRIPT_URLS_SDXL = [
    "https://raw.githubusercontent.com/huggingface/diffusers/main/examples/sdxl_dreambooth/train_dreambooth_lora_sdxl.py",
    "https://huggingface.co/datasets/diffusers/training_scripts/resolve/main/sdxl_dreambooth/train_dreambooth_lora_sdxl.py",
]


def ensure_train_script(local_path: Path, urls: list[str]) -> None:
    if local_path.exists():
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading DreamBooth LoRA training script to {local_path} ...")

    # Optional Hugging Face token support if the dataset URL requires auth or higher rate limits
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    headers = {"User-Agent": "ai-image-gen/1.0"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    last_err: Exception | None = None
    for url in urls:
        try:
            r = requests.get(url, timeout=60, headers=headers)
            r.raise_for_status()
            local_path.write_bytes(r.content)
            print(f"Download complete from: {url}")
            return
        except Exception as e:
            print(f"Warning: failed to download from {url}: {e}")
            last_err = e
            continue
    # If we reach here, all sources failed
    if last_err:
        raise last_err


def build_args(ns: argparse.Namespace) -> list[str]:
    args = [
        str(ns.script_path),
        "--pretrained_model_name_or_path",
        ns.model_path,
        "--instance_data_dir",
        ns.instance_data_dir,
        "--instance_prompt",
        ns.instance_prompt,
        "--output_dir",
        ns.output_dir,
        "--resolution",
        str(ns.resolution),
        "--gradient_accumulation_steps",
        str(ns.grad_accum),
        "--train_batch_size",
        str(ns.batch_size),
        "--learning_rate",
        str(ns.learning_rate),
        "--lr_scheduler",
        ns.lr_scheduler,
        "--lr_warmup_steps",
        str(ns.lr_warmup_steps),
        "--max_train_steps",
        str(ns.steps),
        "--checkpointing_steps",
        str(ns.checkpointing_steps),
        "--seed",
        str(ns.seed),
        "--gradient_checkpointing",
        "--train_text_encoder",
        "--center_crop",
    ]

    if ns.class_data_dir and ns.class_prompt:
        args += [
            "--class_data_dir",
            ns.class_data_dir,
            "--class_prompt",
            ns.class_prompt,
            "--with_prior_preservation",
            "--prior_loss_weight",
            str(ns.prior_loss_weight),
        ]

    # Mixed precision preference (bf16 if available)
    dtype = pick_dtype(training=True)
    # pass mixed precision explicitly
    if dtype is torch.bfloat16:
        args += ["--mixed_precision", "bf16"]
    else:
        args += ["--mixed_precision", "no"]

    return args


def main(argv: list[str] | None = None) -> int:
    ensure_project_dirs()

    parser = argparse.ArgumentParser(description="DreamBooth LoRA training (diffusers script wrapper)")
    parser.add_argument("--model-path", required=True, help="Base model path or HF hub id (downloaded into models/ if local path not absolute)")
    parser.add_argument("--instance-data-dir", required=True, help="Directory with instance images")
    parser.add_argument("--instance-prompt", required=True, help="Prompt that describes the instance (e.g., 'photo of sks person')")
    parser.add_argument("--output-dir", default=str((OUTPUT_DIR / "dreambooth_run").resolve()))
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lr-scheduler", default="constant")
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--checkpointing-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)

    # Prior preservation (optional)
    parser.add_argument("--class-data-dir", default=None)
    parser.add_argument("--class-prompt", default=None)
    parser.add_argument("--prior-loss-weight", type=float, default=1.0)

    # Script management
    parser.add_argument("--script-path", default=None, help="Path to training script. If not set, auto-downloads SD or SDXL DreamBooth LoRA script.")
    parser.add_argument("--patch-unet-added-cond", action="store_true", default=True, help="If local model has UNet addition_embed_type set, patch it to null to avoid added_cond_kwargs errors with SD 1.x/2.x training")

    ns = parser.parse_args(argv)

    # Normalize model path: prefer local path if exists; otherwise treat as HF hub id
    mpath = Path(ns.model_path)
    if mpath.exists() or mpath.is_absolute():
        # local path provided
        if not mpath.is_absolute():
            mpath = (MODELS_DIR / mpath).resolve()
        ns.model_path = str(mpath)
    else:
        # try under models/
        local_under_models = (MODELS_DIR / mpath).resolve()
        if local_under_models.exists():
            ns.model_path = str(local_under_models)
        else:
            # Hub id: download a local snapshot so we can patch configs if needed
            print(f"Model '{ns.model_path}' not found locally. Downloading snapshot under models/ ...")
            dest = download_hf_repo(ns.model_path, MODELS_DIR, allow_patterns=None)
            ns.model_path = str(dest)

    # If using a local model path and requested, patch UNet config to avoid added_cond_kwargs NoneType issues.
    # Some SD 1.x/2.x configs set addition_embed_type which expects added_cond_kwargs; training script may pass None.
    # We set it to null so training follows the classic SD path.
    if ns.patch_unet_added_cond:
        try:
            local_model_path = Path(ns.model_path)
            if local_model_path.exists() and local_model_path.is_dir():
                unet_cfg = local_model_path / "unet" / "config.json"
                if unet_cfg.exists():
                    data = json.loads(unet_cfg.read_text())
                    if data.get("addition_embed_type") is not None:
                        backup = unet_cfg.with_suffix(".json.bak")
                        if not backup.exists():
                            backup.write_text(unet_cfg.read_text())
                        data["addition_embed_type"] = None
                        # Some configs have addition_time_embed_dim; it's safe to keep, but we can also null it.
                        # data["addition_time_embed_dim"] = None
                        unet_cfg.write_text(json.dumps(data, indent=2))
                        print(f"Patched UNet config to disable addition_embed_type: {unet_cfg}")
        except Exception as e:
            print(f"Warning: failed to patch UNet config: {e}")

    # Determine if model is SDXL (presence of text_encoder_2/tokenizer_2 or model id contains xL markers)
    def is_sdxl_model(path_or_id: str) -> bool:
        low = path_or_id.lower()
        if any(k in low for k in ["sdxl", "stable-diffusion-xl", "-xl", "/xl"]):
            return True
        p = Path(path_or_id)
        try:
            if p.exists() and p.is_dir():
                if (p / "text_encoder_2").exists() or (p / "tokenizer_2").exists():
                    return True
        except Exception:
            pass
        return False

    is_sdxl = is_sdxl_model(ns.model_path)

    # Choose default script path per family if not provided
    if ns.script_path is None:
        if is_sdxl:
            script_path = (PACKAGES_DIR / "diffusers_training_scripts" / "sdxl_dreambooth" / "train_dreambooth_lora_sdxl.py").resolve()
        else:
            script_path = (PACKAGES_DIR / "diffusers_training_scripts" / "dreambooth" / "train_dreambooth_lora.py").resolve()
        ns.script_path = str(script_path)
    else:
        script_path = Path(ns.script_path).resolve()

    # Ensure training script is available (download only when using the default location).
    # If the user supplied --script-path, do NOT auto-download; require the path to exist to avoid surprises.
    if Path(ns.script_path) == script_path and script_path.exists():
        pass  # user-supplied path exists
    elif ns.script_path and Path(ns.script_path).exists():
        script_path = Path(ns.script_path)
    elif ns.script_path and not Path(ns.script_path).exists():
        raise FileNotFoundError(f"--script-path does not exist: {ns.script_path}. If you intend to use the diffusers example, clone the repo and point to examples/sdxl_dreambooth/train_dreambooth_lora_sdxl.py (for SDXL) or examples/dreambooth/train_dreambooth_lora.py (for SD 1.x/2.x).")
    else:
        # No explicit script path provided; fetch the appropriate script
        ensure_train_script(script_path, TRAIN_SCRIPT_URLS_SDXL if is_sdxl else TRAIN_SCRIPT_URLS_SD)

    cmd = [sys.executable, "-u"] + build_args(ns)

    print("Running DreamBooth training script:\n", " ".join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
