from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from .utils.paths import ensure_project_dirs, PACKAGES_DIR, OUTPUT_DIR, MODELS_DIR
from .utils.devices import pick_dtype
import torch
import json

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
    # validate project setup correctly
    ensure_project_dirs()

    parser = argparse.ArgumentParser(description="DreamBooth LoRA training (diffusers script wrapper)")
    # Model, and data directories
    parser.add_argument("--model-path", required=True, help="Base model path or HF hub id (downloaded into models/ if local path not absolute)")
    parser.add_argument("--instance-data-dir", required=True, help="Directory with instance images")
    parser.add_argument("--instance-prompt", required=True, help="Prompt that describes the instance (e.g., 'photo of sks person')")
    parser.add_argument("--output-dir", default=str((OUTPUT_DIR / "dreambooth_run").resolve()))

    # Training parameters
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
    parser.add_argument("--script-path", default=None, help="Path to training script. If not set, defaults to packages/diffusers_repo/examples/dreambooth/... based on model family.")
    parser.add_argument("--patch-unet-added-cond", dest="patch_unet_added_cond", action="store_true", default=True, help="If local SD 1.x/2.x model has UNet addition_embed_type set, patch it to null to avoid added_cond_kwargs errors")
    parser.add_argument("--no-patch-unet-added-cond", dest="patch_unet_added_cond", action="store_false")


    ns = parser.parse_args(argv)

    inst_dir = Path(ns.instance_data_dir)
    if not inst_dir.exists() or not inst_dir.is_dir() or not any(inst_dir.iterdir()):
        raise FileNotFoundError(f"--instance-data-dir must exist and contain images: {ns.instance_data_dir}")

    # Normalize model path. Look first locally for model. otherwise treat as HuggingFace hub id
    mpath = Path(ns.model_path)
    if mpath.exists() or mpath.is_absolute():
        # local path provided
        if not mpath.is_absolute():
            mpath = (MODELS_DIR / mpath).resolve()
        ns.model_path = str(mpath)
    else:
        # look locally under models/
        local_under_models = (MODELS_DIR / mpath).resolve()
        if local_under_models.exists():
            ns.model_path = str(local_under_models)
        else:
            # look again locally. when downloading, names sometimes change from slashes to underscores (e.g. owner/model -> owner__model)
            def _sanitize_id_to_dir(name: str) -> str:
                return name.replace("/", "__")

            sanitized = _sanitize_id_to_dir(str(mpath))
            local_sanitized = (MODELS_DIR / sanitized).resolve()
            if local_sanitized.exists():
                ns.model_path = str(local_sanitized)
                print(f"Resolved local model snapshot via sanitized ID mapping: {local_sanitized}")

    # Determine if model is SDXL (presence of text_encoder_2/tokenizer_2 or model id contains XL markers)
    def is_sdxl_model(path_or_id: str) -> bool:
        p = Path(path_or_id)
        if p.exists() and p.is_dir():
            if (p / "tokenizer_2").exists() or (p / "text_encoder_2").exists():
                return True

        # Guess from ID string
        name = str(path_or_id).lower()
        return any(k in name for k in ["sdxl", "xl", "x-l", "sd_xl"]) and not any(k in name for k in ["sd15", "1.5", "v1-5"]) 

    is_sdxl = is_sdxl_model(ns.model_path)

    # Choose default script path for model family (if one not explicitly provided)
    if ns.script_path is None:
        examples_dir = (PACKAGES_DIR / "diffusers_repo" / "examples" / "dreambooth")
        script_path = (examples_dir / ("train_dreambooth_lora_sdxl.py" if is_sdxl else "train_dreambooth_lora.py")).resolve()
        ns.script_path = str(script_path)
    else:
        script_path = Path(ns.script_path).resolve()

    if not script_path.exists():
        raise FileNotFoundError(
            f"Training script not found at {script_path}. Run setup to clone diffusers into packages/diffusers_repo, "
            f"or pass --script-path to the correct local examples/dreambooth script."
        )

    # If using a local SD 1.x/2.x model path and requested, patch UNet config to avoid added_cond_kwargs issues
    if (not is_sdxl) and ns.patch_unet_added_cond:
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
                        unet_cfg.write_text(json.dumps(data, indent=2))
                        print(f"Patched UNet config to disable addition_embed_type: {unet_cfg}")
        except Exception as e:
            print(f"Warning: failed to patch UNet config: {e}")

    # Avoid ZeroDivisionError in some scripts when checkpointing_steps == 0
    checkpointing_disabled = False
    if getattr(ns, "checkpointing_steps", 0) <= 0:
        ns.checkpointing_steps = 1_000_000_000  # effectively disabled
        checkpointing_disabled = True

    cmd = [sys.executable, "-u"] + build_args(ns)

    # Run info
    print("DreamBooth run config:")
    print(f"  model_path: {ns.model_path}")
    print(f"  family: {'SDXL' if is_sdxl else 'SD 1.x/2.x'}")
    print(f"  script: {script_path}")
    print(f"  steps: {ns.steps}, batch_size: {ns.batch_size}, grad_accum: {ns.grad_accum}, lr: {ns.learning_rate}")
    if checkpointing_disabled:
        print("  checkpointing: disabled to safe guard against modulo-by-zero errors")
    print("Running DreamBooth training script:\n", " ".join(cmd))
    proc = subprocess.run(cmd)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
