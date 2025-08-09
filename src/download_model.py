from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from .utils.paths import ensure_project_dirs, MODELS_DIR


def download(model_id: str, local_dir: Path, allow_patterns: list[str] | None = None) -> Path:
    ensure_project_dirs()
    local_dir.mkdir(parents=True, exist_ok=True)
    dest = local_dir / model_id.replace("/", "__")
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=str(dest), allow_patterns=allow_patterns)
    return dest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Download a Hugging Face model/repo into models/")
    p.add_argument("model_id", help="HF repo id, e.g. 'stable-diffusion-v1-5/stable-diffusion-v1-5'")
    p.add_argument("--only", nargs="*", default=None, help="Optional allow_patterns list (e.g. *.safetensors *.json)")
    args = p.parse_args(argv)

    dest = download(args.model_id, MODELS_DIR, allow_patterns=args.only)
    print(f"Downloaded to: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
