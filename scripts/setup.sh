#!/usr/bin/env bash
set -euo pipefail

# Simple local setup for DreamBooth/LoRA training on AMD ROCm
# - Ensures repo directories exist
# - Verifies Poetry is available and installs project deps
# - Clones diffusers into packages/diffusers_repo if missing and installs it in editable mode
# - Installs DreamBooth example requirements
#
# Usage:
#   bash scripts/setup.sh
#
# Notes:
# - PyTorch with ROCm should already be installed in your Poetry env; if not, see README for ROCm install instructions.
# - This script will not overwrite an existing diffusers_repo. Delete or move it if you want a fresh clone.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
PACKAGES_DIR="$ROOT_DIR/packages"
DIFFUSERS_DIR="$PACKAGES_DIR/diffusers_repo"
EX_DREAMBOOTH_DIR="$DIFFUSERS_DIR/examples/dreambooth"

mkdir -p "$ROOT_DIR/models" "$ROOT_DIR/output" "$PACKAGES_DIR"

# 1) Ensure Poetry
if ! command -v poetry >/dev/null 2>&1; then
  echo "[ERROR] Poetry is not installed. See https://python-poetry.org/docs/#installation" >&2
  exit 1
fi

# 2) Install project deps
echo "[INFO] Installing project dependencies via Poetry..."
poetry install --with dev || {
  echo "[WARN] Poetry install failed. Trying without optional groups..."
  poetry install
}

# 3) Verify PyTorch + ROCm in the environment (non-fatal, with tailored guidance)
set +e
POETRY_PY="$(poetry run python -c 'import sys; print(sys.executable)' 2>/dev/null)"
if [ -n "$POETRY_PY" ]; then
  echo "[INFO] Python used by Poetry: $POETRY_PY"
  PY_INFO="$(poetry run python - <<'PYCHK'
import json
try:
    import torch
    info = {
        "status": "OK",
        "torch_version": getattr(torch, "__version__", None),
        "hip_version": getattr(getattr(torch, "version", None), "hip", None),
        "cuda_available": bool(getattr(getattr(torch, "cuda", None), "is_available", lambda: False)()),
        "device_count": int(getattr(getattr(torch, "cuda", None), "device_count", lambda: 0)()),
        "device_names": [],
    }
    try:
        dc = info["device_count"]
        for i in range(dc):
            name = torch.cuda.get_device_name(i)
            info["device_names"].append(name)
    except Exception:
        pass
except Exception:
    info = {"status": "NO_TORCH"}
print(json.dumps(info))
PYCHK
)"
  # Always continue even if check fails
  :
  # Parse JSON
  STATUS=$(printf '%s' "$PY_INFO" | sed -n 's/.*"status"\s*:\s*"\([^"]*\)".*/\1/p')
  TORCH_VER=$(printf '%s' "$PY_INFO" | sed -n 's/.*"torch_version"\s*:\s*"\([^"]*\)".*/\1/p')
  HIP_VER=$(printf '%s' "$PY_INFO" | sed -n 's/.*"hip_version"\s*:\s*"\([^"]*\)".*/\1/p')
  CUDA_AVAIL=$(printf '%s' "$PY_INFO" | sed -n 's/.*"cuda_available"\s*:\s*\(true\|false\).*/\1/p')
  DEV_COUNT=$(printf '%s' "$PY_INFO" | sed -n 's/.*"device_count"\s*:\s*\([0-9]\+\).*/\1/p')

  echo "[INFO] PyTorch probe: $PY_INFO"

  if [ "$STATUS" = "NO_TORCH" ] || [ -z "$STATUS" ]; then
    echo "[WARN] PyTorch is not installed in the Poetry environment."
    echo "       Install ROCm-enabled PyTorch inside Poetry with:"
    echo "       poetry run pip install --index-url https://download.pytorch.org/whl/rocm6.1 torch torchvision torchaudio"
  else
    echo "[INFO] Detected torch=$TORCH_VER, HIP=$HIP_VER, cuda_available=$CUDA_AVAIL, device_count=${DEV_COUNT:-0}"
    if [ -z "$HIP_VER" ] || [ "$HIP_VER" = "null" ]; then
      echo "[WARN] This torch build does not report a HIP version. It may not be a ROCm wheel."
      echo "       Reinstall with ROCm wheels:"
      echo "       poetry run pip install --upgrade --index-url https://download.pytorch.org/whl/rocm6.1 torch torchvision torchaudio"
    elif [ "$CUDA_AVAIL" != "true" ] || [ "${DEV_COUNT:-0}" = "0" ]; then
      echo "[WARN] ROCm detected (HIP=$HIP_VER) but no GPU is visible to PyTorch."
      echo "       Check your AMD drivers/ROCm install and try the env vars below (HSA_ENABLE_SDMA=0, etc.)."
    else
      echo "[INFO] ROCm looks good. Visible devices: ${DEV_COUNT:-0}"
    fi
  fi
fi
set -e

# 4) Clone diffusers if needed
if [ ! -d "$DIFFUSERS_DIR/.git" ]; then
  echo "[INFO] Cloning diffusers into $DIFFUSERS_DIR ..."
  git clone --depth 1 https://github.com/huggingface/diffusers "$DIFFUSERS_DIR"
else
  echo "[INFO] diffusers already present at $DIFFUSERS_DIR"
fi

# 5) Install diffusers (editable)
echo "[INFO] Installing diffusers in editable mode..."
poetry run pip install -U pip setuptools wheel
poetry run pip install -e "$DIFFUSERS_DIR" \
  -r "$EX_DREAMBOOTH_DIR/requirements.txt" \
  -r "$EX_DREAMBOOTH_DIR/requirements_sdxl.txt"

# 6) ROCm stability hints (non-fatal)
echo "[INFO] ROCm stability hints (set in your shell/profile if needed):"
echo "  export HSA_ENABLE_SDMA=0"
echo "  export MIOPEN_DEBUG_DISABLE_FIND_DB=1"
echo "  export MIOPEN_USER_DB_PATH=/tmp/miopen_db"

echo "[INFO] If PyTorch with ROCm is not installed in your Poetry env, install it with:"
echo "  poetry run pip install --index-url https://download.pytorch.org/whl/rocm6.1 torch torchvision torchaudio"
echo "[INFO] Verify install with:"
echo "  poetry run python -c 'import torch; print(torch.__version__, getattr(torch.version, \"hip\", None), torch.cuda.is_available())'"

echo "[DONE] Setup complete. You can now run training via:"
echo "  poetry run python -m src.train_dreambooth_wrapper --model-path <local|HF_id> --instance-data-dir <dir> --instance-prompt 'photo of sks person'"
