ROCm Stable Diffusion Toolkit — SD/SDXL Inference + DreamBooth LoRA
===================================================================

Generate images and run DreamBooth training on local models and AMD GPUs.

Description
-----------
- Run Stable Diffusion (SD/SDXL) text‑to‑image inference on AMD GPUs (ROCm).
- Fine‑tune LoRA adapters via DreamBooth using the official Diffusers scripts (SD and SDXL).
- Auto‑detect SD vs SDXL from the model; use local model folders or Hugging Face IDs.

Installation
------------
1) Prerequisites
   - Linux with AMD GPU and ROCm drivers
   - Python 3.12 and Poetry

2) Run setup
```
bash scripts/setup.sh
```

3) Determine your ROCm version
  - `/opt/rocm/bin/rocminfo | grep Runtime`
  - `hipconfig --version` (if available)

4) Install ROCm‑enabled PyTorch matching your version
```
poetry run pip install --index-url https://download.pytorch.org/whl/rocm6.1 torch torchvision torchaudio
```
> Note: Replace `rocm6.1` in above example with your ROCm version (e.g., `rocm6.0`, `rocm6.2`).

5) Verify PyTorch Installation:
```
poetry run python -c "import torch; print(torch.__version__, getattr(torch.version, 'hip', None), torch.cuda.is_available())"
```

Usage
-----
This toolkit provides three primary actions:

1) Download a model snapshot
```
poetry run python -m src.download_model stabilityai/stable-diffusion-2-1-base
```

2) Generate images (inference)
```
poetry run python -m src.infer \
  --model-path stabilityai/stable-diffusion-2-1-base \
  --prompt "a cinematic portrait, dramatic lighting, high detail" \
  --steps 28 --guidance 7.0 --width 768 --height 768 \
  --num-images 2 \
  --output-dir output/inference
```

3) Train a LoRA adapter (DreamBooth)
```
poetry run python -m src.train_dreambooth_wrapper \
  --model-path stabilityai/stable-diffusion-2-1-base \
  --instance-data-dir /path/to/images \
  --instance-prompt "photo of sks person" \
  --steps 200 \
  --batch-size 1 --grad-accum 4 \
  --output-dir output/dreambooth_run
```
Notes:
- For stronger results later: steps 800–1200, learning rate 2e‑5 to 5e‑5, SDXL resolution 1024.
- The training wrapper auto‑selects the SD or SDXL script based on the model.

Troubleshooting
---------------
- PyTorch/ROCm not found in Poetry env:
  - Install with ROCm wheels: `poetry run pip install --index-url https://download.pytorch.org/whl/rocm6.1 torch torchvision torchaudio`
  - Re‑run `bash scripts/setup.sh` to re‑probe.
- Torch shows no HIP version or 0 devices:
  - Reinstall with ROCm wheels; check AMD drivers/ROCm install.
  - Try env hints: `HSA_ENABLE_SDMA=0`, `MIOPEN_DEBUG_DISABLE_FIND_DB=1`, `MIOPEN_USER_DB_PATH=/tmp/miopen_db`.
- Training script not found:
  - Ensure Diffusers is cloned at `packages/diffusers_repo` (run setup script).
- Checkpointing‑steps = 0 error:
  - The wrapper guards this by using a large value; avoid passing 0 if scripting directly.
- SD vs SDXL selection:
  - Wrapper detects SDXL via local files (`tokenizer_2`/`text_encoder_2`) or model ID heuristics.

Project layout
--------------
- `src/infer.py` — SD/SDXL inference (generate image)
- `src/train_dreambooth_wrapper.py` — DreamBooth LoRA wrapper (train adapter)
- `src/download_model.py` — Helper to snapshot HF repos into `models/` (download a model)
- `src/utils/` — Paths, device/dtype helpers
- `models/` — Base models and fine‑tuned outputs (gitignored contents)
- `packages/` — External repos (e.g., `diffusers_repo`, gitignored contents)
- `output/` — Images, logs, LoRA adapters

License
-------
Follow upstream licenses for models/datasets and the Diffusers training scripts.
