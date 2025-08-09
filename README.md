# TODO
Do i need to set these in environment?
HSA_ENABLE_SDMA=0 MIOPEN_DEBUG_DISABLE_FIND_DB=1
MIOPEN_USER_DB_PATH=/tmp/miopen_db

example: HSA_ENABLE_SDMA=0 MIOPEN_DEBUG_DISABLE_FIND_DB=1 MIOPEN_USER_DB_PATH=/tmp/miopen_db
poetry run python -m src.infer
--model-path stabilityai/stable-diffusion-2-1-base
--prompt "a cinematic portrait of a red-haired woman, soft rim lighting, 85mm lens, f/1.8, volumetric light, detailed skin texture, RAW photo, high dynamic range, subtle film grain"
--negative-prompt "blurry, lowres, jpeg artifacts, overexposed, underexposed, extra fingers, deformed"
--steps 28 --guidance 7.0 --width 768 --height 768
--precision bf16 --seed 42


AI Image Generation (ROCm, DreamBooth/LoRA)
===========================================

Experiments for local Stable Diffusion (SD/SDXL) inference and DreamBooth fine-tuning on AMD GPUs with ROCm.

Project layout
--------------
- `src/` — Python modules and scripts
  - `src/infer.py` — SD/SDXL inference with ROCm
  - `src/train_dreambooth_wrapper.py` — Wrapper around the official Diffusers DreamBooth LoRA script; downloads the trainer on first run
  - `src/download_model.py` — Download Hugging Face repos into `models/`
  - `src/utils/` — Helpers for paths and device/dtype selection
- `models/` — Base models and fine-tuned checkpoints (tracked via `.gitkeep`)
- `packages/` — External helper scripts (e.g., downloaded DreamBooth script)
- `output/` — Images, logs, and checkpoints

Requirements
------------
- Linux with AMD GPU (e.g., RX 7900 XTX) and ROCm drivers installed
- Python 3.12 (managed by Poetry)
- Internet access for first runs (to download models/scripts), unless you pre-populate `models/`

Setup
-----
1) Install Poetry (see https://python-poetry.org/docs/)

2) Install project dependencies (excluding PyTorch):
```
poetry install
```

3) Install PyTorch ROCm builds (pick the ROCm index matching your system, e.g., rocm6.1):
> **AMD/Nvidia**: the command below is needed to use AMD with the `torch` package. If running Nvidia, install `torch` normally.
```
poetry run pip install --index-url https://download.pytorch.org/whl/rocm6.1 torch torchvision torchaudio
```
Other indices if needed: `rocm6.0`, `rocm6.2`.

4) Verify ROCm Torch sees your GPU:
```
poetry run python - << 'PY'
import torch
print('HIP:', getattr(torch.version, 'hip', None))
print('CUDA available (True on ROCm builds):', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
PY
```

Download a model
----------------
To download a model repo into `models/` (e.g., stable-diffusion-v1-5/stable-diffusion-v1-5):
```
poetry run python -m src.download_model stable-diffusion-v1-5/stable-diffusion-v1-5
```
This will create a directory like `models/stable-diffusion-v1-5/stable-diffusion-v1-5/` containing model files.

Inference (SD/SDXL)
-------------------
Run text-to-image generation. You can pass either a local path under `models/` or a Hugging Face repo id.
```
poetry run python -m src.infer \
  --model-path stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --prompt "a high quality portrait photo of a person in studio lighting" \
  --disable-safety-checker \
  --steps 30 --guidance 5.0 --width 1024 --height 1024 \
  --num-images 2 \
  --output-dir output/inference
```

DreamBooth LoRA training
------------------------
The training wrapper downloads and runs the official Diffusers DreamBooth LoRA script. It auto-selects mixed precision (bf16 on ROCm-capable cards).

Example (small test run):
```
poetry run python -m src.train_dreambooth_wrapper \
  --model-path stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --instance-data-dir /path/to/your/person_images \
  --instance-prompt "photo of sks person" \
  --steps 200 \
  --batch-size 1 --grad-accum 4 \
  --output-dir output/dreambooth_run
```
Tips:
- Use 10–20 curated instance images to sanity-check; increase steps for better results (e.g., 800–2000).
- The wrapper supports optional prior preservation via `--class-data-dir`, `--class-prompt`, and `--prior-loss-weight`.
- Outputs (checkpoints, logs) go under `output/dreambooth_run`.

Troubleshooting
---------------
- ROCm/PyTorch install: Ensure you used the correct ROCm index URL for your driver stack.
- HIP not detected: Check ROCm installation and that your user has access to the GPU.
- Memory errors: Reduce `--width/--height`, steps, or try smaller batch size/gradient accumulation.
- SD vs SDXL: The inference script auto-detects SDXL when the model id contains `xl`/`sdxl`.

License
-------
This repository provides glue code and wrappers; follow upstream licenses for models/datasets and the Diffusers training script.
