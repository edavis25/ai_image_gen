from __future__ import annotations

import os
import torch


def pick_device() -> torch.device:
    # On ROCm: torch.cuda.is_available() should also be True and device will use type 'cuda'
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def pick_dtype(training: bool = False):
    # RDNA3 GPUs (e.g., 7900XTX) support bfloat16; prefer bf16 for training and inference
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16

        # If cuda but no bf16 support, fallback to float32 for training and float 16 for inference.
        return torch.float32 if training else torch.float16

    return torch.float32


def set_torch_defaults():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # Disable TF32 by default on AMD; set explicitly if desired
    torch.backends.cuda.matmul.allow_tf32 = False
    # On ROCm some SDPA kernels can be unstable; force math backend for stability
    try:
        torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False)
    except Exception:
        pass


__all__ = ["pick_device", "pick_dtype", "set_torch_defaults"]
