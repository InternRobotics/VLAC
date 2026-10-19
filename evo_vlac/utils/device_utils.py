"""Device selection for the XPU, CUDA and CPU backends.

Detection order is XPU, then CUDA, then CPU. An explicit ``device_map`` passed by
the caller always wins over detection.

Flash attention only exists for CUDA. The InternVL2 remote code has no SDPA
class for the language model (only ``eager`` and ``flash_attention_2``), so every
other backend loads with ``eager`` attention.
"""

import importlib.util

import torch
from loguru import logger

_FALLBACK_LOGGED = False


def is_available(device_type: str) -> bool:
    """Whether ``device_type`` is usable in this torch build and on this host."""
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "cuda":
        return torch.cuda.is_available()
    return device_type == "cpu"


def default_device_type() -> str:
    """Best available backend: ``xpu`` > ``cuda`` > ``cpu``."""
    for device_type in ("xpu", "cuda"):
        if is_available(device_type):
            return device_type
    return "cpu"


def default_device_map() -> str:
    """Default ``device_map`` for ``init_model``: the first device of the best backend."""
    device_type = default_device_type()
    return "cpu" if device_type == "cpu" else f"{device_type}:0"


def resolve_attn_impl(device_map) -> str:
    """Attention implementation for swift's ``get_model_tokenizer``.

    ``flash_attn`` is kept when the model lands on CUDA and the package is
    installed, so CUDA hosts behave as before. Anything else uses ``eager``.
    """
    global _FALLBACK_LOGGED
    target = str(device_map).split(":", 1)[0]
    if target == "auto":
        target = default_device_type()
    if target == "cuda" and importlib.util.find_spec("flash_attn") is not None:
        return "flash_attn"
    if not _FALLBACK_LOGGED:
        logger.warning(f"flash-attn is CUDA only, loading with eager attention on '{target}'")
        _FALLBACK_LOGGED = True
    return "eager"
