"""mlx-vlm adapter for ``prism_hadamard_qwen35`` packs (Prism Ternary Bonsai 2).

These packs are a stock mlx-vlm ``qwen3_5`` graph whose language-model
projections are stored in a rotated (Hadamard) basis at 2-bit/group-128 affine
precision.  Ordinary loaders cannot read them because the required activation
transform lives only in the pack's own ``runtime/`` directory, and because
``model_type`` deliberately does not name a known architecture.

This module makes ``mlx_vlm.utils.get_model_and_args`` resolve that model type
to the stock qwen3_5 model with ``Packed`` layers installed, so the normal
``load_model`` path (including strict ``load_weights``) just works.

``fwht`` and ``Packed`` are ported from the pack's bundled ``runtime.py``
(MIT, Copyright (c) 2023 Apple Inc.) so the numerics stay identical.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

import mlx.core as mx
from mlx import nn

from .qwen3_5 import LanguageModel, VisionModel  # noqa: F401  (re-exported)
from .qwen3_5.config import ModelConfig as _Qwen35ModelConfig
from .qwen3_5.config import TextConfig, VisionConfig  # noqa: F401  (re-exported)
from .qwen3_5.qwen3_5 import Model as _Qwen35Model

PACK_MODEL_TYPE = "prism_hadamard_qwen35"
BITS = 2
GROUP_SIZE = 128
VALUES_PER_WORD = 32 // BITS
SUPPORTED_BLOCKS = (512, 1024, 2048, 4096)


def fwht(x, block, signs, inverse=False):
    shape, dtype = x.shape, x.dtype
    if shape[-1] % block:
        raise ValueError("Hadamard block does not divide activation width")
    x = x.astype(mx.float32)
    if not inverse:
        x = x * signs
    x = mx.hadamard_transform(x.reshape(-1, block), scale=1 / math.sqrt(block)).reshape(
        shape
    )
    if inverse:
        x = x * signs
    return x.astype(dtype)


class Packed(nn.Module):
    """A 2-bit affine projection whose activations carry a Hadamard rotation."""

    def __init__(self, arrays, block=0, signs=None, embedding=False, dtype=mx.float16):
        super().__init__()
        self.weight, self.scales, self.biases = [mx.array(a) for a in arrays]
        self.block, self.embedding, self.dtype = block, embedding, dtype
        if signs is not None:
            self.signs = signs

    def __call__(self, x):
        if self.embedding:
            shape = x.shape
            indices = x.reshape(-1)
            out = (
                mx.dequantize(
                    self.weight[indices],
                    self.scales[indices],
                    self.biases[indices],
                    group_size=GROUP_SIZE,
                    bits=BITS,
                )
                .reshape(*shape, -1)
                .astype(self.dtype)
            )
            return (
                fwht(out, self.block, self.signs, inverse=True) if self.block else out
            )
        if self.block:
            x = fwht(x, self.block, self.signs)
        return mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=GROUP_SIZE,
            bits=BITS,
        )


def _placeholder(original, record):
    """Build a correctly shaped, empty Packed layer for strict load_weights."""
    if record.get("dtype") != "float16":
        raise ValueError(f"Unsupported activation dtype for {record['path']}")
    if not isinstance(original, (nn.Linear, nn.Embedding)):
        raise ValueError(f"Unsupported packed module target at {record['path']}")
    if bool(record["embedding"]) != isinstance(original, nn.Embedding):
        raise ValueError(f"Packed module kind mismatch at {record['path']}")

    rows, width = original.weight.shape
    if width % GROUP_SIZE:
        raise ValueError(f"Invalid packed width at {record['path']}")

    block = record["block"]
    if block and block not in SUPPORTED_BLOCKS:
        raise ValueError(f"Unsupported block size at {record['path']}")
    if block and width % block:
        raise ValueError(f"Invalid transform dimensions at {record['path']}")

    arrays = (
        mx.zeros((rows, width // VALUES_PER_WORD), dtype=mx.uint32),
        mx.zeros((rows, width // GROUP_SIZE), dtype=mx.float16),
        mx.zeros((rows, width // GROUP_SIZE), dtype=mx.float16),
    )
    signs = mx.ones((width,), dtype=mx.float32) if block else None
    return Packed(arrays, block, signs, bool(record["embedding"]), mx.float16)


def install_packed_modules(language_model, records):
    seen = set()
    for record in records:
        path = record["path"]
        if path in seen:
            raise ValueError(f"Duplicate packed module {path}")
        seen.add(path)
        parts = path.split(".")
        parent = language_model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        setattr(parent, parts[-1], _placeholder(getattr(parent, parts[-1]), record))
    return len(seen)


@dataclass
class ModelConfig(_Qwen35ModelConfig):
    # ``modules`` from the pack config, renamed so it cannot shadow anything.
    packed_modules: List[Dict[str, Any]] = field(default_factory=list)
    base_model_type: str = "qwen3_5"
    components: Dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        params.setdefault("packed_modules", params.get("modules") or [])
        return super().from_dict(params)


class Model(_Qwen35Model):
    def __init__(self, config: ModelConfig):
        if getattr(config, "base_model_type", "qwen3_5") != "qwen3_5":
            raise ValueError(
                f"Unsupported base model type {config.base_model_type!r}"
            )
        super().__init__(config)
        install_packed_modules(
            self.language_model, getattr(config, "packed_modules", None) or []
        )


def _register_prompt_format():
    """Teach mlx-vlm to build qwen3_5-style image messages for this type."""
    try:
        from .. import prompt_utils
    except Exception:
        return
    table = getattr(prompt_utils, "MODEL_CONFIG", None)
    if isinstance(table, dict) and "qwen3_5" in table:
        table.setdefault(PACK_MODEL_TYPE, table["qwen3_5"])


def _register_processor():
    try:
        from .base import install_auto_processor_patch
        from .qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    except Exception:
        return
    install_auto_processor_patch(PACK_MODEL_TYPE, Qwen3VLProcessor)


_register_prompt_format()
_register_processor()
