# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER MXFP8 linear GEMM for AMD CDNA4 (gfx950).

The AITER group32 GEMM consumes native E4M3 values and E8M0 scales in
either per-row 1x32 or compact 32x32 weight-scale layouts. Under automatic
selection, per-row weights use AITER while compact weights retain vLLM's
shape-tuned block32 kernel. Explicit ``--linear-backend aiter`` routes both
layouts through AITER.
"""

from importlib.util import find_spec
from pathlib import Path

import torch
from torch.nn.parameter import Parameter

from vllm._aiter_ops import is_aiter_found_and_supported, rocm_aiter_ops
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_DTYPE,
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform

from .Mxfp8LinearKernel import Mxfp8LinearKernel, Mxfp8LinearLayerConfig
from .rocm_block32_gemm import rocm_mxfp8_block32_gemm
from .rocm_native import _as_block32_scale


def _group32_gemm_available() -> bool:
    """Check for the AITER group32 module without importing AITER.

    Importing AITER during backend discovery initializes HIP and can change
    engine process-spawn behavior. The dedicated module was introduced with
    the group32 public-dispatch contract, so its presence is a safe,
    side-effect-free capability check.
    """
    spec = find_spec("aiter")
    if spec is None or spec.submodule_search_locations is None:
        return False
    relative = Path("ops/triton/gemm/basic/gemm_a8w8_blockscale_group32.py")
    return any(
        (Path(root) / relative).is_file() for root in spec.submodule_search_locations
    )


class AiterMxfp8LinearKernel(Mxfp8LinearKernel):
    """Native MXFP8 linear using AITER's gfx950 group32 GEMM."""

    supports_pre_processed_weights = True

    def __init__(self, config: Mxfp8LinearLayerConfig) -> None:
        super().__init__(config)

        # Keep vLLM's faster shape-tuned compact-32x32 path under automatic
        # selection. An explicit backend request is useful for A/B validation
        # and routes compact weights through AITER as well.
        from vllm.model_executor.kernels.linear import _get_linear_backend

        self._aiter_compact_scales = (
            _get_linear_backend(quantization="mxfp8") == "aiter"
        )

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, "not ROCm"
        if not current_platform.supports_mx():
            return False, "native MX requires CDNA4 (gfx95x)"
        if not is_aiter_found_and_supported():
            return False, "AITER not found or not supported on the current platform"
        if not _group32_gemm_available():
            return False, "installed AITER does not provide the group32 MXFP8 GEMM"
        return True, None

    @classmethod
    def can_implement(cls, config: Mxfp8LinearLayerConfig) -> tuple[bool, str | None]:
        n, k = config.weight_shape
        if n <= 0 or k <= 0:
            return False, "weight dimensions must be positive"
        if k % MXFP8_BLOCK_SIZE != 0:
            return False, f"K must be divisible by {MXFP8_BLOCK_SIZE}"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight.data
        n, k = weight.shape
        scale_k = k // MXFP8_BLOCK_SIZE
        weight_scale = layer.weight_scale.data[:n, :scale_k].contiguous()

        # The loader expands compact checkpoint scales to one row per output
        # channel. Recover the compact representation for DSV4.1 and preserve
        # genuinely per-row scales for Hy4.
        block_scale = _as_block32_scale(weight_scale)
        if block_scale is not None:
            weight_scale = block_scale

        layer.weight = Parameter(weight.contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(weight_scale, requires_grad=False)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if layer.weight_scale.dtype != MXFP8_SCALE_DTYPE:
            raise ValueError(
                f"Expected {MXFP8_SCALE_DTYPE} weight_scale, got "
                f"{layer.weight_scale.dtype}."
            )

        out_shape = (*x.shape[:-1], layer.weight.shape[0])
        x2d = x.reshape(-1, x.shape[-1])
        compact_scales = layer.weight_scale.shape[0] != layer.weight.shape[0]
        x_q, x_scale = mxfp8_e4m3_quantize(x2d)

        if compact_scales and not self._aiter_compact_scales:
            out = rocm_mxfp8_block32_gemm(
                x_q, x_scale, layer.weight, layer.weight_scale, x.dtype
            )
        else:
            weight_group_rows = 32 if compact_scales else 1
            out = rocm_aiter_ops.gemm_a8w8_blockscale(
                x_q,
                layer.weight,
                x_scale,
                layer.weight_scale,
                [weight_group_rows, MXFP8_BLOCK_SIZE],
                x.dtype,
            )

        out = out.reshape(out_shape)
        if bias is not None:
            out = out + bias
        return out
