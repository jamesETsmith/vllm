# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contracts for the AITER MXFP8 linear adapter."""

from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    AiterMxfp8LinearKernel,
    RocmDotScaledMxfp8LinearKernel,
    init_mxfp8_linear_kernel,
)
from vllm.model_executor.kernels.linear.mxfp8 import aiter as aiter_linear
from vllm.platforms import PlatformEnum

pytestmark = pytest.mark.cpu_test


def _layer(n: int, k: int, *, compact: bool) -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.zeros((n, k), dtype=torch.float8_e4m3fn), requires_grad=False
    )
    scale_rows = n // 32 if compact else n
    scale = torch.arange(scale_rows, dtype=torch.uint8)[:, None].expand(
        scale_rows, k // 32
    )
    if compact:
        scale = scale.repeat_interleave(32, dim=0)
    layer.weight_scale = torch.nn.Parameter(scale.contiguous(), requires_grad=False)
    return layer


@pytest.mark.parametrize("backend", ["auto", "aiter"])
def test_aiter_mxfp8_selected_on_supported_rocm(backend):
    with (
        patch("vllm.model_executor.kernels.linear.current_platform") as platform,
        patch(
            "vllm.model_executor.kernels.linear._get_linear_backend",
            return_value=backend,
        ),
        patch.object(AiterMxfp8LinearKernel, "is_supported", return_value=(True, None)),
    ):
        platform._enum = PlatformEnum.ROCM
        kernel = init_mxfp8_linear_kernel(weight_shape=(64, 64))

    assert isinstance(kernel, AiterMxfp8LinearKernel)
    assert kernel._aiter_compact_scales is (backend == "aiter")


def test_aiter_mxfp8_falls_back_when_unavailable():
    with (
        patch("vllm.model_executor.kernels.linear.current_platform") as platform,
        patch(
            "vllm.model_executor.kernels.linear._get_linear_backend",
            return_value="auto",
        ),
        patch.object(
            AiterMxfp8LinearKernel,
            "is_supported",
            return_value=(False, "AITER unavailable"),
        ),
        patch.object(
            RocmDotScaledMxfp8LinearKernel,
            "is_supported",
            return_value=(True, None),
        ),
    ):
        platform._enum = PlatformEnum.ROCM
        kernel = init_mxfp8_linear_kernel(weight_shape=(64, 64))

    assert isinstance(kernel, RocmDotScaledMxfp8LinearKernel)


def test_aiter_mxfp8_rejects_older_aiter_without_group32():
    with (
        patch.object(aiter_linear.current_platform, "is_rocm", return_value=True),
        patch.object(aiter_linear.current_platform, "supports_mx", return_value=True),
        patch.object(aiter_linear, "is_aiter_found_and_supported", return_value=True),
        patch.object(aiter_linear, "_group32_gemm_available", return_value=False),
    ):
        supported, reason = AiterMxfp8LinearKernel.is_supported()

    assert not supported
    assert reason is not None and "group32" in reason


@pytest.mark.parametrize("compact", [False, True])
def test_process_weights_preserves_scale_layout(compact):
    n, k = 64, 64
    layer = _layer(n, k, compact=compact)
    kernel = object.__new__(AiterMxfp8LinearKernel)

    kernel.process_weights_after_loading(layer)

    expected_rows = n // 32 if compact else n
    assert layer.weight.shape == (n, k)
    assert layer.weight.is_contiguous()
    assert layer.weight_scale.shape == (expected_rows, k // 32)
    assert layer.weight_scale.is_contiguous()


def test_per_row_scales_use_aiter_and_restore_output_shape():
    n, k = 64, 64
    layer = _layer(n, k, compact=False)
    kernel = object.__new__(AiterMxfp8LinearKernel)
    kernel._aiter_compact_scales = False
    x = torch.zeros((2, 3, k), dtype=torch.bfloat16)
    x_q = torch.zeros((6, k), dtype=torch.float8_e4m3fn)
    x_scale = torch.zeros((6, k // 32), dtype=torch.uint8)
    gemm_out = torch.ones((6, n), dtype=torch.bfloat16)
    bias = torch.arange(n, dtype=torch.bfloat16)

    with (
        patch.object(aiter_linear, "mxfp8_e4m3_quantize", return_value=(x_q, x_scale)),
        patch.object(
            aiter_linear.rocm_aiter_ops,
            "gemm_a8w8_blockscale",
            return_value=gemm_out,
        ) as gemm,
    ):
        out = kernel.apply_weights(layer, x, bias)

    assert out.shape == (2, 3, n)
    torch.testing.assert_close(out, (gemm_out + bias).reshape(2, 3, n))
    gemm.assert_called_once_with(
        x_q,
        layer.weight,
        x_scale,
        layer.weight_scale,
        [1, 32],
        torch.bfloat16,
    )


@pytest.mark.parametrize("force_aiter", [False, True])
def test_compact_scales_preserve_auto_path_and_support_explicit_aiter(force_aiter):
    n, k = 64, 64
    layer = _layer(n, k, compact=True)
    kernel = object.__new__(AiterMxfp8LinearKernel)
    kernel._aiter_compact_scales = force_aiter
    kernel.process_weights_after_loading(layer)
    x = torch.zeros((3, k), dtype=torch.bfloat16)
    x_q = torch.zeros((3, k), dtype=torch.float8_e4m3fn)
    x_scale = torch.zeros((3, k // 32), dtype=torch.uint8)
    expected = torch.ones((3, n), dtype=torch.bfloat16)

    with (
        patch.object(aiter_linear, "mxfp8_e4m3_quantize", return_value=(x_q, x_scale)),
        patch.object(
            aiter_linear, "rocm_mxfp8_block32_gemm", return_value=expected
        ) as block32,
        patch.object(
            aiter_linear.rocm_aiter_ops,
            "gemm_a8w8_blockscale",
            return_value=expected,
        ) as gemm,
    ):
        out = kernel.apply_weights(layer, x)

    torch.testing.assert_close(out, expected)
    if force_aiter:
        block32.assert_not_called()
        gemm.assert_called_once_with(
            x_q,
            layer.weight,
            x_scale,
            layer.weight_scale,
            [32, 32],
            torch.bfloat16,
        )
    else:
        gemm.assert_not_called()
        block32.assert_called_once_with(
            x_q,
            x_scale,
            layer.weight,
            layer.weight_scale,
            torch.bfloat16,
        )
