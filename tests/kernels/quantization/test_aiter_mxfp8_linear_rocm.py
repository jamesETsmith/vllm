# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gfx950 numerical and graph coverage for AITER MXFP8 dense linears."""

import pytest
import torch

from vllm._aiter_ops import is_aiter_found_and_supported
from vllm.model_executor.kernels.linear.mxfp8.aiter import (
    AiterMxfp8LinearKernel,
)
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
    Mxfp8LinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    _mxfp8_e4m3_quantize_torch,
    dequant_mxfp8_to_bf16,
    mxfp8_e4m3_quantize,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm()
    or not current_platform.supports_mx()
    or not is_aiter_found_and_supported(),
    reason="AITER MXFP8 linear requires AITER on CDNA4",
)

# Representative Hy4 dense projections at TP4. Hy4 stores independent E8M0
# scales for every output row, unlike DSV4.1's compact 32-row scale blocks.
HY4_DENSE_SHAPES = [
    ("q_a", 2048, 6144),
    ("q_b_tp4", 4096, 2048),
    ("dense_gate_up_tp4", 9216, 6144),
    ("dense_down_tp4", 6144, 4608),
]


def _make_layer(n: int, k: int) -> torch.nn.Module:
    weight_bf16 = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.05
    weight, weight_scale = _mxfp8_e4m3_quantize_torch(weight_bf16)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
    return layer


@pytest.mark.parametrize("name,n,k", HY4_DENSE_SHAPES, ids=lambda value: str(value))
@pytest.mark.parametrize("num_tokens", [1, 7, 32])
@torch.inference_mode()
def test_aiter_mxfp8_hy4_shapes_match_dequant_reference(name, n, k, num_tokens):
    torch.manual_seed(num_tokens)
    layer = _make_layer(n, k)
    kernel = AiterMxfp8LinearKernel(Mxfp8LinearLayerConfig(weight_shape=(n, k)))
    kernel.process_weights_after_loading(layer)
    assert layer.weight_scale.shape == (n, k // 32)

    x = torch.randn(num_tokens, k, device="cuda", dtype=torch.bfloat16) * 0.5
    out = kernel.apply_weights(layer, x)
    x_q, x_scale = mxfp8_e4m3_quantize(x)
    expected = (
        dequant_mxfp8_to_bf16(x_q, x_scale).float()
        @ dequant_mxfp8_to_bf16(layer.weight, layer.weight_scale).float().T
    )

    assert out.shape == (num_tokens, n)
    assert torch.isfinite(out).all(), f"{name}: non-finite output"
    rel = (out.float() - expected).norm() / expected.norm()
    assert rel < 5e-3, f"{name}: relative error {rel:.4f}"


@torch.inference_mode()
def test_aiter_mxfp8_fullgraph_dynamic_m_and_cudagraph():
    n, k = 2048, 6144
    layer = _make_layer(n, k)
    kernel = AiterMxfp8LinearKernel(Mxfp8LinearLayerConfig(weight_shape=(n, k)))
    kernel.process_weights_after_loading(layer)

    def run(inp: torch.Tensor) -> torch.Tensor:
        return kernel.apply_weights(layer, inp)

    compiled = torch.compile(run, dynamic=True, fullgraph=True)
    for m in (2, 17):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        torch.testing.assert_close(compiled(x), run(x), rtol=5e-3, atol=5e-3)

    x = torch.randn(8, k, device="cuda", dtype=torch.bfloat16)
    expected = run(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(x)
    graph.replay()
    torch.testing.assert_close(captured, expected, rtol=5e-3, atol=5e-3)
