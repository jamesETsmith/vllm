# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split-KV sliding-window attention used by the ROCm Beam DFlash2 drafter."""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")


def _reference(q, kv, query_start_loc, seq_lens, block_table, scale, window, causal):
    num_heads, head_size = q.shape[1], q.shape[2]
    page_size, num_kv_heads = kv.shape[2], kv.shape[1]
    out = torch.zeros_like(q, dtype=torch.float32)
    for seq in range(seq_lens.numel()):
        start, end = int(query_start_loc[seq]), int(query_start_loc[seq + 1])
        seq_len = int(seq_lens[seq])
        pos = torch.arange(seq_len, device=q.device)
        pages = block_table[seq, pos // page_size].long()
        rows = kv[pages, :, pos % page_size].float()  # [seq_len, Hkv, 2D]
        rep = num_heads // num_kv_heads
        k = rows[..., :head_size].repeat_interleave(rep, 1)
        v = rows[..., head_size:].repeat_interleave(rep, 1)
        q_pos = seq_len - (end - start) + torch.arange(end - start, device=q.device)
        dist = q_pos[:, None] - pos[None, :]
        visible = torch.ones_like(dist, dtype=torch.bool)
        if causal:
            visible &= dist >= 0
        if window:
            visible &= dist.abs() < window
        s = torch.einsum("qhd,khd->hqk", q[start:end].float(), k) * scale
        s = s.masked_fill(~visible[None], float("-inf"))
        out[start:end] = torch.einsum("hqk,khd->qhd", s.softmax(-1), v)
    return out


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [0, 4096, 37])
@pytest.mark.parametrize("query_lens", [[6, 6, 6], [1, 16, 3, 40]])
@pytest.mark.parametrize("hnd", [True, False])
# Per-rank drafter heads at TP8 and TP4.
@pytest.mark.parametrize("heads", [(5, 1), (10, 2)])
def test_split_kv_matches_reference(causal, window, query_lens, hnd, heads):
    from vllm.models.beam.amd.dflash2_attention_ops import (
        dflash2_split_kv_attention,
        split_kv_workspace,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    (num_heads, num_kv_heads), head_size, page_size = heads, 128, 16
    context_lens = [9000, 30, 5000, 0][: len(query_lens)]
    seq_lens = [c + q for c, q in zip(context_lens, query_lens)]
    max_pages = (max(seq_lens) + page_size - 1) // page_size
    num_blocks = len(seq_lens) * max_pages + 1
    shape = (num_blocks, page_size, num_kv_heads, 2 * head_size)
    kv = torch.randn(shape, device=device, dtype=torch.bfloat16).transpose(1, 2)
    if hnd:
        kv = kv.contiguous()
    block_table = (
        (torch.randperm(num_blocks - 1, device=device)[: len(seq_lens) * max_pages] + 1)
        .view(len(seq_lens), max_pages)
        .int()
    )
    lens = torch.tensor(query_lens, device=device)
    query_start_loc = torch.zeros(len(query_lens) + 1, device=device, dtype=torch.int32)
    query_start_loc[1:] = lens.cumsum(0)
    seq_lens_t = torch.tensor(seq_lens, device=device, dtype=torch.int32)
    num_tokens = sum(query_lens)
    q = torch.randn(
        num_tokens, num_heads, head_size, device=device, dtype=torch.bfloat16
    )
    out = torch.full_like(q, float("nan"))
    scale = head_size**-0.5

    dflash2_split_kv_attention(
        q,
        kv,
        out,
        query_start_loc,
        seq_lens_t,
        block_table,
        len(query_lens),
        max(query_lens),
        scale,
        window,
        causal,
        split_kv_workspace(4096, num_heads, head_size, device),
        512,
    )

    expected = _reference(
        q, kv, query_start_loc, seq_lens_t, block_table, scale, window, causal
    )
    torch.testing.assert_close(out.float(), expected, rtol=2e-2, atol=2e-2)
