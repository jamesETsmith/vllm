# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types

import pytest
import torch

import vllm.model_executor.layers.sparse_attn_indexer as sparse_indexer
import vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse as rocm_sparse
from vllm.v1.attention.backends.mla.sparse_utils import (
    localize_dcp_global_topk_torch,
)


class _TwoTensorGather:
    def __init__(self, scores: torch.Tensor, ids: torch.Tensor) -> None:
        self._values = iter((scores, ids))

    def all_gather(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        assert dim == 1
        value = next(self._values)
        assert value.dtype == input_.dtype
        return value.clone()


def test_dcp_world_one_merge_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    indices = torch.tensor([[2, 0, -1]], dtype=torch.int32)
    original = indices.clone()
    monkeypatch.setattr(
        sparse_indexer,
        "get_dcp_group",
        lambda: pytest.fail("DCP=1 must not issue a collective"),
    )

    sparse_indexer._merge_dcp_topk_global(
        torch.tensor([[0.5, 0.1, 0.9]]),
        indices,
        topk_tokens=3,
        dcp_rank=0,
        dcp_world_size=1,
        cp_interleave=1,
    )

    torch.testing.assert_close(indices, original)


def test_portable_global_merge_two_ranks_short_rows_and_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rank_logits = [
        torch.tensor([[5.0, 4.0, 1.0], [3.0, 2.0, 1.0]]),
        torch.tensor([[5.0, 3.0, 2.0], [4.0, 0.0, -1.0]]),
    ]
    rank_topks = [
        torch.tensor([[0, 1, -1], [0, 1, 2]], dtype=torch.int32),
        torch.tensor([[0, 1, -1], [0, -1, -1]], dtype=torch.int32),
    ]
    packed = [
        sparse_indexer._make_dcp_topk_candidates_torch(
            rank_logits[rank],
            rank_topks[rank],
            rank,
            2,
            1,
        )
        for rank in range(2)
    ]
    gathered_scores = torch.cat([value[0] for value in packed], dim=1)
    gathered_ids = torch.cat([value[1] for value in packed], dim=1)
    # Equal score 5.0 selects lower global token id 0 before id 1.
    expected = torch.tensor([[0, 1, 2], [1, 0, 2]], dtype=torch.int32)

    for rank in range(2):
        monkeypatch.setattr(
            sparse_indexer,
            "get_dcp_group",
            lambda: _TwoTensorGather(gathered_scores, gathered_ids),
        )
        actual = rank_topks[rank].clone()
        sparse_indexer._merge_dcp_topk_global(
            rank_logits[rank],
            actual,
            topk_tokens=3,
            dcp_rank=rank,
            dcp_world_size=2,
            cp_interleave=1,
        )
        torch.testing.assert_close(actual, expected)


def test_candidate_row_starts_and_multiple_speculative_rows() -> None:
    logits = torch.tensor(
        [
            [99.0, 4.0, 3.0, 2.0],
            [99.0, 98.0, 7.0, 6.0],
        ]
    )
    indices = torch.tensor([[0, 1], [0, -1]], dtype=torch.int32)
    scores, global_ids = sparse_indexer._make_dcp_topk_candidates_torch(
        logits,
        indices,
        dcp_rank=1,
        dcp_world_size=2,
        cp_interleave=1,
        row_starts=torch.tensor([1, 2], dtype=torch.int32),
    )

    torch.testing.assert_close(scores, torch.tensor([[4.0, 3.0], [7.0, -torch.inf]]))
    torch.testing.assert_close(
        global_ids, torch.tensor([[1, 3], [1, -1]], dtype=torch.int32)
    )


def test_localize_compacts_uneven_ownership_and_empty_rows() -> None:
    req_id = torch.tensor([0, 1, 0], dtype=torch.int32)
    block_table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
    global_ids = torch.tensor(
        [
            [1, 4, 3, -1],  # rank 1 owns local ids 0 and 1
            [0, 2, 4, -1],  # rank 1 owns none
            [7, 5, 6, 1],  # rank 1 owns local ids 3, 2, 0
        ],
        dtype=torch.int32,
    )

    slots, counts = localize_dcp_global_topk_torch(
        req_id,
        block_table,
        global_ids,
        dcp_size=2,
        dcp_rank=1,
        block_size=2,
    )

    torch.testing.assert_close(counts, torch.tensor([2, 0, 3], dtype=torch.int32))
    torch.testing.assert_close(
        slots,
        torch.tensor(
            [
                [20, 21, -1, -1],
                [-1, -1, -1, -1],
                [23, 22, 20, -1],
            ],
            dtype=torch.int32,
        ),
    )


def test_localize_rejects_nontrivial_interleave() -> None:
    with pytest.raises(NotImplementedError, match="interleave_size=1"):
        localize_dcp_global_topk_torch(
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([[0]], dtype=torch.int32),
            torch.tensor([[0]], dtype=torch.int32),
            dcp_size=2,
            dcp_rank=0,
            block_size=1,
            cp_kv_cache_interleave_size=2,
        )


@pytest.mark.parametrize("supports_lse", [False, True])
def test_aiter_lse_capability_uses_runtime_signature(
    monkeypatch: pytest.MonkeyPatch,
    supports_lse: bool,
) -> None:
    aiter = types.ModuleType("aiter")
    aiter_mla = types.ModuleType("aiter.mla")

    def _with_lse(*args, return_lse=False, **kwargs):
        return None

    def _without_lse(*args, **kwargs):
        return None

    aiter_mla.mla_decode_fwd = _with_lse if supports_lse else _without_lse
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    monkeypatch.setitem(sys.modules, "aiter.mla", aiter_mla)
    rocm_sparse._aiter_mla_decode_supports_lse.cache_clear()
    try:
        assert rocm_sparse._aiter_mla_decode_supports_lse() is supports_lse
    finally:
        rocm_sparse._aiter_mla_decode_supports_lse.cache_clear()
