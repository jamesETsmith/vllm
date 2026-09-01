# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.v1.attention.backend import AttentionLayer
from vllm.v1.attention.backends.mla.rocm_aiter_mla import AiterMLAHelper
from vllm.v1.attention.backends.mla.rocm_aiter_mla_sparse import (
    ROCMAiterMLASparseBackend,
    ROCMAiterMLASparseImpl,
    ROCMAiterMLASparseMetadata,
    ROCMAiterMLASparseMetadataBuilder,
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    rocm_sparse_attn_decode_fp8_ds_mla,
)


class HYV4ROCMAiterMLASparseImpl(ROCMAiterMLASparseImpl):
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: ROCMAiterMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.kv_cache_dtype != "fp8_ds_mla":
            return super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)

        if isinstance(q, tuple):
            ql_nope, q_pe = q
            q = self.q_concat_buffer[: ql_nope.shape[0]]
            ops.concat_mla_q(ql_nope, q_pe, q)

        num_actual_toks = attn_metadata.num_actual_tokens
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]
        triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            attn_metadata.paged_kv_indptr,
            attn_metadata.paged_kv_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            BLOCK_STRIDE_ROWS=(
                kv_c_and_k_pe_cache.stride(0)
                // int(np.prod(kv_c_and_k_pe_cache.shape[2:]))
            ),
            NUM_TOPK_TOKENS=attn_metadata.topk_tokens,
        )

        q = AiterMLAHelper.get_mla_padded_q(self.num_heads, q)
        sinks = self.sinks
        if sinks is not None and q.shape[1] > sinks.shape[0]:
            repeats = q.shape[1] // sinks.shape[0]
            sinks = sinks.repeat_interleave(repeats)
        output = torch.zeros(
            (q.shape[0], q.shape[1], self.kv_lora_rank),
            dtype=attn_metadata.attn_out_dtype,
            device=q.device,
        )
        rocm_sparse_attn_decode_fp8_ds_mla(
            q=q,
            kv_cache=kv_c_and_k_pe_cache,
            indices=attn_metadata.paged_kv_indices,
            indptr=attn_metadata.paged_kv_indptr,
            attn_sink=sinks,
            scale=self.scale,
            output=output,
        )
        output = AiterMLAHelper.get_mla_unpadded_o(self.num_heads, output)
        return output, None


class HYV4ROCMAiterMLASparseMetadataBuilder(ROCMAiterMLASparseMetadataBuilder):
    supports_draft_decode_metadata_update = True
    use_persistent_mla_metadata = False


class HYV4ROCMAiterMLASparseBackend(ROCMAiterMLASparseBackend):
    use_fp8_ds_mla_layout = True
    supported_kv_cache_dtypes = [
        *ROCMAiterMLASparseBackend.supported_kv_cache_dtypes,
        "fp8_ds_mla",
    ]

    @staticmethod
    def get_builder_cls() -> type[HYV4ROCMAiterMLASparseMetadataBuilder]:
        return HYV4ROCMAiterMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type[HYV4ROCMAiterMLASparseImpl]:
        return HYV4ROCMAiterMLASparseImpl

    @classmethod
    def supports_sink(cls) -> bool:
        return True
