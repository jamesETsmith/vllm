# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Initial Step 5 Preview model path.

This path brings up the released BF16 checkpoint by reusing vLLM's Step3p5
backbone and Step VL vision tower. The checkpoint's sparse-GQA indexer is not
implemented here yet, so full-attention layers deliberately use dense
attention at a bounded model length.
"""

from collections.abc import Iterable

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.step3p5 import Step3p5ForCausalLM
from vllm.model_executor.models.step_vl import StepVLForConditionalGeneration

logger = init_logger(__name__)

MAX_DENSE_FALLBACK_MODEL_LEN = 8192


def _uses_sparse_gqa(vllm_config: VllmConfig) -> bool:
    sparse_config = getattr(vllm_config.model_config.hf_config, "sparse_config", None)
    return isinstance(sparse_config, dict) and sparse_config.get("enabled", False)


def _validate_dense_fallback(vllm_config: VllmConfig) -> bool:
    if not _uses_sparse_gqa(vllm_config):
        return False

    max_model_len = vllm_config.model_config.max_model_len
    if max_model_len > MAX_DENSE_FALLBACK_MODEL_LEN:
        raise ValueError(
            "Step 5 Preview sparse-GQA is not implemented yet. "
            f"Set --max-model-len to {MAX_DENSE_FALLBACK_MODEL_LEN} or lower "
            "to use the initial dense-attention fallback."
        )
    return True


def _is_sparse_indexer_weight(name: str) -> bool:
    return ".sparse_indexer_" in name or name.endswith(".ssmax_s")


class Step4ForCausalLM(Step3p5ForCausalLM):
    """Step 5 text backbone with an explicit dense-attention fallback."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        if _validate_dense_fallback(vllm_config):
            logger.warning_once(
                "Step 5 Preview is using dense attention for sparse-GQA layers. "
                "Sparse-indexer weights are ignored and long-context behavior is "
                "not equivalent to the reference model."
            )

        super().__init__(vllm_config=vllm_config, prefix=prefix)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # Keep the unsupported weights out of the generic loader deliberately,
        # instead of relying on its unknown-parameter behavior.
        dense_fallback_weights = (
            (name, tensor)
            for name, tensor in weights
            if not _is_sparse_indexer_weight(name)
        )
        return super().load_weights(dense_fallback_weights)


class MMGPTStepRoboticsForCausalLM(StepVLForConditionalGeneration):
    """Step 5 Preview multimodal wrapper."""
