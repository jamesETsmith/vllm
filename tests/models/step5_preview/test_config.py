# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.step3p5 import Step3p5ForCausalLM, _get_rotary_dim
from vllm.models.step5_preview.config import (
    Step5PreviewConfig,
    Step5PreviewTextConfig,
    Step5PreviewVisionConfig,
)
from vllm.models.step5_preview.model import (
    MAX_DENSE_FALLBACK_MODEL_LEN,
    Step4ForCausalLM,
    _is_sparse_indexer_weight,
    _validate_dense_fallback,
)
from vllm.transformers_utils.config import get_config

pytestmark = pytest.mark.skip_global_cleanup


def test_step5_preview_config_parses_nested_checkpoint_configs():
    config = Step5PreviewConfig(
        architectures=["MMGPTStepRoboticsForCausalLM"],
        vision_config={
            "model_type": "perception_encoder",
            "image_size": 728,
            "patch_size": 14,
            "width": 1536,
            "layers": 47,
            "heads": 16,
            "output_dim": None,
            "use_cls_token": False,
            "ls_init_value": 0.1,
            "use_ln_post": False,
            "hidden_act": "quick_gelu",
        },
        text_config={
            "architectures": ["Step4ForCausalLM"],
            "model_type": "step4",
            "hidden_size": 4096,
            "num_hidden_layers": 92,
            "num_attention_heads": 64,
            "num_attention_groups": 4,
            "head_dim": 192,
            "partial_rotary_factors": [1.0, 1.0, 1.0, 1 / 3],
            "sparse_config": {"enabled": True, "topk": 512},
        },
    )

    assert isinstance(config.text_config, Step5PreviewTextConfig)
    assert config.text_config.architectures == ["Step4ForCausalLM"]
    assert config.text_config.partial_rotary_factors[-1] == pytest.approx(1 / 3)
    assert config.text_config.sparse_config["enabled"] is True

    assert isinstance(config.vision_config, Step5PreviewVisionConfig)
    assert config.vision_config.use_abs_posemb is True
    assert config.vision_config.use_rope2d is True
    assert config.vision_config.use_ln_pre is True
    assert config.vision_config.mlp_ratio == 4.0


def test_step5_preview_config_registry_resolves_local_checkpoint(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "architectures": ["MMGPTStepRoboticsForCausalLM"],
                "auto_map": {
                    "AutoConfig": "configuration_step_robotics.StepRoboticsConfig"
                },
                "model_type": "step3p5v",
                "vision_config": {
                    "model_type": "perception_encoder",
                    "width": 1536,
                    "layers": 47,
                    "heads": 16,
                },
                "text_config": {
                    "architectures": ["Step4ForCausalLM"],
                    "model_type": "step4",
                    "hidden_size": 4096,
                    "head_dim": 192,
                    "partial_rotary_factors": [1.0, 1.0, 1.0, 1 / 3],
                    "sparse_config": {"enabled": True},
                },
            }
        )
    )

    config = get_config(tmp_path, trust_remote_code=False)

    assert isinstance(config, Step5PreviewConfig)
    assert isinstance(config.text_config, Step5PreviewTextConfig)
    assert isinstance(config.vision_config, Step5PreviewVisionConfig)


@pytest.mark.parametrize(
    ("head_dim", "factor", "expected"),
    [
        (192, 1.0, 192),
        (192, 0.5, 96),
        (192, 1 / 3, 64),
    ],
)
def test_step3p5_rotary_dim_accepts_step5_factor(
    head_dim: int, factor: float, expected: int
):
    assert _get_rotary_dim(head_dim, factor) == expected


@pytest.mark.parametrize("factor", [0.0, -0.5, 0.3, 2.0])
def test_step3p5_rotary_dim_rejects_invalid_factor(factor: float):
    with pytest.raises(ValueError):
        _get_rotary_dim(192, factor)


def test_step5_dense_fallback_is_bounded():
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(sparse_config={"enabled": True}),
            max_model_len=MAX_DENSE_FALLBACK_MODEL_LEN,
        )
    )
    assert _validate_dense_fallback(vllm_config)

    vllm_config.model_config.max_model_len += 1
    with pytest.raises(ValueError, match="sparse-GQA is not implemented"):
        _validate_dense_fallback(vllm_config)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("model.layers.3.self_attn.sparse_indexer_q.weight", True),
        ("model.layers.3.self_attn.sparse_indexer_k_norm.bias", True),
        ("model.layers.3.self_attn.ssmax_s", True),
        ("model.layers.3.self_attn.q_proj.weight", False),
    ],
)
def test_step5_dense_fallback_filters_only_sparse_indexer_weights(
    name: str, expected: bool
):
    assert _is_sparse_indexer_weight(name) is expected


def test_step5_load_weights_omits_sparse_indexer_tensors(monkeypatch):
    received_names = []

    def fake_load_weights(self, weights):
        received_names.extend(name for name, _ in weights)
        return {"model.layers.3.self_attn.q_proj.weight"}

    monkeypatch.setattr(Step3p5ForCausalLM, "load_weights", fake_load_weights)
    model = object.__new__(Step4ForCausalLM)
    result = model.load_weights(
        [
            (
                "model.layers.3.self_attn.sparse_indexer_q.weight",
                torch.empty(1),
            ),
            ("model.layers.3.self_attn.ssmax_s", torch.empty(1)),
            ("model.layers.3.self_attn.q_proj.weight", torch.empty(1)),
        ]
    )

    assert received_names == ["model.layers.3.self_attn.q_proj.weight"]
    assert result == {"model.layers.3.self_attn.q_proj.weight"}
