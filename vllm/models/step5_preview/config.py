# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration classes for the Step 5 Preview checkpoint."""

from typing import Any

from transformers import PretrainedConfig

from vllm.transformers_utils.configs.step3p5 import Step3p5Config


class Step5PreviewTextConfig(Step3p5Config):
    """Step 5 text-backbone config.

    The checkpoint calls this architecture ``Step4ForCausalLM``, but its
    dense, attention, and MoE weights are compatible with the Step3p5
    backbone. Sparse-GQA is intentionally handled by the Step 5 model path.
    """

    model_type = "step4"
    architectures = ["Step4ForCausalLM"]

    def __init__(
        self,
        partial_rotary_factors: list[float] | None = None,
        sparse_config: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        self.partial_rotary_factors = partial_rotary_factors
        self.sparse_config = sparse_config
        super().__init__(
            partial_rotary_factors=partial_rotary_factors,
            sparse_config=sparse_config,
            **kwargs,
        )


class Step5PreviewVisionConfig(PretrainedConfig):
    """Perception-encoder config used by Step 5 Preview."""

    model_type = "perception_encoder"

    def __init__(
        self,
        image_size: int = 728,
        patch_size: int = 14,
        width: int = 1536,
        layers: int = 47,
        heads: int = 16,
        output_dim: int | None = None,
        use_cls_token: bool = False,
        ls_init_value: float | None = 0.1,
        use_ln_post: bool = False,
        hidden_act: str = "quick_gelu",
        use_abs_posemb: bool = True,
        use_rope2d: bool = True,
        use_ln_pre: bool = True,
        mlp_ratio: float = 4.0,
        **kwargs,
    ) -> None:
        self.image_size = image_size
        self.patch_size = patch_size
        self.width = width
        self.layers = layers
        self.heads = heads
        self.output_dim = output_dim
        self.use_cls_token = use_cls_token
        self.ls_init_value = ls_init_value
        self.use_ln_post = use_ln_post
        self.hidden_act = hidden_act
        self.use_abs_posemb = use_abs_posemb
        self.use_rope2d = use_rope2d
        self.use_ln_pre = use_ln_pre
        self.mlp_ratio = mlp_ratio

        # Standard aliases used by generic Transformers/vLLM config helpers.
        self.hidden_size = width
        self.num_hidden_layers = layers
        self.num_attention_heads = heads

        super().__init__(**kwargs)


class Step5PreviewConfig(PretrainedConfig):
    """Top-level multimodal config for Step 5 Preview."""

    model_type = "step3p5v"
    architectures = ["MMGPTStepRoboticsForCausalLM"]
    is_composition = True
    sub_configs = {
        "text_config": Step5PreviewTextConfig,
        "vision_config": Step5PreviewVisionConfig,
    }

    def __init__(
        self,
        vision_config: dict | Step5PreviewVisionConfig | None = None,
        text_config: dict | Step5PreviewTextConfig | None = None,
        understand_projector_stride: int = 2,
        projector_bias: bool = False,
        image_token_id: int = 128001,
        **kwargs,
    ) -> None:
        if vision_config is None:
            vision_config = Step5PreviewVisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = Step5PreviewVisionConfig(**vision_config)
        self.vision_config = vision_config

        if text_config is None:
            text_config = Step5PreviewTextConfig()
        elif isinstance(text_config, dict):
            text_config = Step5PreviewTextConfig(**text_config)
        self.text_config = text_config

        self.understand_projector_stride = understand_projector_stride
        self.projector_bias = projector_bias
        self.image_token_id = image_token_id
        self.hidden_size = text_config.hidden_size

        super().__init__(**kwargs)

    def get_text_config(self, decoder: bool = False) -> Step5PreviewTextConfig:
        return self.text_config
