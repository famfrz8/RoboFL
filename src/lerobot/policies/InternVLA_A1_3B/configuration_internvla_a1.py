#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Sequence

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES
from lerobot.transforms.core import *
from lerobot.policies.InternVLA_A1_3B.transform_internvla_a1 import Qwen3_VLProcessorTransformFn, UnifyQwenA1InputsTransformFn


@DatasetConfig.register_subclass("qwena1")
@dataclass
class QwenA1DatasetConfig(DatasetConfig):
    height: int = 224
    width: int = 224
    max_state_dim: int = 32
    max_action_dim: int = 32

    data_transforms: TransformGroup = field(
        default_factory=lambda: TransformGroup(
            inputs=[
                DeltaActionTransformFn(), 
                ResizeImagesWithPadFn(
                    height=QwenA1DatasetConfig.height, 
                    width=QwenA1DatasetConfig.width, 
                ),  # ✅
                RemapImageKeyTransformFn(),  # ✅
                Qwen3_VLProcessorTransformFn(), 
                NormalizeTransformFn(),
                ComposeFieldsTransform(),
                PadStateAndActionTransformFn(
                    max_state_dim=QwenA1DatasetConfig.max_state_dim, 
                    max_action_dim=QwenA1DatasetConfig.max_action_dim, 
                ),  # ✅
                UnifyQwenA1InputsTransformFn(), 
            ],
            outputs=[]
        )
    )

    def __post_init__(self):
        super().__post_init__()
        inputs = list(self.data_transforms.inputs)
        has_delta = any(isinstance(t, DeltaActionTransformFn) for t in inputs)
        if self.action_mode == "delta":
            if not has_delta:  # add DeltaActionTransformFn
                inputs = [DeltaActionTransformFn(), *inputs]
                self.data_transforms = replace(self.data_transforms, inputs=inputs)
        else:  # self.action_mode == "abs"
            if has_delta:  # remove DeltaActionTransformFn
                inputs = [t for t in inputs if not isinstance(t, DeltaActionTransformFn)]
                self.data_transforms = replace(self.data_transforms, inputs=inputs)


@PreTrainedConfig.register_subclass("qwena1")
@dataclass
class QwenA1Config(PreTrainedConfig):
    qwen3_vl_variant: str = "qwen3_vl_2b"
    action_expert_variant: str = "qwen3_600m"
    dtype: str = "bfloat16"  # Options: "bfloat16", "float32"

    n_obs_steps: int = 1
    chunk_size: int = 50  # Number of action steps to predict, in openpi called "action_horizon"
    n_action_steps: int = 50  # Number of action steps to execute

    # Shorter state and action vectors will be padded to these dimensions
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Flow matching parameters: see openpi `PI0Pytorch`
    num_inference_steps: int = 10  # Number of denoising steps during inference
    time_sampling_beta_alpha: float = 1.5
    time_sampling_beta_beta: float = 1.0
    time_sampling_scale: float = 0.999
    time_sampling_offset: float = 0.001
    min_period: float = 4e-3
    max_period: float = 4.0

    image_resolution: tuple[int, int] = (224, 224)  # see openpi `preprocessing_pytorch.py`
    image_history_stride: int = 15

    # Add empty images. Used to add empty cameras when no image features are present.
    empty_cameras: int = 0

    # Normalization
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # Training settings
    gradient_checkpointing: bool = False  # Enable gradient checkpointing for memory optimization
    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode
    device: str | None = None  # Device to use for the model (None = auto-detect)

    # Optimizer settings: see openpi `AdamW``
    optimizer_lr: float = 2.5e-5  # see openpi `CosineDecaySchedule: peak_lr`
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0

    # Scheduler settings: see openpi `CosineDecaySchedule`
    # Note: These will auto-scale if --steps < scheduler_decay_steps
    # For example, --steps=3000 will scale warmup to 100 and decay to 3000
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    tokenizer_max_length: int = 48  # see openpi `__post_init__`

    freeze_vision_encoder: bool = False
    train_expert_only: bool = False
    train_vlm_only: bool = False

    # LoRA settings
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "state_proj")

    # LoRA-MoE (Mixture of Experts) settings
    use_lora_moe: bool = False
    use_lora_moe_forced_last: bool = False
    loramoe_num_experts: int = 4
    loramoe_router_top_k: int = 2
    loramoe_router_hidden_dim: int = 512
    # Share-LoRA-A mode: shared lora_A with expert-specific lora_B branches.
    loramoe_enable_a_experts: bool = False
    loramoe_enable_b_experts: bool = True
    loramoe_share_a_across_experts: bool = True
    # AB routing: use separate routers for lora_A and lora_B
    loramoe_ab_routing: bool = False
    loramoe_router_top_k_a: int = 2  # top-k for lora_A router
    loramoe_router_top_k_b: int = 2  # top-k for lora_B router
    lambda_aux: float = 0.001  # weight for load-balancing (auxiliary) loss
    lambda_ot: float = 0.01  # weight for visual-token transport-consistency loss

    # TCR regularization for federated LoRA-MoE router training
    enable_tcr: bool = False
    tcr_lambda_proto: float = 0.5
    tcr_lambda_contrast: float = 0.5
    tcr_margin: float = 0.1
    tcr_tau_keep: float = 0.0
    tcr_use_gen_feature: bool = True
    tcr_prototype_momentum: float = 1.0
    tcr_enable_loss: bool = False
    tcr_loss_weight: float = 0.001

    # FedForesight: perception/foresight/action router alignment for FL LoRA-MoE.
    enable_fard: bool = False
    lambda_fard: float = 0.01
    fard_eps: float = 1e-8
    fard_warmup_rounds: int = 10
    fard_current_round: int = 0
    enable_pcea: bool = False

    # Lightweight affordance-conditioned action adaptation. The bridge uses
    # observed Cosmos features only; future features are used as a detached
    # training target and are never fed to the action path.
    enable_affordance: bool = False
    lambda_affordance: float = 0.001
    affordance_dim: int = 128
    # V2 adds task/motion context, gated conditioning, and action-horizon decay.
    # It is opt-in so existing Affordance V1 checkpoints keep their architecture.
    enable_affordance_v2: bool = False
    # V3 uses language-conditioned spatial affordance only as a bounded prior
    # over Action LoRA-MoE routers. It never modifies action tokens.
    enable_affordance_v3: bool = False
    affordance_v3_state_router_prior: bool = True
    # Disable the V1 state-router input residual for inference ablations while
    # keeping the affordance action residual enabled.
    affordance_state_router_input: bool = True
    affordance_action_horizon: int = 15
    affordance_action_decay_end: int = 30
    affordance_gate_init: float = 0.05
    # Keep uploaded LoRA experts fixed while the server calibrates routing and
    # shared affordance conditioning at the start of each MoE phase.
    affordance_router_lead_in_fraction: float = 0.05

    # Parameter-free router-guided visual token pruning.
    use_visual_token_prune: bool = True

    # Visual LoRA target modules (for und_expert.visual)
    # DISABLED due to OOM - visual has 24+ blocks with 4 modules each = 96+ LoRA layers
    # LoRA parameters require gradient computation, which causes OOM even with gradient checkpointing
    lora_visual_target_modules: tuple = ()

    # PTQ settings (native expert LLM only; excludes LoRA/LoRA-MoE)
    use_ptq: bool = False
    ptq_mode: str = "weight_only_int8"
    ptq_backend: str = "naive"
    ptq_experts: tuple = ("und", "gen", "act")
    ptq_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

    scale_factor: int = 8  # param for pixel shuffle / unshuffle
    lambda_gen: float = 0.01

    def __post_init__(self):
        super().__post_init__()

        if self.enable_affordance_v3:
            self.enable_affordance = True
            self.affordance_router_lead_in_fraction = 0.0

        # Validate configuration
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})"
            )

        if self.dtype not in ["bfloat16", "float32"]:
            raise ValueError(f"Invalid dtype: {self.dtype}")

        if self.ptq_mode not in ["weight_only_int8", "weight_only_int4"]:
            raise ValueError(f"Invalid ptq_mode: {self.ptq_mode}")

        if self.ptq_backend not in ["naive", "bnb_int8"]:
            raise ValueError(f"Invalid ptq_backend: {self.ptq_backend}")

        if self.ptq_backend == "bnb_int8" and self.ptq_mode != "weight_only_int8":
            raise ValueError("ptq_backend='bnb_int8' currently supports only ptq_mode='weight_only_int8'")

        if self.use_lora and (self.use_lora_moe or self.use_lora_moe_forced_last):
            self.use_lora = False

        enabled_lora_modes = sum(
            bool(flag)
            for flag in (
                self.use_lora,
                self.use_lora_moe,
                self.use_lora_moe_forced_last,
            )
        )
        if enabled_lora_modes > 1:
            raise ValueError(
                "use_lora, use_lora_moe, and use_lora_moe_forced_last are mutually exclusive"
            )

        if self.use_lora_moe_forced_last and self.loramoe_ab_routing:
            raise ValueError("use_lora_moe_forced_last does not support loramoe_ab_routing")

        if (self.enable_fard or self.enable_pcea) and (
            self.loramoe_share_a_across_experts or self.loramoe_ab_routing
        ):
            raise ValueError(
                "FARD/PCEA require standard single-router LoRA-MoE with complete A/B experts; "
                "set loramoe_share_a_across_experts=False and loramoe_ab_routing=False"
            )

        if (self.enable_fard or self.enable_pcea) and not (
            self.loramoe_enable_a_experts and self.loramoe_enable_b_experts
        ):
            raise ValueError(
                "FARD/PCEA require both loramoe_enable_a_experts=True and "
                "loramoe_enable_b_experts=True"
            )

        if self.lambda_fard < 0:
            raise ValueError(f"lambda_fard must be non-negative, got {self.lambda_fard}")

        if self.fard_eps <= 0:
            raise ValueError(f"fard_eps must be positive, got {self.fard_eps}")

        if self.fard_warmup_rounds < 0:
            raise ValueError(f"fard_warmup_rounds must be non-negative, got {self.fard_warmup_rounds}")

        if self.lambda_affordance < 0:
            raise ValueError(f"lambda_affordance must be non-negative, got {self.lambda_affordance}")
        if self.affordance_dim <= 0:
            raise ValueError(f"affordance_dim must be positive, got {self.affordance_dim}")
        if self.enable_affordance_v2 and not self.enable_affordance:
            raise ValueError("enable_affordance_v2=True requires enable_affordance=True")
        if self.enable_affordance_v3 and not self.use_lora_moe:
            raise ValueError("enable_affordance_v3=True requires use_lora_moe=True")
        if self.enable_affordance_v3 and self.loramoe_ab_routing:
            raise ValueError("enable_affordance_v3=True requires standard single-router LoRA-MoE")
        if self.enable_affordance_v2:
            if self.affordance_action_horizon < 0:
                raise ValueError(
                    f"affordance_action_horizon must be non-negative, got {self.affordance_action_horizon}"
                )
            if self.affordance_action_decay_end < self.affordance_action_horizon:
                raise ValueError(
                    "affordance_action_decay_end must be greater than or equal to "
                    f"affordance_action_horizon, got {self.affordance_action_decay_end} < "
                    f"{self.affordance_action_horizon}"
                )
            if self.affordance_action_decay_end > self.chunk_size:
                raise ValueError(
                    "affordance_action_decay_end cannot exceed chunk_size, got "
                    f"{self.affordance_action_decay_end} > {self.chunk_size}"
                )
            if not 0 < self.affordance_gate_init < 1:
                raise ValueError(
                    f"affordance_gate_init must be in (0, 1), got {self.affordance_gate_init}"
                )
        if not 0 <= self.affordance_router_lead_in_fraction < 1:
            raise ValueError(
                "affordance_router_lead_in_fraction must be in [0, 1), got "
                f"{self.affordance_router_lead_in_fraction}"
            )
    def validate_features(self) -> None:
        """Validate and set up input/output features."""
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, *self.image_resolution),  # Use configured image resolution
            )
            self.input_features[key] = empty_camera

        if "observation.state" not in self.input_features:
            state_feature = PolicyFeature(
                type=FeatureType.STATE,
                shape=(self.max_state_dim,),  # Padded to max_state_dim
            )
            self.input_features["observation.state"] = state_feature

        if "action" not in self.output_features:
            action_feature = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(self.max_action_dim,),  # Padded to max_action_dim
            )
            self.output_features["action"] = action_feature

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def image_delta_indices(self) -> list | None: 
        return [-self.image_history_stride, 0, self.image_history_stride]
