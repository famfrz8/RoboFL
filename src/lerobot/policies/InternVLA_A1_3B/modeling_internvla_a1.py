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

import logging
import math
import os
import weakref
from collections import deque
from pathlib import Path
from types import MethodType
from typing import Literal
import os
# Read the env var; fall back to the default cache path

from huggingface_hub import snapshot_download
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
import torch._dynamo as dynamo
from einops import rearrange

from transformers.models.auto import CONFIG_MAPPING
from transformers.models.qwen3_vl import modeling_qwen3_vl
from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLTextModel

try:
    from peft import LoraConfig, get_peft_model, PeftModel
    PEFt_AVAILABLE = True
except ImportError:
    PEFt_AVAILABLE = False
    LoraConfig = None
    get_peft_model = None
    PeftModel = None

from lerobot.policies.InternVLA_A1_3B.cosmos_tokenizer.image_lib import ImageTokenizer
from lerobot.policies.InternVLA_A1_3B.configuration_internvla_a1 import QwenA1Config
# from lerobot.policies.InternVLA_A1_3B.ptq import quantize_native_expert_llms
from lerobot.policies.InternVLA_A1_3B.visual_token_prune import (
    apply_visual_token_prune_to_loramoe_modules,
    build_visual_token_prune_plan,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.utils import format_big_number
from lerobot.utils.affordance_utils import (
    factorized_future_effect_cosine_loss,
    factorized_signed_future_effect_cosine_loss,
    make_affordance_action_horizon_mask,
    masked_spatial_softmax,
    masked_token_mean,
)
from lerobot.utils.fedforesight_utils import foresight_consensus, three_path_consensus
from lerobot.utils.constants import (
    HF_HOME,
    ACTION,
    OBS_STATE,
    OBS_PREFIX,
    OBS_IMAGES,
    OPENPI_ATTENTION_MASK_VALUE,
)


def _print_visual_token_prune_plan_status(
    *,
    phase: str,
    plan,
    cfg,
    lang_tokens: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    image_token_id: int,
) -> None:
    print_enabled = os.environ.get("USE_VISUAL_TOKEN_PRUNE_PRINT_STATS", "false").lower() not in {
        "0",
        "false",
        "no",
        "off",
        "",
    }
    if not print_enabled or int(os.environ.get("LOCAL_RANK", "0")) != 0:
        return

    prune_enabled = bool(getattr(cfg, "use_visual_token_prune", False))
    if not prune_enabled:
        reason = "disabled"
    elif plan is None:
        reason = "no_visual_tokens"
    else:
        reason = "ready"

    detected_visual_counts = (lang_tokens == image_token_id).sum(dim=1)
    for batch_idx in range(lang_tokens.shape[0]):
        visual_before = int(detected_visual_counts[batch_idx].item())
        prefix_before = int(prefix_pad_masks.shape[1])
        prefix_valid = int(prefix_pad_masks[batch_idx].sum().item())
        if plan is None:
            print(
                "[VisualTokenPrunePlan] "
                f"phase={phase} batch={batch_idx} enabled={str(prune_enabled).lower()} "
                f"plan_built=false reason={reason} "
                f"visual_tokens={visual_before}->{visual_before} "
                f"prefix_tokens={prefix_before}->{prefix_before} prefix_valid={prefix_valid}",
                flush=True,
            )
            continue

        plan_visual_count = int(plan.visual_token_counts[batch_idx])
        print(
            "[VisualTokenPrunePlan] "
            f"phase={phase} batch={batch_idx} enabled=true plan_built=true reason=ready "
            f"visual_tokens={plan_visual_count} selection=router_effective_count "
            f"prefix_tokens={prefix_before} prefix_valid={prefix_valid} "
            "reduction=pending_router_selection",
            flush=True,
        )


hf_home_path = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
cosmos_tokenizer_path = Path(hf_home_path) / "hub" / "Cosmos-Tokenizer-CI8x8"
if not cosmos_tokenizer_path.is_dir() or not any(cosmos_tokenizer_path.iterdir()):
    snapshot_download(
        repo_id="nvidia/Cosmos-Tokenizer-CI8x8",
        local_dir=cosmos_tokenizer_path,
    )

def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "mps" and target_dtype == torch.float64:
        return torch.float32
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(  # see openpi `create_sinusoidal_pos_embedding` (exact copy)
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):  # see openpi `sample_beta` (exact copy)
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):  # see openpi `make_att_2d_masks` (exact copy)
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


def pad_vector(vector, new_dim):
    """Pad the last dimension of a vector to new_dim with zeros.

    Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] >= new_dim:
        return vector
    return F.pad(vector, (0, new_dim - vector.shape[-1]))


# Define the complete layer computation function for gradient checkpointing
def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, und_expert, gen_expert, act_expert):
    # Handle PEFT wrapping: get the underlying language_model for rotary_emb
    # und_expert = Qwen3VLForConditionalGeneration (wrapped by PEFT or not)
    # und_expert.language_model = Qwen3VLTextModel (also wrapped by PEFT if LoRA enabled)
    if hasattr(und_expert, 'base_model'):
        # PEFT wrapped: und_expert -> base_model (original ForCausalLM)
        # -> model (Qwen3VLModel) -> language_model (Qwen3VLTextModel)
        temp = und_expert.base_model
        # Check if temp has .model (Qwen3VLForCausalLM style) or is directly Qwen3VLModel
        if hasattr(temp, 'model'):
            temp = temp.model  # Qwen3VLModel
        if hasattr(temp, 'language_model'):
            und_expert_for_rope = temp.language_model
            # If language_model is also wrapped by PEFT, unwrap it
            if hasattr(und_expert_for_rope, 'base_model'):
                # Get the underlying Qwen3VLTextModel from PEFT
                und_expert_for_rope = und_expert_for_rope.base_model
                if hasattr(und_expert_for_rope, 'model'):
                    und_expert_for_rope = und_expert_for_rope.model
        else:
            und_expert_for_rope = temp
    else:
        und_expert_for_rope = und_expert.model.language_model

    # For the models list, we need to handle LoRA wrapping too
    models = []
    for exp in [und_expert, gen_expert, act_expert]:
        if hasattr(exp, 'base_model'):
            # PEFT wrapped - get the underlying model
            exp_base = exp.base_model
            if hasattr(exp_base, 'model'):
                # Check if this is ForCausalLM style (has .language_model inside)
                if hasattr(exp_base.model, 'language_model'):
                    models.append(exp_base.model.language_model)
                else:
                    models.append(exp_base.model)
            else:
                models.append(exp_base)
        else:
            if hasattr(exp, 'language_model'):
                models.append(exp.language_model)
            else:
                models.append(exp)
    query_states = []
    key_states = []
    value_states = []
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states = layer.input_layernorm(hidden_states)  # noqa: PLW2901
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
        if layer.self_attn.q_proj.weight.dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype=torch.bfloat16)
        query_state = layer.self_attn.q_norm(layer.self_attn.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_state = layer.self_attn.k_norm(layer.self_attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)
    # Concatenate and process attention
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)
    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = und_expert_for_rope.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_qwen3_vl.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )
    batch_size = query_states.shape[0]
    # Use models[0] which is the properly unwrapped language_model
    und_lang_model = models[0]
    scaling = und_lang_model.layers[layer_idx].self_attn.scaling
    # Attention computation
    att_output, _ = modeling_qwen3_vl.eager_attention_forward(
        und_lang_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )
    # Get head_dim from the current layer, not from the model
    head_dim = und_lang_model.layers[layer_idx].self_attn.head_dim
    num_attention_heads = und_lang_model.layers[layer_idx].self_attn.config.num_attention_heads
    att_output = att_output.reshape(batch_size, -1, 1 * num_attention_heads * head_dim)
    # Process layer outputs
    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        # first residual
        out_emb = out_emb + hidden_states
        after_first_residual = out_emb.clone()
        out_emb = layer.post_attention_layernorm(out_emb)
        # Convert to bfloat16 if the next layer (mlp) uses bfloat16
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        # second residual
        out_emb = out_emb + after_first_residual
        outputs_embeds.append(out_emb)
        start_pos = end_pos
    return outputs_embeds


class QwenConfig:
    """Configuration for Qwen model variants."""

    def __init__(self, head_dim, hidden_size, intermediate_size, num_attention_heads, num_hidden_layers, num_key_value_heads):
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads


def get_qwen_config(variant: str) -> QwenConfig:  # see openpi `gemma.py: get_config`
    """Returns config for specified gemma variant."""
    num_hidden_layers = int(variant.split('_')[-1][:-1])  # pattern: qwen3_vl_28l or qwen3_xxl
    if variant.startswith("qwen3_vl"):
        return QwenConfig(
            head_dim=128,
            hidden_size=2048,
            intermediate_size=6144,
            num_attention_heads=16,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=8,
        )
    elif variant.startswith("qwen3"):
        return QwenConfig(
            head_dim=128,
            hidden_size=1024,
            intermediate_size=3072,
            num_attention_heads=16,
            num_hidden_layers=num_hidden_layers,
            num_key_value_heads=8,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")


class Qwen3VLWithExpertModel(
    nn.Module
):
    """Qwen3_VL model with action expert for QwenA1."""

    def __init__(
        self,
        vlm_config,
        action_expert_config,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["qwen3_vl"]()
        vlm_config_hf.text_config.hidden_size = vlm_config.hidden_size
        vlm_config_hf.text_config.intermediate_size = vlm_config.intermediate_size
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_attention_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.num_hidden_layers
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_key_value_heads
        vlm_config_hf.text_config.max_position_embeddings = 262144
        vlm_config_hf.text_config.rope_scaling = {
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
            "rope_type": "default"
        }
        vlm_config_hf.text_config.tie_word_embeddings = True
        vlm_config_hf.tie_word_embeddings = True
        vlm_config_hf.vision_config.deepstack_visual_indexes=[5, 11, 17]
        vlm_config_hf.vision_config.depth=24
        vlm_config_hf.vision_config.hidden_size=1024
        vlm_config_hf.vision_config.intermediate_size=4096
        vlm_config_hf.vision_config.out_hidden_size=2048

        # self.und_expert = Qwen3VLForConditionalGeneration(config=vlm_config_hf)
        self.und_expert = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-2B-Instruct",
            config=vlm_config_hf,
            ignore_mismatched_sizes=True
        )

        gen_expert_config_hf = CONFIG_MAPPING["qwen3_vl_text"]()
        gen_expert_config_hf.head_dim=action_expert_config.head_dim
        gen_expert_config_hf.hidden_size=action_expert_config.hidden_size
        gen_expert_config_hf.intermediate_size=action_expert_config.intermediate_size
        gen_expert_config_hf.num_attention_heads=action_expert_config.num_attention_heads
        gen_expert_config_hf.num_hidden_layers=action_expert_config.num_hidden_layers
        gen_expert_config_hf.num_key_value_heads=action_expert_config.num_key_value_heads
        gen_expert_config_hf.max_position_embeddings = self.und_expert.config.text_config.max_position_embeddings
        gen_expert_config_hf.rope_scaling = self.und_expert.config.text_config.rope_scaling
        self.gen_expert = Qwen3VLTextModel(config=gen_expert_config_hf)
        self.gen_expert.embed_tokens = None
        self.gen_expert.lm_head = None

        action_expert_config_hf = CONFIG_MAPPING["qwen3_vl_text"]()
        action_expert_config_hf.head_dim=action_expert_config.head_dim
        action_expert_config_hf.hidden_size=action_expert_config.hidden_size
        action_expert_config_hf.intermediate_size=action_expert_config.intermediate_size
        action_expert_config_hf.num_attention_heads=action_expert_config.num_attention_heads
        action_expert_config_hf.num_hidden_layers=action_expert_config.num_hidden_layers
        action_expert_config_hf.num_key_value_heads=action_expert_config.num_key_value_heads
        action_expert_config_hf.max_position_embeddings = self.und_expert.config.text_config.max_position_embeddings
        action_expert_config_hf.rope_scaling = self.und_expert.config.text_config.rope_scaling
        self.act_expert = Qwen3VLTextModel(config=action_expert_config_hf)
        self.act_expert.embed_tokens = None
        self.act_expert.lm_head = None

        assert self.und_expert.config.text_config.num_hidden_layers == self.act_expert.config.num_hidden_layers

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            # "visual.patch_embed.proj.weight",
            # "visual.patch_embed.proj.bias",
            # "visual.pos_embed.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
    ):
        if inputs_embeds[1] is None and inputs_embeds[2] is None:
            prefix_output = self.und_expert.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            middle_output = None
            suffix_output = None

        elif inputs_embeds[0] is None and inputs_embeds[2] is None:
            middle_output = self.gen_expert.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            past_key_values = middle_output.past_key_values
            prefix_output = None
            middle_output = middle_output.last_hidden_state
            suffix_output = None

        elif inputs_embeds[0] is None and inputs_embeds[1] is None:
            suffix_output = self.act_expert.forward(
                inputs_embeds=inputs_embeds[2],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            past_key_values = None
            prefix_output = None
            middle_output = None
            suffix_output = suffix_output.last_hidden_state
        else:
            models = [self.und_expert.language_model, self.gen_expert, self.act_expert]
            num_layers = self.und_expert.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.act_expert, "gradient_checkpointing")
                and self.gen_expert.gradient_checkpointing
                and self.act_expert.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)
            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        use_reentrant=False,
                        preserve_rng_state=False,
                        und_expert=self.und_expert,
                        gen_expert=self.gen_expert,
                        act_expert=self.act_expert,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        und_expert=self.und_expert,
                        gen_expert=self.gen_expert,
                        act_expert=self.act_expert,
                    )

            # final norm
            def compute_final_norms(inputs_embeds):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb = models[i].norm(hidden_states)
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms,
                    inputs_embeds,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds)

            past_key_values = None
            prefix_output = outputs_embeds[0]
            middle_output = outputs_embeds[1]
            suffix_output = outputs_embeds[2]

        return [prefix_output, middle_output, suffix_output], past_key_values


class QwenA1(nn.Module):

    def __init__(self, config: QwenA1Config):
        super().__init__()
        self.config = config

        vlm_config = get_qwen_config(config.qwen3_vl_variant)
        action_expert_config = get_qwen_config(config.action_expert_variant)

        self.qwen3_vl_with_expert = Qwen3VLWithExpertModel(
            vlm_config,
            action_expert_config,
            precision=config.dtype,
        )

        if not os.path.exists(f"{HF_HOME}/hub/Cosmos-Tokenizer-CI8x8/encoder.jit"):
            logging.warning(f"Cosmos-Tokenizer-CI8x8 not found, downloading...")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="nvidia/Cosmos-Tokenizer-CI8x8", local_dir=f"{HF_HOME}/hub/Cosmos-Tokenizer-CI8x8")

        self.cosmos = ImageTokenizer(
            checkpoint_enc=f"{HF_HOME}/hub/Cosmos-Tokenizer-CI8x8/encoder.jit",
            checkpoint_dec=f"{HF_HOME}/hub/Cosmos-Tokenizer-CI8x8/decoder.jit",
        )

        vae_dim = 16
        gen_proj_dim = action_expert_config.hidden_size
        ds = self.config.scale_factor
        # self.downsample_conv = nn.Conv2d(in_channels=vae_dim, out_channels=gen_proj_dim, kernel_size=ds, stride=ds, padding=0) # junhao
        # self.cosmos_in_proj = nn.Conv2d(in_channels=gen_proj_dim, out_channels=vlm_config.hidden_size, kernel_size=1, stride=1, padding=0) # junhao
        # self.cosmos_out_proj = nn.Conv2d(in_channels=vlm_config.hidden_size, out_channels=gen_proj_dim, kernel_size=1, stride=1, padding=0) # junhao
        # self.upsample_conv = nn.ConvTranspose2d(in_channels=gen_proj_dim, out_channels=vae_dim, kernel_size=ds, stride=ds, padding=0, output_padding=0) # junhao

        self.cosmos_in_proj = nn.Conv2d(in_channels=vae_dim, out_channels=gen_proj_dim, kernel_size=1, stride=1, padding=0) # jia
        self.downsample_conv = nn.Conv2d(in_channels=gen_proj_dim, out_channels=gen_proj_dim, kernel_size=ds, stride=ds, padding=0) # jia
        self.upsample_conv = nn.ConvTranspose2d(in_channels=gen_proj_dim, out_channels=gen_proj_dim, kernel_size=ds, stride=ds, padding=0, output_padding=0) # jia
        # self.cosmos_out_proj = nn.Conv2d(in_channels=gen_proj_dim, out_channels=vae_dim, kernel_size=1, stride=1, padding=0) # jia
        self.cosmos_out_proj = nn.Linear(gen_proj_dim, vae_dim)
        self.cosmos_out_layer_norm = nn.LayerNorm(gen_proj_dim) # jia

        self.action_in_proj = nn.Linear(config.max_action_dim, action_expert_config.hidden_size)
        self.action_out_proj = nn.Linear(action_expert_config.hidden_size, config.max_action_dim)

        self.state_proj = nn.Linear(config.max_state_dim, action_expert_config.hidden_size)
        self.action_time_mlp_in = nn.Linear(2 * action_expert_config.hidden_size, action_expert_config.hidden_size)
        self.action_time_mlp_out = nn.Linear(action_expert_config.hidden_size, action_expert_config.hidden_size)

        self._last_current_cosmos_features = None
        self._last_affordance_spatial_factor = None
        self._last_affordance_view_mask = None
        self._last_affordance_loss = torch.zeros(())
        self._last_affordance_target_valid = 0.0
        self._last_affordance_spatial_entropy = 0.0
        self._last_affordance_spatial_peak = 0.0
        self._last_affordance_action_gate = 0.0
        self._last_affordance_router_gate = 0.0
        self._last_affordance_v3_router_prior = None
        self._last_affordance_v3_raw_router_logits = None
        self._last_affordance_action_align_loss = torch.zeros(())
        if bool(getattr(config, "enable_affordance", False)):
            affordance_dim = int(config.affordance_dim)
            self.affordance_bridge = nn.Sequential(
                nn.LayerNorm(gen_proj_dim),
                nn.Linear(gen_proj_dim, affordance_dim),
                nn.SiLU(),
                nn.Linear(affordance_dim, affordance_dim),
            )
            if not bool(getattr(config, "enable_affordance_v3", False)):
                # V1/V2 compatibility paths. V3 never changes action tokens or
                # the state geometry seen by the original router.
                self.affordance_state_router = nn.Linear(affordance_dim, config.max_state_dim, bias=False)
                self.affordance_action_proj = nn.Linear(affordance_dim, action_expert_config.hidden_size)
            effect_channels = vae_dim * (2 if bool(getattr(config, "enable_affordance_v3", False)) else 1)
            self.affordance_delta_head = nn.Linear(affordance_dim, effect_channels)
            self.affordance_spatial_head = nn.Linear(gen_proj_dim, 1)

            # Preserve the original LoRA-MoE behavior at initialization. The
            # affordance branch only conditions existing router inputs.
            if not bool(getattr(config, "enable_affordance_v3", False)):
                nn.init.zeros_(self.affordance_state_router.weight)
                nn.init.zeros_(self.affordance_action_proj.weight)
                nn.init.zeros_(self.affordance_action_proj.bias)
            nn.init.zeros_(self.affordance_spatial_head.weight)
            nn.init.zeros_(self.affordance_spatial_head.bias)

            if bool(getattr(config, "enable_affordance_v3", False)):
                self.affordance_task_query = nn.Linear(vlm_config.hidden_size, affordance_dim, bias=False)
                self.affordance_spatial_key = nn.Linear(gen_proj_dim, affordance_dim, bias=False)
                self.affordance_router_prior = nn.Linear(
                    affordance_dim,
                    config.loramoe_num_experts,
                    bias=False,
                )
                nn.init.normal_(self.affordance_task_query.weight, std=0.02)
                nn.init.zeros_(self.affordance_spatial_key.weight)
                nn.init.zeros_(self.affordance_router_prior.weight)
            elif bool(getattr(config, "enable_affordance_v2", False)):
                # V2 context paths are zero-initialized where possible so the
                # new switch starts from the V1 affordance behavior.
                self.affordance_v2_motion_proj = nn.Linear(gen_proj_dim, affordance_dim, bias=False)
                self.affordance_v2_task_context_proj = nn.Linear(vlm_config.hidden_size, affordance_dim, bias=False)
                self.affordance_v2_task_query = nn.Linear(vlm_config.hidden_size, affordance_dim, bias=False)
                self.affordance_v2_spatial_key = nn.Linear(gen_proj_dim, affordance_dim, bias=False)
                v2_num_experts = config.loramoe_num_experts + int(
                    bool(getattr(config, "use_lora_moe_forced_last", False))
                )
                self.affordance_v2_router_logits = nn.Linear(
                    affordance_dim,
                    v2_num_experts,
                    bias=False,
                )
                nn.init.zeros_(self.affordance_v2_motion_proj.weight)
                nn.init.zeros_(self.affordance_v2_task_context_proj.weight)
                nn.init.normal_(self.affordance_v2_task_query.weight, std=0.02)
                nn.init.zeros_(self.affordance_v2_spatial_key.weight)
                self.affordance_v2_action_gate = nn.Linear(
                    affordance_dim + config.max_state_dim,
                    1,
                )
                self.affordance_v2_router_gate = nn.Linear(
                    affordance_dim + config.max_state_dim,
                    1,
                )
                nn.init.zeros_(self.affordance_v2_action_gate.weight)
                nn.init.zeros_(self.affordance_v2_router_gate.weight)
                gate_bias = math.log(
                    float(config.affordance_gate_init) / (1.0 - float(config.affordance_gate_init))
                )
                nn.init.constant_(self.affordance_v2_action_gate.bias, gate_bias)
                nn.init.constant_(self.affordance_v2_router_gate.bias, gate_bias)
                nn.init.zeros_(self.affordance_v2_router_logits.weight)

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        # Don't apply LoRA here - wait until after pretrained weights are loaded
        self.lora_model = None
        self._lora_applied = False
        self._ptq_applied = False
        self._ptq_summary = None
        self._use_forced_last_moe = False
        self._state_proj_force_last_expert = False
        self._state_proj_forced_last_expert_index = None
        self._state_proj_force_last_train_only = True

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            # Also compile the main forward pass used during training
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

        self.set_requires_grad()

    def apply_ptq(self):
        """Apply PTQ to native expert LLMs only."""
        if self._ptq_applied:
            logging.info("[PTQ] PTQ already applied, skipping.")
            return self._ptq_summary

        if self._is_any_lora_mode_enabled():
            raise ValueError("PTQ currently supports only native expert structure. Disable LoRA/LoRA-MoE first.")

        if self.config.ptq_mode not in ("weight_only_int8", "weight_only_int4"):
            raise ValueError(f"Unsupported PTQ mode: {self.config.ptq_mode}")

        num_bits = 8 if self.config.ptq_mode == "weight_only_int8" else 4

        summary = quantize_native_expert_llms(
            self.qwen3_vl_with_expert,
            expert_names=tuple(self.config.ptq_experts),
            target_module_names=tuple(self.config.ptq_target_modules),
            num_bits=num_bits,
            backend=self.config.ptq_backend,
        )
        self._ptq_applied = True
        self._ptq_summary = summary
        return summary

    def _is_forced_last_mode_enabled(self) -> bool:
        return bool(getattr(self.config, 'use_lora_moe_forced_last', False))

    def _is_any_lora_mode_enabled(self) -> bool:
        return bool(self.config.use_lora or self.config.use_lora_moe or self._is_forced_last_mode_enabled())

    @staticmethod
    def _select_topk_with_optional_forced_last(
        probs: torch.Tensor,
        top_k: int,
        force_last: bool,
        forced_last_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_experts = probs.shape[-1]
        top_k = min(top_k, total_experts)
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        if not force_last:
            top_k_probs, top_k_indices = torch.topk(probs, top_k, dim=-1)
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            return top_k_probs, top_k_indices

        forced_last_index = max(0, min(forced_last_index, total_experts - 1))
        forced_probs = probs[:, forced_last_index:forced_last_index + 1]
        forced_indices = torch.full(
            (probs.shape[0], 1),
            forced_last_index,
            device=probs.device,
            dtype=torch.long,
        )

        remaining_top_k = min(top_k - 1, total_experts - 1)
        if remaining_top_k > 0:
            remaining_probs = torch.cat(
                [probs[:, :forced_last_index], probs[:, forced_last_index + 1:]],
                dim=-1,
            )
            remaining_top_k_probs, remaining_top_k_indices = torch.topk(remaining_probs, remaining_top_k, dim=-1)
            remaining_top_k_indices = torch.where(
                remaining_top_k_indices >= forced_last_index,
                remaining_top_k_indices + 1,
                remaining_top_k_indices,
            )
            top_k_probs = torch.cat([forced_probs, remaining_top_k_probs], dim=-1)
            top_k_indices = torch.cat([forced_indices, remaining_top_k_indices], dim=-1)
        else:
            top_k_probs = forced_probs
            top_k_indices = forced_indices

        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        return top_k_probs, top_k_indices

    def _apply_lora(self):
        """Apply LoRA to all experts except the action head."""
        config = self.config

        # Create LoRA configuration
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            inference_mode=False,
        )

        logging.info("=" * 60)
        logging.info("Applying LoRA to model experts:")
        logging.info(f"  - LoRA rank: {config.lora_rank}")
        logging.info(f"  - LoRA alpha: {config.lora_alpha}")
        logging.info(f"  - LoRA dropout: {config.lora_dropout}")
        logging.info(f"  - Target modules: {config.lora_target_modules}")
        logging.info("-" * 60)

        # Apply LoRA to the understanding expert's language model
        logging.info("  [1/3] Applying LoRA to und_expert (Qwen3-VL)...")
        self.qwen3_vl_with_expert.und_expert = get_peft_model(
            self.qwen3_vl_with_expert.und_expert,
            lora_config
        )

        # Apply LoRA to generation expert
        logging.info("  [2/3] Applying LoRA to gen_expert (Generation)...")
        self.qwen3_vl_with_expert.gen_expert = get_peft_model(
            self.qwen3_vl_with_expert.gen_expert,
            lora_config
        )

        # Apply LoRA to action expert (but not action head - action_in_proj, action_out_proj)
        logging.info("  [3/3] Applying LoRA to act_expert (Action)...")
        self.qwen3_vl_with_expert.act_expert = get_peft_model(
            self.qwen3_vl_with_expert.act_expert,
            lora_config
        )

        # ============================================================
        # Apply LoRA to state_proj using the same config
        # ============================================================
        logging.info("  [4/4] Applying LoRA to state_proj...")
        self.state_proj_lora_A = nn.Linear(config.max_state_dim, config.lora_rank, bias=False)
        self.state_proj_lora_B = nn.Linear(config.lora_rank, self.state_proj.out_features, bias=False)
        # Store scaling factor for later use
        self.state_proj_lora_scaling = config.lora_alpha / config.lora_rank
        logging.info(f"    - Added LoRA to state_proj: rank={config.lora_rank}, alpha={config.lora_alpha}")

        # Mark as LoRA model for reference
        self.lora_model = self.qwen3_vl_with_expert.und_expert

        # ============================================================
        # Print detailed LoRA layer information
        # ============================================================
        logging.info("=" * 60)
        logging.info("LoRA Layers Details:")
        logging.info("=" * 60)

        # Collect all LoRA layer names
        lora_layers = {}
        for name, param in self.named_parameters():
            if "lora_" in name or ".lora_" in name:
                # Extract layer name, e.g., "model.layers.0.self_attn.q_proj.lora_A"
                parts = name.split(".")
                # Find the layer identifier
                layer_idx = None
                for i, part in enumerate(parts):
                    if part.isdigit():
                        layer_idx = part
                        break

                # Get the module type (q_proj, k_proj, v_proj, o_proj, etc.)
                module_type = None
                for part in parts:
                    if "proj" in part or "gate" in part or "up" in part or "down" in part:
                        module_type = part
                        break

                if layer_idx and module_type:
                    key = f"layer_{layer_idx}.{module_type}"
                    if key not in lora_layers:
                        lora_layers[key] = 0
                    lora_layers[key] += param.numel()

        # Print grouped by expert
        def print_lora_layers_for_expert(expert_name, expert_model):
            expert_layers = {}
            for name, param in expert_model.named_parameters():
                if "lora_" in name or ".lora_" in name:
                    parts = name.split(".")
                    layer_idx = None
                    for part in parts:
                        if part.isdigit():
                            layer_idx = part
                            break
                    module_type = None
                    for part in parts:
                        if "proj" in part or "gate" in part or "up" in part or "down" in part:
                            module_type = part
                            break
                    if layer_idx and module_type:
                        key = f"layer_{layer_idx}.{module_type}"
                        if key not in expert_layers:
                            expert_layers[key] = 0
                        expert_layers[key] += param.numel()

            if expert_layers:
                logging.info(f"  [{expert_name}]")
                sorted_keys = sorted(expert_layers.keys(), key=lambda x: int(x.split("_")[1]) if x.split("_")[1].isdigit() else 0)
                for key in sorted_keys:
                    params = expert_layers[key]
                    logging.info(f"    - {key}: {params:,} params")

        print_lora_layers_for_expert("und_expert", self.qwen3_vl_with_expert.und_expert)
        print_lora_layers_for_expert("gen_expert", self.qwen3_vl_with_expert.gen_expert)
        print_lora_layers_for_expert("act_expert", self.qwen3_vl_with_expert.act_expert)

        # Count total LoRA layers by type
        lora_type_counts = {}
        for name, param in self.named_parameters():
            if "lora_A" in name or ".lora_A" in name:
                # Get the module type
                for part in name.split("."):
                    if "proj" in part or "gate" in part or "up" in part or "down" in part:
                        lora_type_counts[part] = lora_type_counts.get(part, 0) + 1
                        break

        logging.info("-" * 60)
        logging.info("  LoRA layer count by type:")
        for module_type, count in sorted(lora_type_counts.items()):
            logging.info(f"    - {module_type}: {count} layers")
        logging.info(f"  Total LoRA layers: {sum(lora_type_counts.values())}")
        logging.info("=" * 60)

        logging.info("-" * 60)
        logging.info("  [SKIP] action_in_proj (Action Head - FULL TRAINABLE)")
        logging.info("  [SKIP] action_out_proj (Action Head - FULL TRAINABLE)")
        logging.info("  [SKIP] cosmos (Video Tokenizer - FROZEN)")
        logging.info("=" * 60)

        logging.info(f"Applied LoRA with rank={config.lora_rank}, alpha={config.lora_alpha}, "
                    f"dropout={config.lora_dropout}, target_modules={config.lora_target_modules}")

    def _apply_lora_moe(self):
        """Apply LoRA-MoE to all experts except the action head.

        This method creates multiple LoRA experts per layer with a router to dynamically
        select experts for each input token.

        Supports two routing modes:
        - Standard (default): Single router for both lora_A and lora_B
        - AB routing: Separate routers for lora_A and lora_B (set via config.loramoe_ab_routing)
        """
        config = self.config

        # Check if using AB routing (separate routers for A and B)
        use_ab_routing = getattr(config, 'loramoe_ab_routing', False)

        # Create LoRA-MoE configuration
        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            inference_mode=False,
            enable_loramoe=True,
            num_experts=config.loramoe_num_experts,
            router_top_k=config.loramoe_router_top_k,
        )

        num_experts = config.loramoe_num_experts
        top_k_a = getattr(config, 'loramoe_router_top_k_a', config.loramoe_router_top_k)
        top_k_b = getattr(config, 'loramoe_router_top_k_b', config.loramoe_router_top_k)

        logging.info("=" * 60)
        logging.info("Applying LoRA-MoE to model experts:")
        logging.info(f"  - LoRA rank: {config.lora_rank}")
        logging.info(f"  - LoRA alpha: {config.lora_alpha}")
        logging.info(f"  - Number of experts: {num_experts}")
        logging.info(f"  - Router top-k: {config.loramoe_router_top_k}")
        logging.info(f"  - AB routing: {use_ab_routing}")
        if use_ab_routing:
            logging.info(f"  - Router A top-k: {top_k_a}")
            logging.info(f"  - Router B top-k: {top_k_b}")
        logging.info(f"  - Target modules: {config.lora_target_modules}")
        logging.info("-" * 60)

        # Apply LoRA-MoE to each expert with num_experts adapters
        from peft import inject_adapter_in_model

        for expert_name, expert in [
            ("und_expert", self.qwen3_vl_with_expert.und_expert),
            ("gen_expert", self.qwen3_vl_with_expert.gen_expert),
            ("act_expert", self.qwen3_vl_with_expert.act_expert),
        ]:
            logging.info(f"  Applying LoRA-MoE to {expert_name}...")

            # First adapter named "0", injected via get_peft_model
            expert = get_peft_model(expert, lora_config, adapter_name="0")

            # Debug: check adapters after get_peft_model
            adapter_names = list(expert.peft_config.keys())
            logging.info(f"    [DEBUG] Adapters after get_peft_model: {adapter_names}")

            # Add remaining adapters: "1", "2", ..., "E-1"
            for exp_id in range(1, num_experts):
                inject_adapter_in_model(lora_config, expert, str(exp_id))

            # Debug: check adapters after inject
            adapter_names_after = list(expert.peft_config.keys())
            logging.info(f"    [DEBUG] Adapters after inject: {adapter_names_after}")

            # Initialize router for all LoRA layers
            for name, module in expert.named_modules():
                if hasattr(module, 'init_lora_router'):
                    if use_ab_routing and hasattr(module, 'init_lora_router_ab'):
                        # Use separate A/B routers
                        module.init_lora_router_ab(
                            num_experts=num_experts,
                            top_k_a=top_k_a,
                            top_k_b=top_k_b
                        )
                    else:
                        # Use standard single router
                        module.init_lora_router(
                            num_experts=num_experts,
                            top_k=config.loramoe_router_top_k
                        )
                    module.set_lora_moe_mode(True)

            # Router inputs are evaluated in float32, so keep router weights aligned.
            lora_param_count = 0
            for name, param in expert.named_parameters():
                if "lora_A" in name or "lora_B" in name or "lora_router" in name:
                    param.data = param.data.to(torch.float32)
                    lora_param_count += 1

            logging.info(f"    - Converted {lora_param_count} LoRA params to float32")

            # Store back to the expert
            if expert_name == "und_expert":
                self.qwen3_vl_with_expert.und_expert = expert
            elif expert_name == "gen_expert":
                self.qwen3_vl_with_expert.gen_expert = expert
            else:
                self.qwen3_vl_with_expert.act_expert = expert

        # ============================================================
        # Apply LoRA-MoE to state_proj (manual implementation)
        # ============================================================
        logging.info("  Applying LoRA-MoE to state_proj...")
        if use_ab_routing:
            self._apply_state_proj_lora_moe_ab(config)
        else:
            self._apply_state_proj_lora_moe(config)

        # Mark as LoRA model for reference
        self.lora_model = self.qwen3_vl_with_expert.und_expert

        # Set the mode flag
        self._moe_mode = True
        self._use_ab_routing = use_ab_routing
        self._install_affordance_v3_action_router_hooks()

        logging.info("=" * 60)
        logging.info(f"Applied LoRA-MoE with rank={config.lora_rank}, num_experts={num_experts}, "
                    f"router_top_k={config.loramoe_router_top_k}, ab_routing={use_ab_routing}")

    def _apply_lora_moe_forced_last(self):
        """Apply server-only LoRA-MoE with one extra expert forced during training."""
        config = self.config
        if getattr(config, 'loramoe_ab_routing', False):
            raise ValueError("use_lora_moe_forced_last does not support loramoe_ab_routing")

        base_num_experts = int(config.loramoe_num_experts)
        server_num_experts = base_num_experts + 1
        forced_last_index = server_num_experts - 1

        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            inference_mode=False,
            enable_loramoe=True,
            num_experts=server_num_experts,
            router_top_k=config.loramoe_router_top_k,
        )

        logging.info("=" * 60)
        logging.info("Applying LoRA-MoE Forced-Last to model experts:")
        logging.info(f"  - LoRA rank: {config.lora_rank}")
        logging.info(f"  - LoRA alpha: {config.lora_alpha}")
        logging.info(f"  - Base experts (client): {base_num_experts}")
        logging.info(f"  - Server experts: {server_num_experts}")
        logging.info(f"  - Forced training expert index: {forced_last_index}")
        logging.info(f"  - Router top-k: {config.loramoe_router_top_k}")
        logging.info(f"  - Target modules: {config.lora_target_modules}")
        logging.info("-" * 60)

        from peft import inject_adapter_in_model

        for expert_name, expert in [
            ("und_expert", self.qwen3_vl_with_expert.und_expert),
            ("gen_expert", self.qwen3_vl_with_expert.gen_expert),
            ("act_expert", self.qwen3_vl_with_expert.act_expert),
        ]:
            logging.info(f"  Applying LoRA-MoE Forced-Last to {expert_name}...")
            expert = get_peft_model(expert, lora_config, adapter_name="0")
            for exp_id in range(1, server_num_experts):
                inject_adapter_in_model(lora_config, expert, str(exp_id))

            for _, module in expert.named_modules():
                if hasattr(module, 'init_lora_router'):
                    module.init_lora_router(
                        num_experts=server_num_experts,
                        top_k=config.loramoe_router_top_k,
                    )
                    if hasattr(module, 'configure_forced_last_expert_routing'):
                        module.configure_forced_last_expert_routing(
                            enabled=True,
                            expert_index=forced_last_index,
                            train_only=True,
                        )
                    module.set_lora_moe_mode(True)

            lora_param_count = 0
            for name, param in expert.named_parameters():
                if "lora_A" in name or "lora_B" in name or "lora_router" in name:
                    param.data = param.data.to(torch.float32)
                    lora_param_count += 1
            logging.info(f"    - Converted {lora_param_count} LoRA params to float32")

            if expert_name == "und_expert":
                self.qwen3_vl_with_expert.und_expert = expert
            elif expert_name == "gen_expert":
                self.qwen3_vl_with_expert.gen_expert = expert
            else:
                self.qwen3_vl_with_expert.act_expert = expert

        logging.info("  Applying LoRA-MoE Forced-Last to state_proj...")
        self._apply_state_proj_lora_moe(config, num_experts_override=server_num_experts)
        self._state_proj_force_last_expert = True
        self._state_proj_forced_last_expert_index = forced_last_index
        self._state_proj_force_last_train_only = True

        self.lora_model = self.qwen3_vl_with_expert.und_expert
        self._moe_mode = True
        self._use_ab_routing = False
        self._use_forced_last_moe = True

        logging.info("=" * 60)
        logging.info(
            f"Applied LoRA-MoE Forced-Last with rank={config.lora_rank}, "
            f"base_num_experts={base_num_experts}, server_num_experts={server_num_experts}, "
            f"router_top_k={config.loramoe_router_top_k}"
        )

    def _apply_state_proj_lora_moe(self, config, num_experts_override: int | None = None):
        """Manually implement LoRA-MoE for state_proj layer.

        This is needed because state_proj is not part of the PEFT-wrapped model.
        """
        num_experts = config.loramoe_num_experts if num_experts_override is None else num_experts_override

        # Initialize multiple LoRA expert pairs
        self.state_proj_lora_A_moe = nn.ModuleList([
            nn.Linear(config.max_state_dim, config.lora_rank, bias=False)
            for _ in range(num_experts)
        ])
        self.state_proj_lora_B_moe = nn.ModuleList([
            nn.Linear(config.lora_rank, self.state_proj.out_features, bias=False)
            for _ in range(num_experts)
        ])

        # Initialize router
        self.state_proj_router = nn.Parameter(
            torch.zeros(num_experts, config.max_state_dim, dtype=torch.float32)
        )
        nn.init.normal_(self.state_proj_router, std=0.02)

        # Store scaling factor
        self.state_proj_lora_scaling = config.lora_alpha / config.lora_rank
        self.state_proj_top_k = config.loramoe_router_top_k
        self._state_proj_force_last_expert = False
        self._state_proj_forced_last_expert_index = None
        self._state_proj_force_last_train_only = True

        # Pre-stack LoRA weights for vectorized computation: [E, In, R] and [E, R, Out]
        self._stack_lora_weights_state_proj()

        logging.info(f"    - Added LoRA-MoE to state_proj: rank={config.lora_rank}, "
                    f"num_experts={num_experts}, top_k={config.loramoe_router_top_k}")

    def _stack_lora_weights_state_proj(self):
        """Stack all LoRA expert weights into a single tensor for vectorized computation.

        Registers stacked weights as buffers so they automatically move with .to(device).
        Creates:
            _state_proj_lora_A_stack: [E, Rank, In]   <- Linear(In, Rank).weight = [Rank, In]
            _state_proj_lora_B_stack: [E, Out, Rank]  <- Linear(Rank, Out).weight = [Out, Rank]
        """
        lora_A_list = [m.weight.data for m in self.state_proj_lora_A_moe]
        lora_B_list = [m.weight.data for m in self.state_proj_lora_B_moe]
        self.register_buffer("_state_proj_lora_A_stack", torch.stack(lora_A_list, dim=0), persistent=False)
        self.register_buffer("_state_proj_lora_B_stack", torch.stack(lora_B_list, dim=0), persistent=False)

    def _install_affordance_v3_action_router_hooks(self):
        if not bool(getattr(self.config, "enable_affordance_v3", False)):
            return
        if bool(getattr(self.config, "loramoe_ab_routing", False)):
            raise ValueError("Affordance V3 requires standard single-router LoRA-MoE")

        owner_ref = weakref.ref(self)
        hooked = 0
        for module in self.qwen3_vl_with_expert.act_expert.modules():
            if not hasattr(module, "lora_router_forward") or not hasattr(module, "_cache_router_observables"):
                continue
            if getattr(module, "_affordance_v3_router_hooked", False):
                continue

            def lora_router_forward_with_affordance(layer, x, _owner_ref=owner_ref):
                original_shape = x.shape
                x_flat = x.reshape(-1, x.shape[-1]) if len(original_shape) == 3 else x
                sample_idx = layer._build_sample_idx(x, x_flat)
                router_input = x_flat.float()
                router_feature = router_input
                if isinstance(layer.lora_router, nn.Module):
                    if isinstance(layer.lora_router, nn.Sequential) and len(layer.lora_router) >= 3:
                        hidden = layer.lora_router[1](layer.lora_router[0](router_input))
                        logits = layer.lora_router[2](hidden)
                        router_feature = hidden
                    else:
                        logits = layer.lora_router(router_input)
                else:
                    logits = torch.matmul(router_input, layer.lora_router.t())

                owner = _owner_ref()
                prior = None if owner is None else owner._last_affordance_v3_router_prior
                if prior is not None:
                    if prior.shape[-1] != logits.shape[-1]:
                        raise RuntimeError(
                            "Affordance V3 prior and Action Router disagree on expert count: "
                            f"{prior.shape[-1]} != {logits.shape[-1]}"
                        )
                    logits = logits + prior.to(device=logits.device, dtype=logits.dtype)[sample_idx]

                logits = torch.clamp(logits, min=-50, max=50)
                probs = torch.softmax(logits, dim=-1)
                top_k = min(layer.lora_router_top_k, layer.num_experts)
                top_k_probs, top_k_indices = layer._select_topk_with_optional_forced_last(probs, top_k)

                if layer.training:
                    token_count, num_experts = probs.shape
                    expert_mask = torch.zeros(
                        token_count,
                        num_experts,
                        device=probs.device,
                        dtype=probs.dtype,
                    )
                    expert_mask.scatter_(1, top_k_indices, 1.0)
                    layer.aux_loss = num_experts * torch.sum(expert_mask.mean(dim=0) * probs.mean(dim=0))
                else:
                    layer.aux_loss = None

                layer._cache_router_observables("router", router_feature, probs, sample_idx)
                return top_k_probs, top_k_indices, x_flat

            module.lora_router_forward = MethodType(lora_router_forward_with_affordance, module)
            module._affordance_v3_router_hooked = True
            hooked += 1

        if hooked == 0:
            raise RuntimeError(
                "Affordance V3 found no compatible Action LoRA-MoE routers. "
                "Verify that the patched PEFT LoRA-MoE implementation is installed."
            )
        logging.info("[AffordanceV3] Installed router-only prior on %d Action LoRA-MoE modules", hooked)

    def _set_affordance_v3_router_prior(self, affordance):
        if not bool(getattr(self.config, "enable_affordance_v3", False)) or affordance is None:
            self._last_affordance_v3_router_prior = None
            self._last_affordance_v3_raw_router_logits = None
            return

        raw_prior = self.affordance_router_prior(
            affordance.to(dtype=self.affordance_router_prior.weight.dtype)
        ).float()
        self._last_affordance_v3_raw_router_logits = raw_prior
        centered_prior = raw_prior - raw_prior.mean(dim=-1, keepdim=True)
        self._last_affordance_v3_router_prior = torch.tanh(centered_prior)
        self._last_affordance_router_gate = self._last_affordance_v3_router_prior.detach().abs().mean()

    def _compute_affordance_v3_action_align_loss(self, loss_action):
        prior = self._last_affordance_v3_router_prior
        raw_logits = self._last_affordance_v3_raw_router_logits
        if (
            not self.training
            or not bool(getattr(self.config, "enable_affordance_v3", False))
            or prior is None
            or raw_logits is None
            or not prior.requires_grad
        ):
            return loss_action.new_zeros(())

        action_objective = loss_action.reshape(loss_action.shape[0], -1).mean(dim=1).sum()
        prior_gradient = torch.autograd.grad(
            action_objective,
            prior,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]
        if prior_gradient is None:
            return loss_action.new_zeros(())

        usefulness = -prior_gradient.detach()
        usefulness = usefulness - usefulness.mean(dim=-1, keepdim=True)
        usefulness_scale = usefulness.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
        target = torch.softmax(usefulness / usefulness_scale, dim=-1)
        return F.kl_div(
            F.log_softmax(raw_logits, dim=-1),
            target,
            reduction="batchmean",
        )

    def _affordance_route_input(self, x, affordance):
        if not bool(getattr(self.config, "enable_affordance", False)) or affordance is None:
            return x, x
        if bool(getattr(self.config, "enable_affordance_v2", False)) or bool(
            getattr(self.config, "enable_affordance_v3", False)
        ):
            # V2 adds a bounded residual directly to router logits instead of
            # changing the state geometry seen by the original router.
            return x, x
        if not bool(getattr(self.config, "affordance_state_router_input", True)):
            return x, x
        route_delta = self.affordance_state_router(affordance.to(self.affordance_state_router.weight.dtype))
        return x, (x + route_delta.to(dtype=x.dtype))

    def _affordance_v2_router_logit_delta(self, x, affordance):
        if affordance is None:
            return None

        batch_size = x.shape[0]
        if bool(getattr(self.config, "enable_affordance_v3", False)):
            if not bool(getattr(self.config, "affordance_v3_state_router_prior", True)):
                return None
            prior = self._last_affordance_v3_router_prior
            if prior is None:
                return None
            token_count = x.reshape(batch_size, -1, x.shape[-1]).shape[1]
            return prior[:, None, :].expand(batch_size, token_count, -1).reshape(-1, prior.shape[-1])
        if not bool(getattr(self.config, "enable_affordance_v2", False)) or not hasattr(
            self,
            "affordance_v2_router_logits",
        ):
            return None
        state_summary = x.reshape(batch_size, -1, x.shape[-1]).float().mean(dim=1)
        gate_dtype = self.affordance_v2_router_gate.weight.dtype
        gate_input = torch.cat([affordance.float(), state_summary], dim=-1).to(dtype=gate_dtype)
        gate = torch.sigmoid(self.affordance_v2_router_gate(gate_input))
        delta_dtype = self.affordance_v2_router_logits.weight.dtype
        delta = self.affordance_v2_router_logits(affordance.to(dtype=delta_dtype))
        router_reference = getattr(self, "state_proj_router", None)
        if router_reference is None:
            router_reference = self.state_proj_router_a
        expected_experts = int(router_reference.shape[0])
        if delta.shape[-1] != expected_experts:
            raise RuntimeError(
                "Affordance V2 router head and state_proj router disagree on expert count: "
                f"{delta.shape[-1]} != {expected_experts}"
            )
        token_count = x.reshape(batch_size, -1, x.shape[-1]).shape[1]
        delta = delta[:, None, :].expand(batch_size, token_count, -1).reshape(-1, delta.shape[-1])
        gate = gate[:, None, :].expand(batch_size, token_count, -1).reshape(-1, 1)
        self._last_affordance_router_gate = gate.detach().mean()
        return gate.to(dtype=delta.dtype) * delta

    def state_proj_lora_moe_forward(self, x, affordance=None):
        """Vectorized forward pass for state_proj LoRA-MoE.

        Computes all E expert outputs in a single batched operation using torch.einsum,
        then applies top-k routing probability weighting and sums.
        Eliminates the Python loop over experts.

        Args:
            x: Input tensor of shape (batch, max_state_dim)

        Returns:
            Output tensor with LoRA-MoE applied.
        """
        # Keep the base projection and LoRA expert inputs unchanged. Only the
        # state Router sees the affordance-conditioned route input.
        base_out = self.state_proj(x)
        base_flat = base_out.reshape(-1, base_out.shape[-1])  # (T, action_hidden)

        # Router computation (keep router in float32 for stability)
        x_flat = x.reshape(-1, x.shape[-1])  # (T, max_state_dim)
        _, route_x = self._affordance_route_input(x, affordance)
        route_x_flat = route_x.reshape(-1, route_x.shape[-1])
        logits = torch.matmul(route_x_flat.float(), self.state_proj_router.t())
        v2_router_delta = self._affordance_v2_router_logit_delta(x, affordance)
        if v2_router_delta is not None:
            logits = logits + v2_router_delta.float()
        logits = torch.clamp(logits, min=-50, max=50)
        probs = torch.softmax(logits, dim=-1)
        self._last_state_proj_router_probs = probs.detach()

        # Select top-k experts
        top_k = min(self.state_proj_top_k, logits.shape[-1])
        force_last = bool(getattr(self, '_state_proj_force_last_expert', False))
        if force_last and getattr(self, '_state_proj_force_last_train_only', True):
            force_last = self.training
        top_k_probs, top_k_indices = self._select_topk_with_optional_forced_last(
            probs,
            top_k,
            force_last,
            int(getattr(self, '_state_proj_forced_last_expert_index', logits.shape[-1] - 1) or (logits.shape[-1] - 1)),
        )

        # ---- Auxiliary Loss: expert load-balancing (only during training) ----
        if self.training:
            T, E = probs.shape
            # f_i: fraction of tokens per expert (via top_k_indices count)
            expert_mask = torch.zeros(T, E, device=probs.device, dtype=probs.dtype)
            expert_mask.scatter_(1, top_k_indices, 1.0)
            f_i = expert_mask.mean(dim=0)  # (E,)
            # P_i: mean routing probability per expert
            P_i = probs.mean(dim=0)  # (E,)
            # aux_loss = E * sum(f_i * P_i)
            self.state_proj_aux_loss = E * torch.sum(f_i * P_i)
        else:
            self.state_proj_aux_loss = None

        # ---- Dynamic stacking: get current weights from nn.ModuleList (not Buffer!) ----
        # This ensures gradients flow back to the actual LoRA parameters during training
        # Optimization: collect weights first, then stack in single GPU operation
        lora_A_weights = [m.weight for m in self.state_proj_lora_A_moe]  # Use Parameter
        lora_B_weights = [m.weight for m in self.state_proj_lora_B_moe]

        # Get target device and dtype from input
        target_device = x_flat.device
        target_dtype = x_flat.dtype

        # Optimized: stack first, then convert dtype once (avoids per-element .to() calls)
        # Assumes all weights are on the same device (standard for GPU training)
        lora_A_stack = torch.stack(lora_A_weights, dim=0).to(target_dtype)
        lora_B_stack = torch.stack(lora_B_weights, dim=0).to(target_dtype)

        x_f32 = x_flat.to(lora_A_stack.dtype)

        # 1. LoRA_A: (T, In) * (E, Rank, In) -> (T, E, Rank)
        lora_A_out = torch.einsum('ti, eri -> ter', x_f32, lora_A_stack)

        # 2. LoRA_B: (T, E, Rank) * (E, Out, Rank) -> (T, E, Out)
        lora_B_out = torch.einsum('ter, eor -> teo', lora_A_out, lora_B_stack)

        # 3. Apply scaling
        lora_B_out = lora_B_out * self.state_proj_lora_scaling

        # 4. Apply routing probability weighting
        T, E, _ = lora_B_out.shape
        gate_weights = torch.zeros(T, E, device=lora_B_out.device, dtype=lora_B_out.dtype)
        gate_weights.scatter_(1, top_k_indices, top_k_probs.to(lora_B_out.dtype))

        # 5. Weighted sum over expert dimension: (T, E, Out) * (T, E, 1) -> sum -> (T, Out)
        moe_diff = (lora_B_out * gate_weights.unsqueeze(-1)).sum(dim=1)

        # Add base output and MoE contribution
        result_flat = base_flat + moe_diff
        return result_flat.reshape(*base_out.shape)

    def _apply_state_proj_lora_moe_ab(self, config):
        """Manually implement LoRA-MoE with separate A/B routing for state_proj layer.

        This enables independent expert selection for lora_A and lora_B matrices,
        allowing different experts to handle input vs output transformations.
        """
        num_experts = config.loramoe_num_experts
        top_k_a = getattr(config, 'loramoe_router_top_k_a', config.loramoe_router_top_k)
        top_k_b = getattr(config, 'loramoe_router_top_k_b', config.loramoe_router_top_k)

        # Initialize multiple LoRA expert pairs
        self.state_proj_lora_A_moe = nn.ModuleList([
            nn.Linear(config.max_state_dim, config.lora_rank, bias=False)
            for _ in range(num_experts)
        ])
        self.state_proj_lora_B_moe = nn.ModuleList([
            nn.Linear(config.lora_rank, self.state_proj.out_features, bias=False)
            for _ in range(num_experts)
        ])

        # Initialize separate routers for A and B
        # Router A: selects experts for lora_A based on input
        self.state_proj_router_a = nn.Parameter(
            torch.zeros(num_experts, config.max_state_dim, dtype=torch.float32)
        )
        nn.init.normal_(self.state_proj_router_a, std=0.02)

        # Router B: selects experts for lora_B based on input
        # Can use same input dimension for consistency
        self.state_proj_router_b = nn.Parameter(
            torch.zeros(num_experts, config.max_state_dim, dtype=torch.float32)
        )
        nn.init.normal_(self.state_proj_router_b, std=0.02)

        # Store routing parameters
        self.state_proj_lora_scaling = config.lora_alpha / config.lora_rank
        self.state_proj_top_k_a = top_k_a
        self.state_proj_top_k_b = top_k_b

        # Mark that we're using separate AB routing
        self._use_state_proj_router_ab = True

        # Pre-stack LoRA weights for vectorized computation
        self._stack_lora_weights_state_proj()

        logging.info(f"    - Added LoRA-MoE (AB routing) to state_proj: rank={config.lora_rank}, "
                    f"num_experts={num_experts}, top_k_a={top_k_a}, top_k_b={top_k_b}")

    def state_proj_lora_moe_forward_ab(self, x, affordance=None):
        """Vectorized forward pass for state_proj LoRA-MoE with separate A/B routing (PARALLEL).

        This method uses independent top-k selection for lora_A and lora_B matrices.
        BOTH routers take the ORIGINAL INPUT x (parallel routing).

        Forward pass:
        1. x -> Router_A -> select top_k_a experts for lora_A
        2. x -> Router_B -> select top_k_b experts for lora_B (parallel, independent)
        3. lora_A(x) with Router_A weights
        4. lora_B(lora_A(x)) with Router_B weights
        5. Final = base + Router_B_weights * lora_B(lora_A(x))

        Args:
            x: Input tensor of shape (batch, max_state_dim)

        Returns:
            Output tensor with LoRA-MoE applied.
        """
        # Base output
        base_out = self.state_proj(x)
        base_flat = base_out.reshape(-1, base_out.shape[-1])  # (T, action_hidden)

        # Flatten input. Affordance changes only Router inputs, not the
        # pretrained state projection or the expert LoRA input.
        x_flat = x.reshape(-1, x.shape[-1])  # (T, max_state_dim)
        _, route_x = self._affordance_route_input(x, affordance)
        route_x_flat = route_x.reshape(-1, route_x.shape[-1])

        # ---- Router A: select experts for lora_A ----
        logits_a = torch.matmul(route_x_flat.float(), self.state_proj_router_a.t())
        v2_router_delta = self._affordance_v2_router_logit_delta(x, affordance)
        if v2_router_delta is not None:
            logits_a = logits_a + v2_router_delta.float()
        logits_a = torch.clamp(logits_a, min=-50, max=50)
        probs_a = torch.softmax(logits_a, dim=-1)
        self._last_state_proj_router_a_probs = probs_a.detach()

        top_k_a = min(self.state_proj_top_k_a, logits_a.shape[-1])
        top_k_probs_a, top_k_indices_a = torch.topk(probs_a, top_k_a, dim=-1)
        top_k_probs_a = top_k_probs_a / top_k_probs_a.sum(dim=-1, keepdim=True)

        # ---- Router B: select experts for lora_B ----
        logits_b = torch.matmul(route_x_flat.float(), self.state_proj_router_b.t())
        if v2_router_delta is not None:
            logits_b = logits_b + v2_router_delta.float()
        logits_b = torch.clamp(logits_b, min=-50, max=50)
        probs_b = torch.softmax(logits_b, dim=-1)
        self._last_state_proj_router_b_probs = probs_b.detach()

        top_k_b = min(self.state_proj_top_k_b, logits_b.shape[-1])
        top_k_probs_b, top_k_indices_b = torch.topk(probs_b, top_k_b, dim=-1)
        top_k_probs_b = top_k_probs_b / top_k_probs_b.sum(dim=-1, keepdim=True)

        # ---- Auxiliary Loss: expert load-balancing for both A and B ----
        if self.training:
            T, E = probs_a.shape

            # Aux loss for Router A
            expert_mask_a = torch.zeros(T, E, device=probs_a.device, dtype=probs_a.dtype)
            expert_mask_a.scatter_(1, top_k_indices_a, 1.0)
            f_i_a = expert_mask_a.mean(dim=0)
            P_i_a = probs_a.mean(dim=0)
            aux_loss_a = E * torch.sum(f_i_a * P_i_a)

            # Aux loss for Router B
            expert_mask_b = torch.zeros(T, E, device=probs_b.device, dtype=probs_b.dtype)
            expert_mask_b.scatter_(1, top_k_indices_b, 1.0)
            f_i_b = expert_mask_b.mean(dim=0)
            P_i_b = probs_b.mean(dim=0)
            aux_loss_b = E * torch.sum(f_i_b * P_i_b)

            # Combined aux loss
            self.state_proj_aux_loss = aux_loss_a + aux_loss_b
        else:
            self.state_proj_aux_loss = None

        # ---- Compute LoRA outputs ----
        lora_A_weights = [m.weight for m in self.state_proj_lora_A_moe]
        lora_B_weights = [m.weight for m in self.state_proj_lora_B_moe]

        target_device = x_flat.device
        target_dtype = x_flat.dtype

        lora_A_stack = torch.stack(lora_A_weights, dim=0).to(target_dtype)
        lora_B_stack = torch.stack(lora_B_weights, dim=0).to(target_dtype)

        x_f32 = x_flat.to(lora_A_stack.dtype)

        # 1. LoRA_A: (T, In) * (E, Rank, In) -> (T, E, Rank)
        lora_A_out = torch.einsum('ti, eri -> ter', x_f32, lora_A_stack)

        # 2. Apply Router A probability weighting to lora_A outputs
        T, E, Rank = lora_A_out.shape
        combined_weights_a = torch.zeros(T, E, device=lora_A_out.device, dtype=lora_A_out.dtype)
        combined_weights_a.scatter_(1, top_k_indices_a, top_k_probs_a.to(lora_A_out.dtype))
        # Weighted lora_A output: (T, E, Rank) * (T, E, 1) -> sum -> (T, Rank)
        lora_A_weighted = (lora_A_out * combined_weights_a.unsqueeze(-1)).sum(dim=1)

        # 3. Expand lora_A_weighted for lora_B computation: (T, Rank) -> (T, E, Rank)
        lora_A_weighted_expanded = lora_A_weighted.unsqueeze(1).expand(-1, E, -1)

        # 4. LoRA_B: (T, E, Rank) * (E, Out, Rank) -> (T, E, Out)
        lora_B_out = torch.einsum('ter, eor -> teo', lora_A_weighted_expanded, lora_B_stack)

        # 5. Apply scaling
        lora_B_out = lora_B_out * self.state_proj_lora_scaling

        # 6. Apply Router B probability weighting
        Out = base_flat.shape[-1]
        gate_weights = torch.zeros(T, E, device=lora_B_out.device, dtype=lora_B_out.dtype)
        gate_weights.scatter_(1, top_k_indices_b, top_k_probs_b.to(lora_B_out.dtype))

        # 7. Weighted sum over expert dimension
        moe_diff = (lora_B_out * gate_weights.unsqueeze(-1)).sum(dim=1)

        # Add base output and MoE contribution
        result_flat = base_flat + moe_diff
        return result_flat.reshape(*base_out.shape)

    def set_lora_mode(self, mode: str = "lora"):
        """Dynamically switch between LoRA and LoRA-MoE modes.

        Args:
            mode: "lora" for standard LoRA, "lora_moe" for LoRA-MoE
        """
        if mode == "lora":
            # Disable MoE mode
            for module in self.qwen3_vl_with_expert.modules():
                if hasattr(module, 'set_lora_moe_mode'):
                    module.set_lora_moe_mode(False)
            self._moe_mode = False
            logging.info("Switched to standard LoRA mode")

        elif mode == "lora_moe":
            # Enable MoE mode
            for module in self.qwen3_vl_with_expert.modules():
                if hasattr(module, 'set_lora_moe_mode'):
                    module.set_lora_moe_mode(True)
            self._moe_mode = True
            logging.info("Switched to LoRA-MoE mode")

    def set_requires_grad(self):
        # If using LoRA or LoRA-MoE, LoRA parameters + action head should be trainable
        if self._is_any_lora_mode_enabled():
            # Freeze all base model parameters, but keep action head trainable
            lora_params = 0
            router_params = 0
            action_head_params = 0
            frozen_params = 0

            is_moe = getattr(self, '_moe_mode', False)

            for name, param in self.named_parameters():
                if bool(getattr(self.config, "enable_affordance", False)) and "affordance_" in name:
                    param.requires_grad = True
                    lora_params += param.numel()
                # Action head parameters should be trainable (except bias)
                elif "action_in_proj" in name or "action_out_proj" in name or "action_time_mlp" in name:
                    # Skip bias parameters - they don't need fine-tuning
                    if "bias" in name:
                        param.requires_grad = False
                        frozen_params += param.numel()
                    else:
                        param.requires_grad = True
                        action_head_params += param.numel()
                # state_proj base weight should be frozen, only LoRA params trainable
                elif "state_proj" in name and "lora_" not in name and "router" not in name:
                    # Freeze state_proj base weight and bias
                    param.requires_grad = False
                    frozen_params += param.numel()
                # Router parameters are trainable in LoRA-MoE mode
                # Matches: lora_router (PEFT), state_proj_router (manual implementation)
                # Note: exclude lora_A and lora_B parameters (which contain "router" in some edge cases)
                elif "router" in name and "lora_A" not in name and "lora_B" not in name:
                    param.requires_grad = True
                    router_params += param.numel()
                # LoRA-MoE specific parameters
                elif is_moe and ("lora_A_moe" in name or "lora_B_moe" in name):
                    param.requires_grad = True
                    lora_params += param.numel()
                # Standard LoRA parameters
                elif "lora_" in name or ".lora_" in name:
                    param.requires_grad = True
                    lora_params += param.numel()
                else:
                    param.requires_grad = False
                    frozen_params += param.numel()

            mode_str = "LoRA-MoE" if is_moe else "LoRA"
            logging.info("=" * 60)
            logging.info(f"{mode_str} Mode Enabled - Parameter Summary:")
            logging.info("=" * 60)
            logging.info(f"  [{mode_str}] Trainable parameters: {lora_params:,} ({lora_params/1e6:.2f}M)")
            if is_moe:
                logging.info(f"  [Router] Trainable parameters: {router_params:,} ({router_params/1e6:.2f}M)")
            logging.info(f"  [Action Head] Trainable parameters: {action_head_params:,} ({action_head_params/1e6:.2f}M)")
            logging.info(f"  [Frozen] Base model parameters: {frozen_params:,} ({frozen_params/1e6:.2f}M)")
            total_trainable = lora_params + action_head_params + router_params
            logging.info(f"  [Total] Trainable: {total_trainable:,} ({total_trainable/1e6:.2f}M)")
            logging.info("=" * 60)
            return

        if self.config.freeze_vision_encoder:
            self.qwen3_vl_with_expert.und_expert.visual.eval()
            for params in self.qwen3_vl_with_expert.und_expert.visual.parameters():
                params.requires_grad = False

        if self.config.train_expert_only:
            self.qwen3_vl_with_expert.und_expert.eval()
            for params in self.qwen3_vl_with_expert.und_expert.parameters():
                params.requires_grad = False

        if self.config.train_vlm_only:
            self.qwen3_vl_with_expert.gen_expert.eval()
            for params in self.qwen3_vl_with_expert.gen_expert.parameters():
                params.requires_grad = False
            self.qwen3_vl_with_expert.act_expert.eval()
            for params in self.qwen3_vl_with_expert.act_expert.parameters():
                params.requires_grad = False

        self.cosmos.eval()
        for params in self.cosmos.parameters():
            params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self._is_any_lora_mode_enabled():
            # In LoRA/LoRA-MoE mode, the base model should stay in eval mode
            # Only LoRA layers should be in train mode when training
            # This is handled by peft automatically
            return self

        if self.config.freeze_vision_encoder:
            self.qwen3_vl_with_expert.und_expert.visual.eval()

        if self.config.train_expert_only:
            self.qwen3_vl_with_expert.und_expert.eval()

        self.cosmos.eval()
        return self

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True

        # Try to enable gradient checkpointing on the base model first
        try:
            self.qwen3_vl_with_expert.und_expert.language_model.gradient_checkpointing = True
        except:
            pass
        try:
            self.qwen3_vl_with_expert.und_expert.visual.gradient_checkpointing = True
        except:
            pass
        try:
            self.qwen3_vl_with_expert.gen_expert.gradient_checkpointing = True
        except:
            pass
        try:
            self.qwen3_vl_with_expert.act_expert.gradient_checkpointing = True
        except:
            pass

        # For PEFT/LoRA models, we need to enable checkpointing on the underlying base model
        # The base_model attribute points to the original unwrapped model
        def try_enable_checkpointing(model):
            if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                # PEFT model
                base = model.base_model.model
                if hasattr(base, 'gradient_checkpointing'):
                    base.gradient_checkpointing = True
                    logging.info(f"Enabled gradient checkpointing on PEFT base model")
            elif hasattr(model, 'gradient_checkpointing'):
                model.gradient_checkpointing = True

        # Try to enable on each expert
        try:
            try_enable_checkpointing(self.qwen3_vl_with_expert.und_expert)
        except Exception as e:
            logging.warning(f"Could not enable gradient checkpointing on und_expert: {e}")

        try:
            try_enable_checkpointing(self.qwen3_vl_with_expert.gen_expert)
        except Exception as e:
            logging.warning(f"Could not enable gradient checkpointing on gen_expert: {e}")

        try:
            try_enable_checkpointing(self.qwen3_vl_with_expert.act_expert)
        except Exception as e:
            logging.warning(f"Could not enable gradient checkpointing on act_expert: {e}")

        logging.info("Enabled gradient checkpointing for QwenA1 model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.qwen3_vl_with_expert.und_expert.language_model.gradient_checkpointing = False
        self.qwen3_vl_with_expert.und_expert.visual.gradient_checkpointing = False
        self.qwen3_vl_with_expert.gen_expert.gradient_checkpointing = False
        self.qwen3_vl_with_expert.act_expert.gradient_checkpointing = False
        logging.info("Disabled gradient checkpointing for QwenA1 model")

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, OPENPI_ATTENTION_MASK_VALUE)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(
            self.config.time_sampling_beta_alpha, self.config.time_sampling_beta_beta, bsize, device
        )
        time = time_beta * self.config.time_sampling_scale + self.config.time_sampling_offset
        return time.to(dtype=torch.float32, device=device)

    @dynamo.disable
    def embed_prefix(
        self, pixel_values, image_grid_thw, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_token_id = self.qwen3_vl_with_expert.und_expert.config.image_token_id
        D1 = pixel_values.shape[-1]
        pixel_values = pixel_values.view(-1, D1)
        image_grid_thw = image_grid_thw.view(-1, 3)
        image_embs, _ = self.qwen3_vl_with_expert.und_expert.visual(pixel_values, image_grid_thw)

        embs = self.qwen3_vl_with_expert.und_expert.get_input_embeddings()(lang_tokens)
        B, L, D2 = embs.shape
        embs = embs.view(-1, D2)
        lang_tokens = lang_tokens.view(-1)
        embs[lang_tokens == image_token_id] = image_embs  # replace dummy embeds with image_embeds
        embs = embs.view(B, L, D2)

        pad_masks = lang_masks.to(torch.bool)
        att_masks = torch.zeros_like(pad_masks, dtype=torch.bool, device=pad_masks.device)

        return embs, pad_masks, att_masks

    def get_cosmos_features(self, images):
        shape = images.shape[:-3]
        c, h, w = images.shape[-3:]
        images = images.reshape(-1, c, h, w)
        images = F.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
        images = images * 2 - 1  # [-1, 1]
        features = self.cosmos.encode(images)
        c, h, w = features.shape[-3:]
        features = features.view(*shape, c, h, w)
        return features

    def embed_middle(self, images, img_masks):
        device = images[0].device
        B, N_view, T = images.shape[:3]
        features = self.get_cosmos_features(images)

        # The detached 16-channel latent is small and lets the affordance target
        # preserve spatial change magnitude instead of cancelling motion by
        # subtracting two globally pooled vectors.
        if bool(getattr(self.config, "enable_affordance", False)):
            self._last_current_cosmos_features = features[:, :, -1].detach()
        else:
            self._last_current_cosmos_features = None

        B, N_view, T = features.shape[:3]
        features = rearrange(features, 'b n t c h w -> (b n t) c h w')
        features = self.cosmos_in_proj(features)
        features = self.downsample_conv(features)
        features = rearrange(features, '(b n t) c h w -> b n t c h w', b=B, n=N_view, t=T)
        self.cosmos_feat_shape = features.shape

        B, N_view, T, _, H, W = features.shape
        embs = rearrange(features, 'b n t c h w -> b (n t h w) c', b=B, n=N_view, t=T)
        # pad_masks = torch.ones((B, embs.shape[1]), dtype=torch.bool, device=device)
        pad_masks = torch.zeros((B, N_view, T, H, W), dtype=torch.bool, device=device)
        pad_masks[img_masks] = True
        pad_masks = rearrange(pad_masks, 'b n t h w -> b (n t h w)', b=B, n=N_view, t=T)

        att_masks = [1] + [0] * (embs.shape[1] - 1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(B, len(att_masks))
        return embs, pad_masks, att_masks

    def _compute_affordance_task_context(self, prefix_embs, lang_tokens, lang_masks):
        if not (
            bool(getattr(self.config, "enable_affordance_v2", False))
            or bool(getattr(self.config, "enable_affordance_v3", False))
        ):
            return None
        image_token_id = int(self.qwen3_vl_with_expert.und_expert.config.image_token_id)
        text_mask = lang_masks.to(torch.bool) & lang_tokens.ne(image_token_id)
        for token_name in ("vision_start_token_id", "vision_end_token_id"):
            token_id = getattr(self.qwen3_vl_with_expert.und_expert.config, token_name, None)
            if token_id is not None:
                text_mask = text_mask & lang_tokens.ne(int(token_id))
        # Keep the task context observation-only and prevent the affordance
        # branch from changing the upstream VLM embedding gradients.
        return masked_token_mean(prefix_embs, text_mask).detach()

    def _affordance_spatial_task_logits(self, current_tokens, task_context):
        if (
            not (
                bool(getattr(self.config, "enable_affordance_v2", False))
                or bool(getattr(self.config, "enable_affordance_v3", False))
            )
            or task_context is None
        ):
            return None
        if bool(getattr(self.config, "enable_affordance_v3", False)):
            query_proj = self.affordance_task_query
            key_proj = self.affordance_spatial_key
        else:
            query_proj = self.affordance_v2_task_query
            key_proj = self.affordance_v2_spatial_key
        query = query_proj(task_context.to(dtype=query_proj.weight.dtype)).float()
        key = key_proj(current_tokens.to(dtype=key_proj.weight.dtype)).float()
        return torch.einsum("bvhwd,bd->bvhw", key, query) / math.sqrt(float(query.shape[-1]))

    def compute_affordance(self, middle_embs, middle_pad_masks, task_context=None):
        if not bool(getattr(self.config, "enable_affordance", False)):
            self._last_affordance_spatial_factor = None
            self._last_affordance_view_mask = None
            self._last_affordance_spatial_entropy = 0.0
            self._last_affordance_spatial_peak = 0.0
            return None
        batch_size, num_views, num_frames, channels, height, width = self.cosmos_feat_shape
        affordance_tokens = (
            middle_embs.detach()
            if bool(getattr(self.config, "enable_affordance_v3", False))
            else middle_embs
        )
        spatial_tokens = affordance_tokens.reshape(
            batch_size,
            num_views,
            num_frames,
            height,
            width,
            channels,
        )
        spatial_mask = middle_pad_masks.reshape(
            batch_size,
            num_views,
            num_frames,
            height,
            width,
        )
        current_tokens = spatial_tokens[:, :, -1]
        current_mask = spatial_mask[:, :, -1]
        spatial_logits = self.affordance_spatial_head(
            current_tokens.to(dtype=self.affordance_spatial_head.weight.dtype)
        ).squeeze(-1)
        task_logits = self._affordance_spatial_task_logits(current_tokens, task_context)
        if task_logits is not None:
            spatial_logits = spatial_logits + task_logits.to(dtype=spatial_logits.dtype)
        spatial_factor = masked_spatial_softmax(spatial_logits, current_mask)
        spatial_pooled = (
            current_tokens.float() * spatial_factor.to(current_tokens.device)[:, :, :, :, None]
        ).sum(dim=(1, 2, 3))
        temporal_pooled = masked_token_mean(affordance_tokens, middle_pad_masks).float()
        pooled = 0.5 * (temporal_pooled + spatial_pooled)

        motion_pooled = None
        if bool(getattr(self.config, "enable_affordance_v2", False)) and num_frames >= 2:
            previous_tokens = spatial_tokens[:, :, -2]
            previous_mask = spatial_mask[:, :, -2]
            motion_mask = current_mask & previous_mask
            motion_weights = spatial_factor * motion_mask.to(spatial_factor.dtype)
            motion_weights = motion_weights / motion_weights.sum(
                dim=(1, 2, 3), keepdim=True
            ).clamp_min(1e-12)
            motion_pooled = (
                (current_tokens.float() - previous_tokens.float())
                * motion_weights.to(current_tokens.device)[:, :, :, :, None]
            ).sum(dim=(1, 2, 3))

        self._last_affordance_spatial_factor = spatial_factor
        self._last_affordance_view_mask = current_mask.any(dim=(-1, -2))
        flat_factor = spatial_factor.flatten(start_dim=1)
        valid_count = current_mask.flatten(start_dim=1).sum(dim=-1).clamp_min(2).float()
        entropy = -(flat_factor * flat_factor.clamp_min(1e-12).log()).sum(dim=-1) / valid_count.log()
        if not torch.compiler.is_compiling():
            self._last_affordance_spatial_entropy = float(entropy.detach().mean().item())
            self._last_affordance_spatial_peak = float(flat_factor.detach().max(dim=-1).values.mean().item())

        bridge_input = pooled.to(dtype=self.affordance_bridge[1].weight.dtype)
        affordance = self.affordance_bridge(bridge_input)
        if bool(getattr(self.config, "enable_affordance_v2", False)):
            if motion_pooled is None:
                motion_pooled = torch.zeros_like(pooled)
            if task_context is None:
                task_context = torch.zeros(
                    pooled.shape[0],
                    self.affordance_v2_task_context_proj.in_features,
                    device=pooled.device,
                    dtype=pooled.dtype,
                )
            affordance = affordance + self.affordance_v2_motion_proj(
                motion_pooled.to(dtype=self.affordance_v2_motion_proj.weight.dtype)
            )
            affordance = affordance + self.affordance_v2_task_context_proj(
                task_context.to(dtype=self.affordance_v2_task_context_proj.weight.dtype)
            )
        return F.normalize(affordance.float(), dim=-1)

    def get_last_affordance_spatial_factor(self):
        if self._last_affordance_spatial_factor is None:
            return None
        return self._last_affordance_spatial_factor.detach()

    def compute_affordance_loss(self, affordance, future_embs):
        if (
            affordance is None
            or self._last_current_cosmos_features is None
            or self._last_affordance_spatial_factor is None
        ):
            self._last_affordance_target_valid = 0.0
            return future_embs.new_zeros(())
        current_features = self._last_current_cosmos_features.to(future_embs.device, dtype=torch.float32)
        spatial_factor = self._last_affordance_spatial_factor
        target_height, target_width = future_embs.shape[-2:]
        if spatial_factor.shape[-2:] != (target_height, target_width):
            batch_size, num_views = spatial_factor.shape[:2]
            spatial_factor = F.interpolate(
                spatial_factor.reshape(-1, 1, *spatial_factor.shape[-2:]),
                size=(target_height, target_width),
                mode="bilinear",
                align_corners=False,
            ).reshape(batch_size, num_views, target_height, target_width)
        view_mask = self._last_affordance_view_mask.to(device=spatial_factor.device, dtype=torch.bool)
        spatial_factor = spatial_factor * view_mask[:, :, None, None].to(spatial_factor.dtype)
        spatial_factor = spatial_factor / spatial_factor.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
        predicted_effect = self.affordance_delta_head(
            affordance.to(self.affordance_delta_head.weight.dtype)
        )
        loss_fn = (
            factorized_signed_future_effect_cosine_loss
            if bool(getattr(self.config, "enable_affordance_v3", False))
            else factorized_future_effect_cosine_loss
        )
        loss, valid_fraction = loss_fn(
            spatial_factor,
            predicted_effect,
            current_features,
            future_embs,
            view_mask=view_mask,
        )
        self._last_affordance_target_valid = float(valid_fraction.detach().item())
        return loss

    def _affordance_action_residual(self, state, affordance, action_length, dtype):
        if bool(getattr(self.config, "enable_affordance_v3", False)):
            self._last_affordance_action_gate = 0.0
            return torch.zeros(
                affordance.shape[0],
                1,
                self.action_in_proj.out_features,
                device=affordance.device,
                dtype=dtype,
            )
        affordance_action = self.affordance_action_proj(
            affordance.to(dtype=self.affordance_action_proj.weight.dtype)
        )
        if not bool(getattr(self.config, "enable_affordance_v2", False)):
            self._last_affordance_action_gate = 1.0
            return affordance_action[:, None, :].to(dtype)

        state_summary = state.reshape(state.shape[0], -1, state.shape[-1]).float().mean(dim=1)
        gate_dtype = self.affordance_v2_action_gate.weight.dtype
        gate_input = torch.cat([affordance.float(), state_summary], dim=-1).to(dtype=gate_dtype)
        gate = torch.sigmoid(self.affordance_v2_action_gate(gate_input))
        horizon_mask = make_affordance_action_horizon_mask(
            action_length,
            int(getattr(self.config, "affordance_action_horizon", 15)),
            int(getattr(self.config, "affordance_action_decay_end", 30)),
            device=affordance_action.device,
            dtype=affordance_action.dtype,
        )
        self._last_affordance_action_gate = gate.detach().mean()
        return (
            affordance_action[:, None, :]
            * gate.to(dtype=affordance_action.dtype)[:, None, :]
            * horizon_mask
        ).to(dtype)

    def embed_suffix(self, state, noisy_actions, timestep, affordance=None):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        self._set_affordance_v3_router_prior(affordance)

        if self.state_proj.weight.dtype == torch.float32:
            state = state.to(torch.float32)

        def state_proj_func(state):
            # Base output
            output = self.state_proj(state)
            # Add LoRA output if using standard LoRA (not MoE)
            if self.config.use_lora and not getattr(self, '_moe_mode', False):
                if hasattr(self, 'state_proj_lora_A'):
                    lora_a_out = self.state_proj_lora_A(state)
                    lora_b_out = self.state_proj_lora_B(lora_a_out)
                    output = output + lora_b_out * self.state_proj_lora_scaling
            return output

        # Handle state_proj based on mode
        if getattr(self, '_moe_mode', False) and hasattr(self, 'state_proj_lora_moe_forward'):
            # Use optimized token-wise dispatching (method handles base + MoE)
            if getattr(self, '_use_ab_routing', False) and hasattr(self, 'state_proj_lora_moe_forward_ab'):
                state_emb = self.state_proj_lora_moe_forward_ab(state, affordance=affordance)
            else:
                state_emb = self.state_proj_lora_moe_forward(state, affordance=affordance)
        else:
            state_emb = self._apply_checkpoint(state_proj_func, state)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1]

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=self.config.min_period,
            max_period=self.config.max_period,
            device=timestep.device,
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        def mlp_func(action_time_emb):
            x = self.action_time_mlp_in(action_time_emb)
            x = F.silu(x)
            return self.action_time_mlp_out(x)

        action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)

        if affordance is not None and bool(getattr(self.config, "enable_affordance", False)):
            action_time_emb = action_time_emb + self._affordance_action_residual(
                state,
                affordance,
                action_time_emb.shape[1],
                action_time_emb.dtype,
            )

        embs.append(action_time_emb)
        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.chunk_size - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def get_position_ids(self, lang_tokens, image_grid_thw, pad_masks):
        L = lang_tokens.shape[1]
        pseudo_avail_token_id = 777
        padded_lang_tokens = torch.ones_like(pad_masks).to(lang_tokens) * pseudo_avail_token_id
        padded_lang_tokens[:, :L] = lang_tokens
        attention_mask = pad_masks.to(lang_tokens)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.view(-1, 3)

        # Get the underlying model for get_rope_index (handle PEFT wrapping)
        und_expert = self.qwen3_vl_with_expert.und_expert
        if self._is_any_lora_mode_enabled() and hasattr(und_expert, 'base_model'):
            # PEFT wraps the model, access the underlying Qwen3VLModel
            # und_expert -> base_model (PeftModel) -> model (Qwen3VLForCausalLM) -> model (Qwen3VLModel)
            und_expert_model = und_expert.base_model.model.model
        else:
            und_expert_model = und_expert.model

        position_ids, rope_deltas = und_expert_model.get_rope_index(
            padded_lang_tokens,
            image_grid_thw,
            attention_mask=attention_mask,
        )
        return position_ids, rope_deltas

    def _get_masked_view_cached_positions(
        self,
        lang_tokens,
        image_grid_thw,
        prefix_pad_masks,
        middle_pad_masks,
        suffix_length,
    ):
        """Compute middle/suffix RoPE positions with the training-time mask."""
        prefix_length = prefix_pad_masks.shape[1]
        middle_length = middle_pad_masks.shape[1]
        suffix_pad_masks = torch.ones(
            prefix_pad_masks.shape[0],
            suffix_length,
            dtype=torch.bool,
            device=prefix_pad_masks.device,
        )
        full_pad_masks = torch.cat(
            [prefix_pad_masks, middle_pad_masks, suffix_pad_masks], dim=1
        )
        full_position_ids, _ = self.get_position_ids(
            lang_tokens,
            image_grid_thw,
            full_pad_masks,
        )
        return (
            full_position_ids[:, :, prefix_length : prefix_length + middle_length],
            full_position_ids[:, :, prefix_length + middle_length :],
        )

    def decode_cosmos(self, features):
        b, n, t, c, h, w =self.cosmos_feat_shape
        features = rearrange(features, 'b (n t h w) c -> b n t c h w', b=b, n=n, t=t, h=h, w=w)
        features = features.mean(2)  # b n c h w
        features = rearrange(features, 'b n c h w -> (b n) c h w')

        features = self.upsample_conv(features) # dtype: torch.float32
        h_upsampled, w_upsampled = features.shape[-2:]
        features = features.permute(0, 2, 3, 1)
        features = features.reshape(b * n, -1, c)
        features = self.cosmos_out_proj(self.cosmos_out_layer_norm(features)) # dtype: torch.float32
        features = features.view(b, n, h_upsampled, w_upsampled, features.shape[-1])
        features = features.permute(0, 1, 4, 2, 3)
        return features

    def forward(
        self,
        images,
        img_masks,
        pixel_values,
        image_grid_thw,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
    ) -> Tensor:
        """Do a full training forward pass and compute the loss."""
        import time as _time
        _t0 = _time.perf_counter()

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        _t1 = _time.perf_counter()
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pixel_values, image_grid_thw, lang_tokens, lang_masks
        )
        middle_embs, middle_pad_masks, middle_att_masks = self.embed_middle(
            images[:, :, :2], img_masks,  # remove the future observation
        )
        task_context = self._compute_affordance_task_context(
            prefix_embs,
            lang_tokens,
            lang_masks,
        )
        affordance = self.compute_affordance(
            middle_embs,
            middle_pad_masks,
            task_context=task_context,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, time, affordance=affordance
        )
        _t2 = _time.perf_counter()

        # # ========== DEBUG: Generation Expert Input/Output ==========
        # print("\n" + "="*60)
        # print("=== Generation Expert Debug Info ===")
        # print(f"images input shape: {images.shape}")  # [B, N_view, T_total, C, H, W]
        # print(f"images[:, :, :2] (Generation input) shape: {images[:, :, :2].shape}")
        # print(f"images[:, :, 2] (Future target) shape: {images[:, :, 2].shape}")
        # print(f"middle_embs shape: {middle_embs.shape}")
        # print(f"cosmos_feat_shape: {getattr(self, 'cosmos_feat_shape', 'Not set yet')}")
        # # ============================================================

        if (
            self.qwen3_vl_with_expert.und_expert.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            middle_embs = middle_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, middle_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, middle_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids, rope_deltas = self.get_position_ids(lang_tokens, image_grid_thw, pad_masks)

        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
        prune_requested = bool(getattr(self.config, "use_visual_token_prune", False)) and bool(
            getattr(self, "_moe_mode", False)
        )
        token_prune_plan = None
        if prune_requested:
            token_prune_plan = build_visual_token_prune_plan(
                lang_tokens=lang_tokens,
                prefix_pad_masks=prefix_pad_masks,
                image_token_id=self.qwen3_vl_with_expert.und_expert.config.image_token_id,
                cfg=self.config,
            )
            _print_visual_token_prune_plan_status(
                phase="train",
                plan=token_prune_plan,
                cfg=self.config,
                lang_tokens=lang_tokens,
                prefix_pad_masks=prefix_pad_masks,
                image_token_id=self.qwen3_vl_with_expert.und_expert.config.image_token_id,
            )
        ot_requested = prune_requested
        if self.training and ot_requested and token_prune_plan is None:
            if not getattr(self, "_warned_visual_token_prune_ot_missing_plan", False):
                logging.warning(
                    "[VisualTokenPruneOT] OT loss is enabled, but no prune plan was built. "
                    "Check that visual image tokens are present in the prefix.",
                )
                self._warned_visual_token_prune_ot_missing_plan = True

        def forward_func(prefix_embs, middle_embs, suffix_embs, att_2d_masks_4d, position_ids):
            (_, middle_out, suffix_out), _ = self.qwen3_vl_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, middle_embs, suffix_embs],
                use_cache=False,
            )
            return middle_out, suffix_out

        _t3 = _time.perf_counter()
        if token_prune_plan is None:
            middle_out, suffix_out = self._apply_checkpoint(
                forward_func, prefix_embs, middle_embs, suffix_embs, att_2d_masks_4d, position_ids
            )
            visual_token_prune_ot_loss = None
            visual_token_prune_ot_terms = 0
        else:
            token_prune_plan.reset_ot_losses()
            with apply_visual_token_prune_to_loramoe_modules(self.qwen3_vl_with_expert, token_prune_plan):
                middle_out, suffix_out = self._apply_checkpoint(
                    forward_func, prefix_embs, middle_embs, suffix_embs, att_2d_masks_4d, position_ids
                )
            visual_token_prune_ot_loss = token_prune_plan.get_ot_loss()
            visual_token_prune_ot_terms = token_prune_plan.get_ot_loss_count()
            prune_stats = token_prune_plan.get_prune_stats()
            if prune_stats["module_calls"] == 0:
                raise RuntimeError(
                    "Visual token pruning found LoRA-MoE modules, but their patched forward was never called. "
                    "Verify the installed PEFT Linear.forward dispatch and LoRA-MoE mode."
                )
            if self.training and ot_requested and visual_token_prune_ot_terms == 0:
                if prune_stats["visual_pruned"] == 0:
                    if not getattr(self, "_logged_visual_token_prune_kept_all", False):
                        logging.info(
                            "[VisualTokenPrune] Router uncertainty kept all visual tokens; "
                            "no transport-consistency term is expected for this step."
                        )
                        self._logged_visual_token_prune_kept_all = True
                elif not getattr(self, "_warned_visual_token_prune_ot_no_terms", False):
                    logging.warning(
                        "[VisualTokenPruneOT] Visual tokens were pruned, but no LoRA-MoE layer produced "
                        "a transport-consistency term."
                    )
                    self._warned_visual_token_prune_ot_no_terms = True
        self._visual_token_prune_ot_loss = visual_token_prune_ot_loss
        self._visual_token_prune_ot_terms = visual_token_prune_ot_terms
        self._visual_token_prune_plan_built = token_prune_plan is not None
        self._visual_token_prune_stats = (
            token_prune_plan.get_prune_stats() if token_prune_plan is not None else {}
        )
        _t4 = _time.perf_counter()

        def cosmos_out_func(middle_out):
            return self.decode_cosmos(middle_out)

        pred_cosmos_features = self._apply_checkpoint(cosmos_out_func, middle_out.to(dtype=torch.float32))
        _t5 = _time.perf_counter()

        future_embs = self.get_cosmos_features(images[:, :, 2])
        loss_gen = F.mse_loss(pred_cosmos_features[img_masks], future_embs.to(dtype=torch.float32)[img_masks])
        self._last_affordance_loss = self.compute_affordance_loss(affordance, future_embs)

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        _t6 = _time.perf_counter()

        loss_action = F.mse_loss(u_t, v_t, reduction="none")
        self._last_affordance_action_align_loss = self._compute_affordance_v3_action_align_loss(
            loss_action
        )
        _t_total = _time.perf_counter()

        # Print timing
        self._fw_timing = {
            'noise_sample': (_t1 - _t0) * 1000,
            'embedding': (_t2 - _t1) * 1000,
            'transformer': (_t4 - _t3) * 1000,
            'cosmos': (_t5 - _t4) * 1000,
            'action_proj': (_t6 - _t5) * 1000,
            'total': (_t_total - _t0) * 1000,
        }

        return loss_action, loss_gen

    @torch.no_grad()  # see openpi `sample_actions` (slightly adapted)
    def sample_actions(
        self,
        images,
        img_masks,
        pixel_values,
        image_grid_thw,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        num_steps=None,
        decode_image=False,
    ) -> Tensor:
        """Do a full inference forward and compute the action."""
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        bsize = state.shape[0]
        device = state.device
        dtype = state.dtype

        if noise is None:
            # Sample noise with padded dimension as expected by action_in_proj
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim,
            )  # Use config max_action_dim for internal processing
            noise = self.sample_noise(actions_shape, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            pixel_values, image_grid_thw, lang_tokens, lang_masks
        )
        prefix_position_ids, rope_deltas = self.get_position_ids(lang_tokens, image_grid_thw, prefix_pad_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)

        self.qwen3_vl_with_expert.und_expert.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        prune_requested = bool(getattr(self.config, "use_visual_token_prune", False)) and bool(
            getattr(self, "_moe_mode", False)
        )
        token_prune_plan = None
        if prune_requested:
            token_prune_plan = build_visual_token_prune_plan(
                lang_tokens=lang_tokens,
                prefix_pad_masks=prefix_pad_masks,
                image_token_id=self.qwen3_vl_with_expert.und_expert.config.image_token_id,
                cfg=self.config,
            )
            _print_visual_token_prune_plan_status(
                phase="eval",
                plan=token_prune_plan,
                cfg=self.config,
                lang_tokens=lang_tokens,
                prefix_pad_masks=prefix_pad_masks,
                image_token_id=self.qwen3_vl_with_expert.und_expert.config.image_token_id,
            )

        if prune_requested and token_prune_plan is None:
            if not getattr(self, "_warned_inference_visual_token_prune_missing_plan", False):
                logging.warning(
                    "use_visual_token_prune=True during inference, but no token prune plan was built. "
                    "Check that visual image tokens are present in the prefix."
                )
                self._warned_inference_visual_token_prune_missing_plan = True

        def prefix_forward():
            return self.qwen3_vl_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None, None],
                use_cache=True,
            )

        if token_prune_plan is None:
            _, past_key_values = prefix_forward()
        else:
            with apply_visual_token_prune_to_loramoe_modules(self.qwen3_vl_with_expert, token_prune_plan):
                _, past_key_values = prefix_forward()
        max_prefix_position_ids = prefix_position_ids.max(dim=-1, keepdim=True).values

        middle_embs, middle_pad_masks, middle_att_masks = self.embed_middle(
            images[:, :, :2], img_masks,
        )
        task_context = self._compute_affordance_task_context(
            prefix_embs,
            lang_tokens,
            lang_masks,
        )
        affordance = self.compute_affordance(
            middle_embs,
            middle_pad_masks,
            task_context=task_context,
        )

        middle_len = middle_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, middle_len, prefix_len)
        middle_att_2d_masks = make_att_2d_masks(middle_pad_masks, middle_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, middle_att_2d_masks], dim=2)

        middle_position_ids = torch.arange(1, middle_len + 1).repeat(3, 1, 1).to(max_prefix_position_ids) + max_prefix_position_ids
        suffix_position_ids = None
        view_valid = img_masks.reshape(img_masks.shape[0], img_masks.shape[1], -1).any(dim=2)
        masked_view = view_valid.any(dim=1) & ~view_valid.all(dim=1)
        if masked_view.any():
            # Match training for partially masked rows; preserve full-view legacy positions.
            suffix_len = 1 + self.config.chunk_size
            fixed_middle_position_ids, fixed_suffix_position_ids = self._get_masked_view_cached_positions(
                lang_tokens,
                image_grid_thw,
                prefix_pad_masks,
                middle_pad_masks,
                suffix_len,
            )
            suffix_position_ids = torch.arange(1, suffix_len + 1).repeat(3, 1, 1).to(max_prefix_position_ids) + middle_position_ids.max(dim=-1, keepdim=True).values
            suffix_position_ids = torch.where(
                masked_view[None, :, None],
                fixed_suffix_position_ids,
                suffix_position_ids,
            )
            middle_position_ids = torch.where(
                masked_view[None, :, None],
                fixed_middle_position_ids,
                middle_position_ids,
            )

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.qwen3_vl_with_expert.gen_expert.config._attn_implementation = "eager"  # noqa: SLF001

        (_, middle_out, _), past_key_values = self.qwen3_vl_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=middle_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, middle_embs, None],
            use_cache=True,
        )

        max_position_ids = middle_position_ids.max(dim=-1, keepdim=True).values
        curr_pad_masks = torch.cat([prefix_pad_masks, middle_pad_masks], dim=1)

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                curr_pad_masks,
                past_key_values,
                max_position_ids,
                x_t.to(dtype),
                expanded_time.to(dtype),
                affordance=affordance,
                position_ids=suffix_position_ids,
            )
            x_t = x_t + dt * v_t
            time += dt

        if decode_image:
            def cosmos_out_func(middle_out):
                return self.decode_cosmos(middle_out)
            pred_cosmos_features = self._apply_checkpoint(cosmos_out_func, middle_out.to(dtype=torch.bfloat16))
            pred_cosmos_features = pred_cosmos_features.squeeze(0)
            recon_images = self.cosmos.decode(pred_cosmos_features.squeeze(0))
        else:
            recon_images = None

        return x_t, recon_images

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        max_prefix_position_ids,
        x_t,
        timestep,
        affordance=None,
        position_ids=None,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            state, x_t, timestep, affordance=affordance
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        if position_ids is None:
            # Preserve the original three-view inference path exactly.
            position_ids = torch.arange(1, suffix_len + 1).repeat(3, 1, 1).to(max_prefix_position_ids) + max_prefix_position_ids

        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.qwen3_vl_with_expert.act_expert.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.qwen3_vl_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, None, suffix_embs],
            use_cache=False,
        )

        suffix_out = outputs_embeds[2]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


class QwenA1Policy(PreTrainedPolicy):
    """InternVLA-A1-3B (Qwen-A1) Policy for LeRobot."""

    config_class = QwenA1Config
    name = "qwena1"

    def get_aux_loss(self):
        """Collect and sum auxiliary (load-balancing) losses from all LoRA-MoE layers.

        Traverses all modules in the QwenA1 model (PEFT-injected LoRA layers and
        manually implemented state_proj MoE) and returns the mean aux_loss.

        Returns:
            Scalar tensor with mean aux_loss across all MoE layers.
        """
        total_aux_loss = 0.0
        count = 0

        # Collect from PEFT-injected LoRA layers (und_expert, gen_expert, act_expert)
        for module in self.model.qwen3_vl_with_expert.modules():
            if hasattr(module, 'aux_loss') and isinstance(module.aux_loss, torch.Tensor):
                total_aux_loss = total_aux_loss + module.aux_loss
                count += 1

        # Collect from manually implemented state_proj MoE
        if hasattr(self.model, 'state_proj_aux_loss') and isinstance(self.model.state_proj_aux_loss, torch.Tensor):
            total_aux_loss = total_aux_loss + self.model.state_proj_aux_loss
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        return total_aux_loss / count

    def __init__(
        self,
        config: QwenA1Config,
    ):
        """
        Args:
            config: Policy configuration class instance.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = QwenA1(config)
        self._last_pcea_scores: dict[str, Tensor] = {}
        self._last_pcea_counts: dict[str, int] = {}
        self._warned_fedforesight_unsupported = False

        # Note: LoRA will be applied in from_pretrained after weights are loaded
        # This is done to ensure pretrained weights are loaded correctly before LoRA modifies the model structure

        # NOTE: gradient_checkpointing will be enabled AFTER LoRA is applied below
        # (moved to after _apply_lora for proper PEFT compatibility)

        self.model.to(config.device)

        self.reset()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: QwenA1Config | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ):
        """
        Override from_pretrained to apply LoRA AFTER weights are loaded.
        This ensures pretrained weights load correctly before LoRA modifies the model structure.
        """
        # Call parent's from_pretrained which will:
        # 1. Create instance via __init__ (without LoRA)
        # 2. Load pretrained weights
        # 3. Return the model with weights loaded
        policy = super().from_pretrained(
            pretrained_name_or_path,
            config=config,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            token=token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            strict=strict,
            **kwargs,
        )

        # Now apply LoRA or LoRA-MoE AFTER weights are loaded
        logging.info(f"[from_pretrained]  Checking LoRA config: hasattr(config)={hasattr(policy, 'config')}")
        if hasattr(policy, 'config'):
            logging.info(f"[from_pretrained]  policy.config type: {type(policy.config)}")
            logging.info(
                f"[from_pretrained]  use_lora={policy.config.use_lora}, "
                f"use_lora_moe={policy.config.use_lora_moe}, "
                f"use_lora_moe_forced_last={getattr(policy.config, 'use_lora_moe_forced_last', False)}"
            )
            logging.info(
                f"[from_pretrained] passed config use_lora={config.use_lora if config else 'N/A'}, "
                f"use_lora_moe={config.use_lora_moe if config else 'N/A'}, "
                f"use_lora_moe_forced_last={getattr(config, 'use_lora_moe_forced_last', 'N/A') if config else 'N/A'}"
            )
        else:
            logging.info("[from_pretrained]  policy has no config attribute")

        if hasattr(policy, 'config') and policy.model._is_any_lora_mode_enabled():
            if not PEFt_AVAILABLE:
                raise ImportError("peft is required for LoRA/LoRA-MoE training. Please install it with: pip install peft")
            # Need to re-enable training mode for LoRA application
            policy.model.train()
            logging.info(
                f"[from_pretrained]  Entering LoRA branch, use_lora={policy.config.use_lora}, "
                f"use_lora_moe={policy.config.use_lora_moe}, "
                f"use_lora_moe_forced_last={getattr(policy.config, 'use_lora_moe_forced_last', False)}"
            )
            if getattr(policy.config, 'use_lora_moe_forced_last', False):
                logging.info(
                    f"[from_pretrained]  Calling _apply_lora_moe_forced_last with base_num_experts="
                    f"{policy.config.loramoe_num_experts}"
                )
                policy.model._apply_lora_moe_forced_last()
                policy.model._lora_applied = True
                logging.info(f"Applied LoRA-MoE Forced-Last after loading pretrained weights from {pretrained_name_or_path}")
            elif policy.config.use_lora_moe:
                logging.info(f"[from_pretrained]  Calling _apply_lora_moe with num_experts={policy.config.loramoe_num_experts}")
                policy.model._apply_lora_moe()
                policy.model._lora_applied = True
                logging.info(f"Applied LoRA-MoE after loading pretrained weights from {pretrained_name_or_path}")
            elif policy.config.use_lora:
                logging.info(f"[from_pretrained]  Calling _apply_lora (standard LoRA)")
                policy.model._apply_lora()
                policy.model._lora_applied = True
                logging.info(f"Applied LoRA after loading pretrained weights from {pretrained_name_or_path}")
            # CRITICAL: Call set_requires_grad again AFTER LoRA/LoRA-MoE is applied
            # This is needed because set_requires_grad was called in QwenA1.__init__
            # BEFORE LoRA was applied, so LoRA parameters were not set to trainable
            policy.model.set_requires_grad()

        # Apply PTQ only AFTER pretrained weights are loaded and only for native structure.
        if hasattr(policy, 'config') and policy.config.use_ptq:
            if policy.model._is_any_lora_mode_enabled():
                raise ValueError("PTQ currently supports only native expert structure. Disable LoRA/LoRA-MoE first.")
            logging.info("[from_pretrained]  Applying PTQ after checkpoint loading")
            policy.model.apply_ptq()

        # Enable gradient checkpointing AFTER LoRA is applied (for proper PEFT compatibility)
        if hasattr(policy, 'config') and policy.config.gradient_checkpointing:
            policy.model.gradient_checkpointing_enable()
            logging.info("Enabled gradient checkpointing after LoRA application")

        return policy

    def __str__(self) -> str:
        lines = []

        # ---- basic info ----
        lines.append("=" * 60)
        lines.append(f"Policy: {self.__class__.__name__}")
        lines.append("")

        # ---- parameter counts ----
        num_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_trainable_params = sum(p.numel() for p in self.parameters())

        num_und = sum(p.numel() for p in self.model.qwen3_vl_with_expert.und_expert.parameters())
        num_gen = sum(p.numel() for p in self.model.qwen3_vl_with_expert.gen_expert.parameters())
        num_act = sum(p.numel() for p in self.model.qwen3_vl_with_expert.act_expert.parameters())

        lines.append("Parameter statistics:")
        lines.append(f"  - Total params        : {num_total_params} ({format_big_number(num_total_params)})")
        lines.append(f"  - Trainable params    : {num_trainable_params} ({format_big_number(num_trainable_params)})")
        lines.append(f"  - Und params          : {num_und} ({format_big_number(num_und)})")
        lines.append(f"  - Gen params          : {num_gen} ({format_big_number(num_gen)})")
        lines.append(f"  - Act params          : {num_act} ({format_big_number(num_act)})")

        lines.append("=" * 60)

        return "\n".join(lines)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.model.cosmos.to(torch.bfloat16)
        self.model.action_out_proj.to(torch.float32)
        return self

    def get_optim_params(self):
        """Return only trainable parameters (requires_grad=True) for optimizer."""
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in self.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)
        logging.info(f"[get_optim_params] Total params: {total_params:,}, Trainable: {trainable_count:,} ({trainable_count/total_params*100:.2f}%)")
        return trainable_params

    def reset(self):
        """Reset internal state - called when environment resets."""
        self._action_queue = deque(maxlen=self.config.n_action_steps)
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    def _preprocess_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
        """Preprocess images for the model.
        """
        images = []
        img_masks = []

        for img_idx in range(3):
            img = batch[f"{OBS_IMAGES}.image{img_idx}"]
            mask = batch[f"{OBS_IMAGES}.image{img_idx}_mask"]

            images.append(img)
            img_masks.append(mask)

        images = torch.stack(images, dim=1)  # B, N_view, T, C, H, W
        img_masks = torch.stack(img_masks, dim=1)

        return images, img_masks

    def prepare_state(self, batch):
        """Pad state"""
        state = pad_vector(batch[OBS_STATE], self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def prepare_gen_features(self, batch):
        images = torch.stack([batch[f"{OBS_IMAGES}.image{i}"] for i in range(3)], dim=1)  # B, N_view, T, C, H, W
        B, N_view, T = images.shape[:3]
        images = rearrange(images, 'b n t c h w -> (b n t) c h w')
        images = F.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
        images = images * 2 - 1  # [-1, 1]
        features = self.model.cosmos.encode(images)
        features = rearrange(features, '(b n t) c h w -> b n t c h w', b=B, n=N_view, t=T)
        return features

    def _fedforesight_module_key(self, name: str) -> tuple[str | None, str | None]:
        if name.startswith("und_expert.") or ".und_expert." in name:
            branch = "und"
        elif name.startswith("gen_expert.") or ".gen_expert." in name:
            branch = "gen"
        elif name.startswith("act_expert.") or ".act_expert." in name:
            branch = "act"
        else:
            return None, None

        marker = ".layers."
        idx = name.find(marker)
        if idx < 0:
            idx = name.find("layers.")
            if idx < 0:
                return None, None
            key = name[idx:]
        else:
            key = name[idx + 1:]

        if key.endswith(".base_layer"):
            key = key[: -len(".base_layer")]
        return branch, key

    def _fedforesight_valid_mask(
        self,
        branch: str,
        num_rows: int,
        batch: dict[str, Tensor],
        device: torch.device,
    ) -> Tensor | None:
        if branch == "und":
            mask = batch.get(f"{OBS_PREFIX}attention_mask")
            if mask is None:
                return None
            mask = mask.to(device=device, dtype=torch.bool).reshape(-1)
        elif branch == "gen":
            feature_shape = getattr(self.model, "cosmos_feat_shape", None)
            image_masks = [batch.get(f"{OBS_IMAGES}.image{i}_mask") for i in range(3)]
            if feature_shape is None or any(mask is None for mask in image_masks):
                return None

            batch_size, num_views, num_frames, _, height, width = feature_shape
            view_time_masks = []
            for mask in image_masks:
                mask = mask.to(device=device, dtype=torch.bool).reshape(batch_size, -1)
                if mask.shape[1] == 1 and num_frames > 1:
                    mask = mask.expand(batch_size, num_frames)
                elif mask.shape[1] < num_frames:
                    return None
                else:
                    mask = mask[:, :num_frames]
                view_time_masks.append(mask)

            view_time_mask = torch.stack(view_time_masks, dim=1)
            if tuple(view_time_mask.shape) != (batch_size, num_views, num_frames):
                return None
            mask = view_time_mask[:, :, :, None, None].expand(
                batch_size,
                num_views,
                num_frames,
                height,
                width,
            ).reshape(-1)
        elif branch == "act":
            batch_size = int(batch[ACTION].shape[0])
            if batch_size <= 0 or num_rows % batch_size != 0:
                return None
            rows_per_sample = num_rows // batch_size
            if rows_per_sample <= 1:
                return None
            token_pos = torch.arange(num_rows, device=device) % rows_per_sample
            mask = token_pos > 0  # Drop the state token; keep action tokens only.
        else:
            return None

        if mask.numel() != num_rows:
            return None
        return mask

    def _fedforesight_pool_router_probs(
        self,
        probs: Tensor,
        sample_idx: Tensor,
        branch: str,
        batch: dict[str, Tensor],
    ) -> tuple[Tensor, Tensor] | None:
        if probs.ndim != 2 or sample_idx.ndim != 1 or probs.shape[0] != sample_idx.shape[0]:
            return None

        device = probs.device
        batch_size = int(batch[ACTION].shape[0])
        if batch_size <= 0:
            return None

        sample_idx = sample_idx.to(device=device, dtype=torch.long)
        if sample_idx.numel() == 0 or int(sample_idx.min().item()) < 0 or int(sample_idx.max().item()) >= batch_size:
            return None

        valid_mask = self._fedforesight_valid_mask(branch, probs.shape[0], batch, device)
        if valid_mask is None:
            return None

        valid_probs = probs.float()[valid_mask]
        valid_sample_idx = sample_idx[valid_mask]
        if valid_probs.numel() == 0 and branch != "act":
            return None

        sums = torch.zeros(batch_size, probs.shape[-1], device=device, dtype=torch.float32)
        counts = torch.zeros(batch_size, device=device, dtype=torch.float32)
        sums.index_add_(0, valid_sample_idx, valid_probs)
        counts.index_add_(0, valid_sample_idx, torch.ones_like(valid_sample_idx, dtype=torch.float32))

        valid_samples = counts > 0
        pooled = sums / counts.clamp_min(1.0)[:, None]
        pooled = pooled / pooled.sum(dim=-1, keepdim=True).clamp_min(float(getattr(self.config, "fard_eps", 1e-8)))
        return pooled, valid_samples

    @staticmethod
    def _fedforesight_js_divergence(p: Tensor, q: Tensor, eps: float) -> Tensor:
        p = p.clamp_min(eps)
        q = q.clamp_min(eps)
        p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
        q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
        m = 0.5 * (p + q)
        return 0.5 * (p * (p.log() - m.log())).sum(dim=-1) + 0.5 * (q * (q.log() - m.log())).sum(dim=-1)

    def _compute_fedforesight_losses(self, batch: dict[str, Tensor], reference_loss: Tensor) -> tuple[Tensor, dict]:
        self._last_pcea_scores = {}
        self._last_pcea_counts = {}
        enabled = bool(getattr(self.config, "enable_fard", False) or getattr(self.config, "enable_pcea", False))
        if not enabled:
            return reference_loss.new_zeros(()), {
                "loss_fard": 0.0,
                "loss_fard_weighted": 0.0,
                "fard_triplets": 0,
                "fard_confidence": 0.0,
                "fard_ug_agreement": 0.0,
                "fard_reliability": 0.0,
                "fard_js_fg_a": 0.0,
            }

        unsupported = (
            bool(getattr(self.config, "loramoe_share_a_across_experts", False))
            or bool(getattr(self.config, "loramoe_ab_routing", False))
        )
        if unsupported:
            raise RuntimeError(
                "FARD/PCEA require standard single-router LoRA-MoE with complete A/B experts"
            )

        grouped: dict[str, dict[str, tuple[str, dict]]] = {}
        for module_name, module in self.model.named_modules():
            observables = getattr(module, "_last_router_observables", None)
            if not observables or "router" not in observables:
                continue
            router_obs = observables["router"]
            probs = router_obs.get("p")
            sample_idx = router_obs.get("sample_idx")
            if probs is None or sample_idx is None:
                continue
            branch, key = self._fedforesight_module_key(module_name)
            if branch is None or key is None:
                continue
            grouped.setdefault(key, {})[branch] = (module_name, router_obs)

        eps = float(getattr(self.config, "fard_eps", 1e-8))
        loss_terms = []
        confidence_terms = []
        agreement_terms = []
        reliability_terms = []
        js_terms = []
        pcea_scores: dict[str, Tensor] = {}
        pcea_counts: dict[str, int] = {}

        for branch_map in grouped.values():
            if not all(branch in branch_map for branch in ("und", "gen", "act")):
                continue

            pooled = {}
            module_names = {}
            sample_valid = {}
            for branch in ("und", "gen", "act"):
                module_name, router_obs = branch_map[branch]
                pooled_result = self._fedforesight_pool_router_probs(
                    router_obs["p"],
                    router_obs["sample_idx"],
                    branch,
                    batch,
                )
                if pooled_result is None:
                    pooled = {}
                    break
                pooled[branch], sample_valid[branch] = pooled_result
                module_names[branch] = module_name

            if not pooled:
                continue

            valid_samples = sample_valid["und"] & sample_valid["gen"] & sample_valid["act"]
            if not bool(valid_samples.any()):
                continue

            q_und = pooled["und"][valid_samples]
            q_gen = pooled["gen"][valid_samples]
            q_act = pooled["act"][valid_samples]

            q_und_detached = q_und.detach()
            q_gen_detached = q_gen.detach()
            teacher_detached, agreement, confidence, reliability = foresight_consensus(
                q_und_detached,
                q_gen_detached,
                eps,
            )
            teacher_detached = teacher_detached.detach()
            agreement = agreement.detach()
            confidence = confidence.detach()
            reliability = reliability.detach()

            q_act_safe = q_act.clamp_min(eps)
            q_act_safe = q_act_safe / q_act_safe.sum(dim=-1, keepdim=True).clamp_min(eps)
            teacher_action_js = self._fedforesight_js_divergence(teacher_detached, q_act_safe, eps)
            loss_terms.append((reliability * teacher_action_js).mean())
            confidence_terms.append(confidence.mean().detach())
            agreement_terms.append(agreement.mean().detach())
            reliability_terms.append(reliability.mean().detach())
            js_terms.append(teacher_action_js.mean().detach())

            consensus = three_path_consensus(
                q_und_detached,
                q_gen_detached,
                q_act_safe.detach(),
                eps,
            ).sum(dim=0)
            if bool(torch.isfinite(consensus).all()) and float(consensus.sum().item()) > 0.0:
                for branch in ("und", "gen", "act"):
                    pcea_scores[module_names[branch]] = consensus.detach().cpu()
                    pcea_counts[module_names[branch]] = int(valid_samples.sum().item())

        if not loss_terms:
            return reference_loss.new_zeros(()), {
                "loss_fard": 0.0,
                "loss_fard_weighted": 0.0,
                "fard_triplets": 0,
                "fard_confidence": 0.0,
                "fard_ug_agreement": 0.0,
                "fard_reliability": 0.0,
                "fard_js_fg_a": 0.0,
            }

        loss_fard = torch.stack(loss_terms).mean()
        lambda_fard = float(getattr(self.config, "lambda_fard", 0.01))
        warmup_rounds = int(getattr(self.config, "fard_warmup_rounds", 0))
        current_round = int(getattr(self.config, "fard_current_round", warmup_rounds))
        if warmup_rounds > 0:
            warmup_scale = min(1.0, max(0.0, float(current_round + 1) / float(warmup_rounds)))
        else:
            warmup_scale = 1.0
        effective_lambda_fard = lambda_fard * warmup_scale
        weighted_fard = effective_lambda_fard * loss_fard if bool(getattr(self.config, "enable_fard", False)) else loss_fard.new_zeros(())
        self._last_pcea_scores = pcea_scores if bool(getattr(self.config, "enable_pcea", False)) else {}
        self._last_pcea_counts = pcea_counts if bool(getattr(self.config, "enable_pcea", False)) else {}

        metrics = {
            "loss_fard": float(loss_fard.detach().item()),
            "loss_fard_weighted": float(weighted_fard.detach().item()),
            "lambda_fard_effective": effective_lambda_fard if bool(getattr(self.config, "enable_fard", False)) else 0.0,
            "fard_triplets": len(loss_terms),
            "fard_confidence": float(torch.stack(confidence_terms).mean().item()) if confidence_terms else 0.0,
            "fard_ug_agreement": float(torch.stack(agreement_terms).mean().item()) if agreement_terms else 0.0,
            "fard_reliability": float(torch.stack(reliability_terms).mean().item()) if reliability_terms else 0.0,
            "fard_js_fg_a": float(torch.stack(js_terms).mean().item()) if js_terms else 0.0,
        }
        return weighted_fard, metrics

    def get_last_pcea_scores(self) -> dict[str, Tensor]:
        return {name: scores.clone() for name, scores in self._last_pcea_scores.items()}

    def get_last_pcea_counts(self) -> dict[str, int]:
        return dict(self._last_pcea_counts)

    def clear_fedforesight_router_caches(self) -> None:
        self._last_pcea_scores = {}
        self._last_pcea_counts = {}
        for module in self.model.qwen3_vl_with_expert.modules():
            if hasattr(module, "_last_router_observables"):
                module._last_router_observables = None

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()

        # Action queue logic for n_action_steps > 1
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            # Transpose to get shape (n_action_steps, batch_size, action_dim)
            self._action_queue.extend(actions.transpose(0, 1))

        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], decode_image=False) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        # Prepare inputs
        pixel_values = batch[f"{OBS_PREFIX}pixel_values"]
        image_grid_thw = batch[f"{OBS_PREFIX}image_grid_thw"]
        lang_tokens = batch[f"{OBS_PREFIX}input_ids"]
        lang_masks = batch[f"{OBS_PREFIX}attention_mask"]
        state = self.prepare_state(batch)

        images, img_masks = self._preprocess_images(batch)

        # Sample actions using the model
        actions, recon_images = self.model.sample_actions(
            images,
            img_masks,
            pixel_values,
            image_grid_thw,
            lang_tokens,
            lang_masks,
            state,
            decode_image=decode_image,
        )

        # Unpad actions to actual action dimension
        original_action_dim = self.config.output_features[ACTION].shape[0]
        actions = actions[:, :, :original_action_dim]

        return actions, recon_images

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training."""

        # Prepare inputs
        pixel_values = batch[f"{OBS_PREFIX}pixel_values"]
        image_grid_thw = batch[f"{OBS_PREFIX}image_grid_thw"]
        lang_tokens = batch[f"{OBS_PREFIX}input_ids"]
        lang_masks = batch[f"{OBS_PREFIX}attention_mask"]

        images, img_masks = self._preprocess_images(batch)

        state = self.prepare_state(batch)
        actions = self.prepare_action(batch)

        losses_action, loss_gen = self.model.forward(
            images,
            img_masks,
            pixel_values,
            image_grid_thw,
            lang_tokens,
            lang_masks,
            state,
            actions,
        )

        # Truncate losses to actual action dimensions
        original_action_dim = self.config.output_features[ACTION].shape[0]
        losses_action = losses_action[:, :, :original_action_dim]
        action_is_pad = batch.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=losses_action.device, dtype=torch.bool)
            if action_is_pad.ndim == 1:
                action_is_pad = action_is_pad.unsqueeze(0)
            if action_is_pad.shape[:2] != losses_action.shape[:2]:
                raise ValueError(
                    "action_is_pad shape does not match action loss: "
                    f"{tuple(action_is_pad.shape)} != {tuple(losses_action.shape[:2])}"
                )
            valid = (~action_is_pad).unsqueeze(-1)
            losses_action = losses_action * valid
            loss_action = losses_action.sum() / valid.expand_as(losses_action).sum().clamp_min(1)
        else:
            loss_action = losses_action.mean()

        # Collect auxiliary (load-balancing) loss from LoRA-MoE layers
        aux_loss = self.get_aux_loss()

        lambda_aux = getattr(self.config, 'lambda_aux', 0.001)
        lambda_ot = getattr(self.config, 'lambda_ot', 0.01)
        ot_loss = getattr(self.model, '_visual_token_prune_ot_loss', None)
        if ot_loss is None:
            ot_loss = loss_action.new_zeros(())
        ot_terms = int(getattr(self.model, '_visual_token_prune_ot_terms', 0))
        prune_stats = getattr(self.model, '_visual_token_prune_stats', {})
        weighted_aux_loss = lambda_aux * aux_loss
        weighted_ot_loss = lambda_ot * ot_loss if ot_terms > 0 else ot_loss.new_zeros(())
        router_regularization = weighted_aux_loss + weighted_ot_loss
        weighted_fard_loss, fedforesight_metrics = self._compute_fedforesight_losses(batch, loss_action)
        affordance_loss = getattr(self.model, "_last_affordance_loss", None)
        if affordance_loss is None:
            affordance_loss = loss_action.new_zeros(())
        lambda_affordance = float(getattr(self.config, "lambda_affordance", 0.001))
        affordance_action_align_loss = getattr(
            self.model,
            "_last_affordance_action_align_loss",
            loss_action.new_zeros(()),
        )
        affordance_objective = (
            affordance_loss + affordance_action_align_loss
            if bool(getattr(self.config, "enable_affordance_v3", False))
            else affordance_loss
        )
        weighted_affordance_loss = (
            lambda_affordance * affordance_objective
            if bool(getattr(self.config, "enable_affordance", False))
            else affordance_objective.new_zeros(())
        )

        loss = (
            loss_action
            + self.config.lambda_gen * loss_gen
            + router_regularization
            + weighted_fard_loss
            + weighted_affordance_loss
        )

        loss_dict = {
            "loss": loss.item(),
            "loss_action": loss_action.item(),
            "loss_gen": loss_gen.item(),
            "loss_aux": aux_loss.item() if torch.is_tensor(aux_loss) else float(aux_loss),
            "loss_aux_weighted": (
                weighted_aux_loss.item()
                if torch.is_tensor(weighted_aux_loss)
                else float(weighted_aux_loss)
            ),
            "loss_router_regularization": (
                router_regularization.item()
                if torch.is_tensor(router_regularization)
                else float(router_regularization)
            ),
            "lambda_aux": lambda_aux,
            "lambda_ot": lambda_ot,
            "lambda_affordance": lambda_affordance,
            "loss_affordance": float(affordance_loss.detach().item()),
            "loss_affordance_weighted": float(weighted_affordance_loss.detach().item()),
            "loss_affordance_action_align": float(affordance_action_align_loss.detach().item()),
            "affordance_target_valid": float(
                getattr(self.model, "_last_affordance_target_valid", 0.0)
            ),
            "affordance_spatial_entropy": float(
                getattr(self.model, "_last_affordance_spatial_entropy", 0.0)
            ),
            "affordance_spatial_peak": float(
                getattr(self.model, "_last_affordance_spatial_peak", 0.0)
            ),
            "affordance_action_gate": float(
                getattr(self.model, "_last_affordance_action_gate", 0.0).detach().item()
                if torch.is_tensor(getattr(self.model, "_last_affordance_action_gate", 0.0))
                else getattr(self.model, "_last_affordance_action_gate", 0.0)
            ),
            "affordance_router_gate": float(
                getattr(self.model, "_last_affordance_router_gate", 0.0).detach().item()
                if torch.is_tensor(getattr(self.model, "_last_affordance_router_gate", 0.0))
                else getattr(self.model, "_last_affordance_router_gate", 0.0)
            ),
            "loss_visual_token_prune_ot": ot_loss.item() if torch.is_tensor(ot_loss) else float(ot_loss),
            "loss_visual_token_prune_ot_weighted": (
                weighted_ot_loss.item()
                if torch.is_tensor(weighted_ot_loss)
                else float(weighted_ot_loss)
            ),
            "visual_token_prune_ot_terms": ot_terms,
            "visual_token_prune_plan_built": bool(getattr(self.model, '_visual_token_prune_plan_built', False)),
            "visual_token_pruned_pct": float(prune_stats.get("visual_pruned_pct", 0.0)),
            "visual_token_adapter_pruned_pct": float(prune_stats.get("adapter_pruned_pct", 0.0)),
        }
        loss_dict.update(fedforesight_metrics)

        losses_action = losses_action.mean(dim=[0, 1]).detach().cpu().numpy().tolist()
        loss_dict.update({
            f"loss_action_dim{i}": losses_action[i] for i in range(original_action_dim)
        })

        return loss, loss_dict


if __name__ == "__main__":
    from lerobot.utils.constants import OBS_IMAGES, OBS_STATE, ACTION
    from lerobot.policies.InternVLA_A1_3B.transform_qwena1 import Qwen3_VLProcessorTransformFn
    from pprint import pp
    torch.manual_seed(0)
    device = torch.device("cuda")

    processor = Qwen3_VLProcessorTransformFn()

    cfg = QwenA1Config()
    cfg.qwen3_vl_variant="qwen3_vl_28l"
    cfg.action_expert_variant="qwen3_28l"
    cfg.freeze_vision_encoder=True
    dtype = torch.float32 if cfg.dtype == 'float32' else torch.bfloat16

    model = QwenA1Policy(cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}  ({total_params / 1e9:.2f}B)")
    print(f"Trainable parameters: {trainable_params:,}  ({trainable_params / 1e9:.2f}B)")
    print(f"Und params: {sum(p.numel() for p in model.model.qwen3_vl_with_expert.und_expert.parameters()) / 1e9:.2f}B")
    print(f"Gen params: {sum(p.numel() for p in model.model.qwen3_vl_with_expert.gen_expert.parameters()) / 1e9:.2f}B")
    print(f"Act params: {sum(p.numel() for p in model.model.qwen3_vl_with_expert.act_expert.parameters()) / 1e9:.2f}B")

    B = 2
    samples = [{
        f"{OBS_IMAGES}.image0": torch.rand((3, 3, 224, 224)),
        f"{OBS_IMAGES}.image1": torch.rand((3, 3, 224, 224)),
        f"{OBS_IMAGES}.image2": torch.rand((3, 3, 224, 224)),
        f"{OBS_IMAGES}.image0_mask": torch.tensor(True).cuda(),
        f"{OBS_IMAGES}.image1_mask": torch.tensor(True).cuda(),
        f"{OBS_IMAGES}.image2_mask": torch.tensor(True).cuda(),
        "task": f"This is test sample {i}.",
        OBS_STATE: torch.rand((14, )),
        ACTION: torch.rand((50, 14)),
    } for i in range(B)]
    samples = [processor(sample) for sample in samples]
    inputs = {}
    for key in samples[0].keys():
        if key != "task":
            inputs[key] = torch.stack([sample[key] for sample in samples], dim=0).to(device=device)
    loss, loss_dict = model.forward(inputs)
    pp(loss)
    pp(loss_dict)
