#!/usr/bin/env python

import sys
import os
import time
import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import imageio
import numpy as np
import torch
import tyro
from omegaconf import OmegaConf
from huggingface_hub import snapshot_download

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import load_json
from lerobot.policies.InternVLA_A1_3B.modeling_internvla_a1 import QwenA1Config, QwenA1Policy
from lerobot.policies.InternVLA_A1_3B.transform_internvla_a1 import Qwen3_VLProcessorTransformFn
from lerobot.transforms.core import (
    NormalizeTransformFn,
    ResizeImagesWithPadFn,
    UnNormalizeTransformFn,
    RemapImageKeyTransformFn,
    compose,
)
from lerobot.utils.constants import OBS_IMAGES

ROOT_PATH = Path(__file__).resolve().parents[2]
# RoboTwin dependencies
sys.path.extend(
    [
        str(ROOT_PATH),
        str(ROOT_PATH / "third_party" / "RoboTwin"),
        str(ROOT_PATH / "third_party" / "RoboTwin" / "policy"),
        str(ROOT_PATH / "third_party" / "RoboTwin" / "description" / "utils"),
    ]
)

from envs import CONFIGS_PATH 
from envs.utils.create_actor import UnStableError
from generate_episode_instructions import generate_episode_descriptions
import image_tools
from update_result_markdown import update_markdown_with_tasks


class ActionMoEActivationCounter:
    """Count top-k expert selections for every LoRA-MoE in the action expert."""

    def __init__(self, policy, output_dir: Path, num_experts: int):
        self.output_dir = output_dir
        self.metadata_path = output_dir / "metadata.json"
        self.num_experts = int(num_experts)
        self.completed_tasks: set[int] = set()

        action_expert = policy.model.qwen3_vl_with_expert.act_expert
        modules = [
            (name, module)
            for name, module in action_expert.named_modules()
            if bool(getattr(module, "_moe_mode", False))
            and callable(getattr(module, "lora_router_forward", None))
        ]
        if not modules:
            raise RuntimeError("No LoRA-MoE routers found in the action expert")

        module_expert_counts = {int(getattr(module, "num_experts", 0)) for _, module in modules}
        if module_expert_counts != {self.num_experts}:
            raise RuntimeError(
                "Action MoE expert counts do not match the requested matrix width: "
                f"found={sorted(module_expert_counts)}, requested={self.num_experts}"
            )

        self.module_names = [name for name, _ in modules]
        device = next(policy.parameters()).device
        self.counts = torch.zeros(
            (len(modules), self.num_experts),
            dtype=torch.int64,
            device=device,
        )
        self._load_metadata()

        for row_idx, (_, module) in enumerate(modules):
            original_router_forward = module.lora_router_forward

            def counted_router_forward(
                x,
                *args,
                _original=original_router_forward,
                _row_idx=row_idx,
                **kwargs,
            ):
                result = _original(x, *args, **kwargs)
                top_k_indices = result[1]
                selected = torch.bincount(
                    top_k_indices.reshape(-1),
                    minlength=self.num_experts,
                )
                self.counts[_row_idx].add_(selected)
                return result

            module.lora_router_forward = counted_router_forward

        logging.info(
            "Installed action MoE activation counters: matrix_shape=%s, output=%s",
            tuple(self.counts.shape),
            self.output_dir,
        )
        print(
            f"[ActionMoEActivationCounter] matrix_shape={tuple(self.counts.shape)} "
            f"output_dir={self.output_dir}",
            flush=True,
        )

    def _load_metadata(self) -> None:
        if self.metadata_path.exists():
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            existing_names = metadata.get("module_names")
            if existing_names is not None and existing_names != self.module_names:
                raise ValueError("Existing action MoE module ordering does not match the loaded model")
            completed_tasks = {int(task_idx) for task_idx in metadata.get("completed_tasks", [])}
            self.completed_tasks = {
                task_idx
                for task_idx in completed_tasks
                if any(self.output_dir.glob(f"task_{task_idx:02d}_*.csv"))
            }

    def start_task(self) -> None:
        self.counts.zero_()

    def save(self, *, task_idx: int, task_name: str, checkpoint: str, test_num: int) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        counts = self.counts.detach().cpu().numpy()

        matrix_path = self.output_dir / f"task_{task_idx:02d}_{task_name}.csv"
        matrix_tmp = matrix_path.with_suffix(f"{matrix_path.suffix}.tmp")
        np.savetxt(matrix_tmp, counts, fmt="%d", delimiter=",")
        matrix_tmp.replace(matrix_path)

        metadata = {
            "checkpoint": checkpoint,
            "matrix_shape": list(counts.shape),
            "num_experts": self.num_experts,
            "test_num_per_task": int(test_num),
            "completed_tasks": sorted(self.completed_tasks),
            "module_names": self.module_names,
            "count_definition": "Number of token-level top-k selections for each action-expert LoRA-MoE expert",
        }
        metadata_tmp = self.metadata_path.with_suffix(f"{self.metadata_path.suffix}.tmp")
        metadata_tmp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        metadata_tmp.replace(self.metadata_path)
        return matrix_path


def format_cuda_memory_gb(memory_bytes: int) -> str:
    """Format CUDA memory usage in GB."""
    return f"{memory_bytes / (1024 ** 3):.2f} GB"


def resolve_ckpt_dir(ckpt_path: Union[str, Path]) -> Path:
    """
    Resolve a checkpoint path to a local directory.

    Supports:
    - Local directory path
    - HuggingFace repo id (e.g., "org/repo"), downloaded to HF cache via snapshot_download
    """
    ckpt_str = str(ckpt_path)
    local_dir = Path(ckpt_str).expanduser()
    if local_dir.exists():
        return local_dir.resolve()

    snapshot_dir = snapshot_download(repo_id=ckpt_str)
    return Path(snapshot_dir)


# Task list matching eval_robotwin.py
TASK_NAMES = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
]


def get_embodiment_config(robot_file: str):
    """Load robot embodiment configuration from YAML file."""
    robot_config_file = Path(robot_file) / "config.yml"
    with open(robot_config_file, "r", encoding="utf-8") as f:
        return OmegaConf.load(f)


def class_decorator(task_name: str):
    """Dynamically import and instantiate task environment class."""
    import importlib

    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def build_task_args(task_config: str, task_name: str):
    """Build task arguments from configuration files."""
    task_cfg_file = ROOT_PATH / "third_party" / "RoboTwin" / "task_config" / f"{task_config}.yml"
    with open(task_cfg_file, "r", encoding="utf-8") as f:
        task_args = OmegaConf.to_container(OmegaConf.load(f), resolve=True)

    with open(CONFIGS_PATH + "_embodiment_config.yml", "r", encoding="utf-8") as f:
        embodiment_types = OmegaConf.to_container(OmegaConf.load(f), resolve=True)
    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        camera_cfg = OmegaConf.to_container(OmegaConf.load(f), resolve=True)

    def get_embodiment_file(embodiment_type):
        robot_file = embodiment_types[embodiment_type]["file_path"]
        if robot_file is None:
            raise RuntimeError("No embodiment files found")
        return robot_file

    embodiment_type = task_args["embodiment"]
    head_camera_type = task_args["camera"]["head_camera_type"]
    task_args["head_camera_h"] = camera_cfg[head_camera_type]["h"]
    task_args["head_camera_w"] = camera_cfg[head_camera_type]["w"]

    if len(embodiment_type) == 1:
        robot_file = str(ROOT_PATH / "third_party" / "RoboTwin" / get_embodiment_file(embodiment_type[0]))
        task_args["left_robot_file"] = robot_file
        task_args["right_robot_file"] = robot_file
        task_args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        task_args["left_robot_file"] = str(
            ROOT_PATH / "third_party" / "RoboTwin" / get_embodiment_file(embodiment_type[0])
        )
        task_args["right_robot_file"] = str(
            ROOT_PATH / "third_party" / "RoboTwin" / get_embodiment_file(embodiment_type[1])
        )
        task_args["embodiment_dis"] = embodiment_type[2]
        task_args["dual_arm_embodied"] = False
    else:
        raise RuntimeError(f"Invalid embodiment type length: {len(embodiment_type)}, expected 1 or 3")

    task_args["left_embodiment_config"] = get_embodiment_config(task_args["left_robot_file"])
    task_args["right_embodiment_config"] = get_embodiment_config(task_args["right_robot_file"])
    task_args["task_name"] = task_name
    task_args["task_config"] = task_config
    task_args["eval_mode"] = True
    return task_args


def build_policy_and_transforms(
    ckpt_path: Union[str, Path],
    stats_key: str,
    resize_size: int,
    dtype: torch.dtype,
    compile_inference: bool = True,
    base_model_path: Union[str, Path, None] = None,
    loramoe_num_experts: int = 4,
    loramoe_router_top_k: int = 2,
    loramoe_ab_routing: bool = False,
    loramoe_router_top_k_a: int = 2,
    loramoe_router_top_k_b: int = 2,
    loramoe_enable_a_experts: bool = False,
    loramoe_enable_b_experts: bool = True,
    loramoe_share_a_across_experts: bool = True,
    use_visual_token_prune: bool = True,
    enable_affordance: bool = False,
    lambda_affordance: float | None = None,
    affordance_dim: int | None = None,
    enable_affordance_v2: bool = False,
    enable_affordance_v3: bool = False,
    affordance_v3_state_router_prior: bool = True,
    disable_affordance_router_input: bool = False,
    affordance_action_horizon: int | None = None,
    affordance_action_decay_end: int | None = None,
    affordance_gate_init: float | None = None,
    affordance_router_lead_in_fraction: float | None = None,
    use_ptq: bool = False,
    ptq_mode: str = "weight_only_int8",
    ptq_backend: str = "naive",
    ptq_experts: tuple[str, ...] = ("und", "gen", "act"),
    ptq_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
):
    """Load policy and build input/output transforms.

    Args:
        ckpt_path: Path to the checkpoint
        stats_key: Key for loading stats.json
        resize_size: Image resize size
        dtype: Data type for model
        compile_inference: Whether to enable torch.compile during inference
        base_model_path: Path to base pretrained model (required for LoRA/LoRA+MoE checkpoint inference)
        loramoe_num_experts: Number of experts for LoRA+MoE (default: 4)
        loramoe_router_top_k: Top-k selection for LoRA+MoE router (default: 2)
        loramoe_ab_routing: Use separate routers for lora_A and lora_B (default: False)
        loramoe_router_top_k_a: Top-k for lora_A router (default: 2)
        loramoe_router_top_k_b: Top-k for lora_B router (default: 2)
        loramoe_enable_a_experts: Whether to create expert-specific lora_A branches
        loramoe_enable_b_experts: Whether to create expert-specific lora_B branches
        loramoe_share_a_across_experts: Whether to share lora_A across experts
        use_ptq: Enable native expert PTQ during inference
        ptq_mode: PTQ mode, currently weight_only_int8 or weight_only_int4
        ptq_backend: PTQ backend, e.g. naive or bnb_int8
        ptq_experts: Expert names to quantize
        ptq_target_modules: Linear module names to quantize
    """
    from lerobot.policies.InternVLA_A1_3B.configuration_internvla_a1 import QwenA1Config

    ckpt_dir = resolve_ckpt_dir(ckpt_path)

    # 1. Use a safe loading path (avoids type errors)
    # The config loaded here contains all LoRA / LoRA+MoE settings (read from ckpt_dir)
    config = PreTrainedConfig.from_pretrained(ckpt_dir)

    if not isinstance(config, QwenA1Config):
        raise ValueError(f"Expected QwenA1Config, got {type(config)}")

    config.use_visual_token_prune = use_visual_token_prune
    checkpoint_affordance = bool(getattr(config, "enable_affordance", False))
    checkpoint_affordance_v2 = bool(getattr(config, "enable_affordance_v2", False))
    checkpoint_affordance_v3 = bool(getattr(config, "enable_affordance_v3", False))
    if (
        enable_affordance
        or enable_affordance_v2
        or enable_affordance_v3
        or checkpoint_affordance
        or checkpoint_affordance_v2
        or checkpoint_affordance_v3
    ):
        config.enable_affordance = True
        if lambda_affordance is not None:
            config.lambda_affordance = lambda_affordance
        if affordance_dim is not None:
            config.affordance_dim = affordance_dim
        if enable_affordance_v2 or checkpoint_affordance_v2:
            config.enable_affordance_v2 = True
            if affordance_action_horizon is not None:
                config.affordance_action_horizon = affordance_action_horizon
            if affordance_action_decay_end is not None:
                config.affordance_action_decay_end = affordance_action_decay_end
            if affordance_gate_init is not None:
                config.affordance_gate_init = affordance_gate_init
        if enable_affordance_v3 or checkpoint_affordance_v3:
            config.enable_affordance_v3 = True
            config.affordance_router_lead_in_fraction = 0.0
            config.affordance_v3_state_router_prior = affordance_v3_state_router_prior
        if affordance_router_lead_in_fraction is not None:
            config.affordance_router_lead_in_fraction = affordance_router_lead_in_fraction
        config.affordance_state_router_input = not disable_affordance_router_input
        logging.info(
            "Affordance enabled for inference: lambda=%s, dim=%s, v2=%s, v3=%s, horizon=%s, decay_end=%s, lead_in_fraction=%s, state_router_input=%s",
            getattr(config, "lambda_affordance", None),
            getattr(config, "affordance_dim", None),
            getattr(config, "enable_affordance_v2", False),
            getattr(config, "enable_affordance_v3", False),
            getattr(config, "affordance_action_horizon", None),
            getattr(config, "affordance_action_decay_end", None),
            getattr(config, "affordance_router_lead_in_fraction", None),
            getattr(config, "affordance_state_router_input", True),
        )
        if bool(getattr(config, "enable_affordance_v2", False)):
            horizon = int(config.affordance_action_horizon)
            decay_end = int(config.affordance_action_decay_end)
            gate_init = float(config.affordance_gate_init)
            if not 0 <= horizon <= decay_end <= int(config.chunk_size):
                raise ValueError(
                    "Affordance V2 requires 0 <= horizon <= decay_end <= chunk_size, got "
                    f"{horizon}, {decay_end}, {config.chunk_size}"
                )
            if not 0 < gate_init < 1:
                raise ValueError(f"Affordance V2 gate_init must be in (0, 1), got {gate_init}")
    else:
        config.enable_affordance = False
    if use_visual_token_prune and compile_inference:
        logging.info(
            "Keeping torch.compile enabled with visual token pruning; adaptive compaction may introduce graph breaks, "
            "but supported regions remain compiled."
        )

    if use_ptq:
        logging.info(
            "Enabling PTQ for inference: mode=%s, backend=%s, experts=%s, target_modules=%s",
            ptq_mode,
            ptq_backend,
            ptq_experts,
            ptq_target_modules,
        )
        config.use_ptq = True
        config.ptq_mode = ptq_mode
        config.ptq_backend = ptq_backend
        config.ptq_experts = tuple(ptq_experts)
        config.ptq_target_modules = tuple(ptq_target_modules)
        config.use_lora = False
        config.use_lora_moe = False

    # 2. Check whether this is a LoRA or LoRA+MoE checkpoint
    is_lora_checkpoint = getattr(config, 'use_lora', False)
    is_loramoe_checkpoint = getattr(config, 'use_lora_moe', False)
    is_loramoe_forced_last_checkpoint = getattr(config, 'use_lora_moe_forced_last', False)

    if is_lora_checkpoint or is_loramoe_checkpoint or is_loramoe_forced_last_checkpoint:
        if use_ptq:
            raise ValueError("PTQ inference currently supports native checkpoints only. Disable PTQ or use a non-LoRA checkpoint.")
        if base_model_path is None:
            raise ValueError("LoRA/LoRA+MoE/forced-last checkpoint detected. Please provide --base_model_path")

        base_dir = resolve_ckpt_dir(base_model_path)
        logging.info(f"Step 1: Loading base model weights from {base_dir}")

        # For LoRA+MoE / forced-last, ensure the MoE config matches training
        if is_loramoe_checkpoint or is_loramoe_forced_last_checkpoint:
            if is_loramoe_forced_last_checkpoint:
                logging.info(
                    "Detected LoRA+MoE Forced-Last checkpoint: %s base experts, top-%s",
                    loramoe_num_experts,
                    loramoe_router_top_k,
                )
                config.loramoe_num_experts = loramoe_num_experts
                config.loramoe_router_top_k = loramoe_router_top_k
            else:
                logging.info(f"Detected LoRA+MoE checkpoint: {loramoe_num_experts} experts, top-{loramoe_router_top_k}")
                logging.info(f"AB routing: {loramoe_ab_routing}, top_k_a={loramoe_router_top_k_a}, top_k_b={loramoe_router_top_k_b}")
                logging.info(
                    "Shared-A mode: share_a=%s, enable_a_experts=%s, enable_b_experts=%s",
                    loramoe_share_a_across_experts,
                    loramoe_enable_a_experts,
                    loramoe_enable_b_experts,
                )
                config.loramoe_ab_routing = loramoe_ab_routing
                config.loramoe_router_top_k_a = loramoe_router_top_k_a
                config.loramoe_router_top_k_b = loramoe_router_top_k_b
                config.loramoe_enable_a_experts = loramoe_enable_a_experts
                config.loramoe_enable_b_experts = loramoe_enable_b_experts
                config.loramoe_share_a_across_experts = loramoe_share_a_across_experts
            # Force-override MoE params in the config (to match training)
            config.loramoe_num_experts = loramoe_num_experts
            config.loramoe_router_top_k = loramoe_router_top_k

        # Reuse the config loaded above (with use_lora=True or use_lora_moe=True)
        policy = QwenA1Policy.from_pretrained(pretrained_name_or_path=base_dir, config=config)

        mode_str = "LoRA"
        weights_filename = "model.safetensors"
        if is_loramoe_forced_last_checkpoint:
            mode_str = "LoRA+MoE Forced-Last"
            weights_filename = "moe_weights.safetensors"
        elif is_loramoe_checkpoint:
            mode_str = "LoRA+MoE"
            if not (ckpt_dir / weights_filename).exists() and (ckpt_dir / "moe_weights.safetensors").exists():
                weights_filename = "moe_weights.safetensors"
        if is_loramoe_checkpoint and loramoe_ab_routing:
            mode_str += " (AB routing)"
        logging.info(f"Step 2: Loading {mode_str} & Full Fine-tuned weights from {ckpt_dir}")
        _load_checkpoint_weights(policy, ckpt_dir, weights_filename=weights_filename)
    else:
        logging.info(f"Loading full model checkpoint from {ckpt_dir}")
        policy = QwenA1Policy.from_pretrained(pretrained_name_or_path=ckpt_dir, config=config)

    # Manually enable compile (the checkpoint config may not contain this option)
    print(f"[INFO] compile_model before: {policy.config.compile_model}")
    skip_compile_for_ptq_backend = use_ptq and ptq_backend == "bnb_int8"
    if not compile_inference:
        print("[INFO] Skipping torch.compile because compile_inference=False.")
    elif skip_compile_for_ptq_backend:
        print("[INFO] Skipping torch.compile because ptq_backend=bnb_int8 is not stable with Dynamo/Inductor in this inference path.")
    elif not policy.config.compile_model:
        default_compile_mode = "reduce-overhead" if use_visual_token_prune else "max-autotune"
        compile_mode = os.environ.get("TORCH_COMPILE_MODE", default_compile_mode)
        print(f"[INFO] Enabling torch.compile(mode={compile_mode}, dynamic={use_visual_token_prune})...")
        policy.config.compile_model = True
        policy.config.compile_mode = compile_mode
        torch.set_float32_matmul_precision("high")
        compile_denoise_only = os.environ.get("TORCH_COMPILE_DENOISE_ONLY", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        compile_full_prune_path = os.environ.get(
            "VISUAL_TOKEN_PRUNE_COMPILE_FULL",
            os.environ.get("VISUAL_TOKEN_PRUNE_COMPILE_FULL", "false"),
        ).lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if compile_denoise_only:
            # Compile only the repeated action-denoising kernel. The eager outer
            # sampler avoids compiling prefix/middle setup and Python loop state.
            policy.model.denoise_step = torch.compile(
                policy.model.denoise_step,
                mode=compile_mode,
            )
            print("[INFO] Hot-path compile enabled: denoise_step compiled, outer sampler kept eager.")
        elif use_visual_token_prune and not compile_full_prune_path:
            # Adaptive pruning has dynamic shapes and Python-side plan state. Compile the
            # repeated denoising hot path while leaving prefix prune/restore eager.
            policy.model.denoise_step = torch.compile(
                policy.model.denoise_step,
                mode=compile_mode,
            )
            print("[INFO] Hybrid compile enabled: denoise_step compiled, visual prefix prune kept eager.")
        else:
            policy.model.sample_actions = torch.compile(
                policy.model.sample_actions,
                mode=compile_mode,
                dynamic=use_visual_token_prune,
            )
            policy.model.forward = torch.compile(
                policy.model.forward,
                mode=compile_mode,
                dynamic=use_visual_token_prune,
            )
        print("[INFO] torch.compile enabled!")

    policy.cuda().to(dtype).eval()

    router_attrs = (
        "lora_router",
        "lora_router_a",
        "lora_router_b",
        "state_proj_router",
        "state_proj_router_a",
        "state_proj_router_b",
    )
    router_tensors = []
    seen_router_ids = set()
    for module in policy.model.modules():
        for attr in router_attrs:
            router = getattr(module, attr, None)
            if router is None or id(router) in seen_router_ids:
                continue
            seen_router_ids.add(id(router))
            if isinstance(router, torch.nn.Module):
                router_tensors.extend(router.parameters())
            elif isinstance(router, torch.Tensor):
                router_tensors.append(router)

    for router_tensor in router_tensors:
        router_tensor.data = router_tensor.data.float()

    if getattr(policy.config, "use_lora_moe", False) and not router_tensors:
        raise RuntimeError("LoRA-MoE is enabled, but no router parameters were found")
    if any(router_tensor.dtype != torch.float32 for router_tensor in router_tensors):
        raise RuntimeError("LoRA-MoE router parameters must remain float32 during inference")
    if router_tensors:
        print(f"[INFO] Kept {len(router_tensors)} LoRA-MoE router tensors in float32.")

    # Original whole-function compile (commented out, kept for fallback)
    # policy.model.sample_actions = torch.compile(policy.model.sample_actions, mode="max-autotune")
    # policy.model.forward = torch.compile(policy.model.forward, mode="max-autotune")

    stats = load_json(ckpt_dir / "stats.json")[stats_key]
    stat_keys = ["min", "max", "mean", "std"]

    state_concat = {k: np.asarray(stats["observation.state"][k]) for k in stat_keys}
    state_stat = {"observation.state": state_concat}

    action_concat = {k: np.asarray(stats["action"][k]) for k in stat_keys}
    action_stat = {"action": action_concat}

    unnormalize_fn = UnNormalizeTransformFn(
        selected_keys=["action"],
        mode="mean_std",
        norm_stats=action_stat,
    )

    image_keys = [f"{OBS_IMAGES}.image{i}" for i in range(3)]

    input_transforms = compose(
        [
            ResizeImagesWithPadFn(height=resize_size, width=resize_size),
            RemapImageKeyTransformFn(mapping={k: k for k in image_keys}),
            Qwen3_VLProcessorTransformFn(),
            NormalizeTransformFn(selected_keys=["observation.state"], norm_stats=state_stat),
        ]
    )

    return policy, input_transforms, unnormalize_fn


def _load_checkpoint_weights(policy, ckpt_dir, weights_filename: str = "model.safetensors"):
    """
    Load all weights from the checkpoint at once.
    Since the policy is initialized from the base model and the LoRA branches
    already exist, load_state_dict matches them automatically:
    1. LoRA weights -> filled into the corresponding lora_A/B matrices
    2. Full fine-tuning weights (Action Head) -> overwrite the base layers
    """
    from safetensors.torch import load_file, load_model as load_model_as_safetensor

    model_path = ckpt_dir / weights_filename
    if not model_path.exists():
        raise FileNotFoundError(f"No weights found at {model_path}")

    # `policy.save_pretrained()` can save a full LoRA policy with shared-module
    # aliases (notably `lora_model` -> the understanding expert).  `load_file()`
    # exposes only the serialized alias keys and loses the safetensors metadata
    # needed to restore the canonical names.  Let safetensors load the model
    # directly first so both full-policy and adapter-only checkpoints work.
    try:
        load_device = str(next(policy.parameters()).device)
        missing_keys, unexpected_keys = load_model_as_safetensor(
            policy,
            str(model_path),
            device=load_device,
            strict=False,
        )
        critical_missing = [
            key
            for key in missing_keys
            if "lora_" in key
            or "action_in_proj" in key
            or "action_out_proj" in key
            or "action_time_mlp" in key
        ]
        if critical_missing:
            raise RuntimeError(
                "Checkpoint is missing trainable LoRA/action tensors after safetensors alias loading: "
                f"{critical_missing[:8]}"
            )
        logging.info(
            "Loaded checkpoint with safetensors alias-aware loader: missing=%d unexpected=%d",
            len(missing_keys),
            len(unexpected_keys),
        )
        return
    except Exception as exc:
        logging.warning(
            "Alias-aware checkpoint loading failed; falling back to legacy key loading: %s",
            exc,
        )

    state_dict = load_file(str(model_path))

    # ==================== Debug info start ====================
    logging.info("=" * 60)
    logging.info("[DEBUG] Checkpoint weight analysis:")
    logging.info(f"  - Total key count: {len(state_dict)}")

    # Count LoRA-related keys
    lora_a_keys = [k for k in state_dict.keys() if "lora_A" in k]
    lora_b_keys = [k for k in state_dict.keys() if "lora_B" in k]
    action_head_keys = [k for k in state_dict.keys() if "action_in_proj" in k or "action_out_proj" in k or "state_proj" in k]
    affordance_keys = [k for k in state_dict.keys() if "affordance_" in k]

    logging.info(f"  - LoRA A matrix count: {len(lora_a_keys)}")
    logging.info(f"  - LoRA B matrix count: {len(lora_b_keys)}")
    logging.info(f"  - Action Head related key count: {len(action_head_keys)}")
    logging.info(f"  - Affordance related key count: {len(affordance_keys)}")
    if bool(getattr(policy.config, "enable_affordance", False)) and not affordance_keys:
        raise ValueError(
            "Affordance inference is enabled, but the checkpoint contains no affordance_* weights. "
            "Use the Affordance-trained checkpoint or disable ENABLE_AFFORDANCE."
        )
    if bool(getattr(policy.config, "enable_affordance_v2", False)):
        expected_v2_keys = [
            key for key in policy.model.state_dict().keys() if "affordance_v2_" in key
        ]
        missing_v2_keys = [
            key
            for key in expected_v2_keys
            if key not in state_dict and f"model.{key}" not in state_dict
        ]
        if missing_v2_keys:
            raise ValueError(
                "Affordance V2 inference requested, but the checkpoint is missing V2 weights: "
                f"{missing_v2_keys[:8]}" + (" ..." if len(missing_v2_keys) > 8 else "")
            )

    # Check other keys (not LoRA, not Action Head)
    all_lora_keys = set(lora_a_keys + lora_b_keys)
    all_action_keys = set(action_head_keys)
    other_keys = [k for k in state_dict.keys() if k not in all_lora_keys and k not in all_action_keys]
    logging.info(f"  - Other keys (non-LoRA, non-ActionHead) count: {len(other_keys)}")
    if other_keys:
        logging.info(f"  - Example other keys: {other_keys[:5]}")

    if lora_a_keys:
        logging.info(f"  - Example LoRA A key: {lora_a_keys[0]}")
        logging.info(f"  - Example LoRA A shape: {state_dict[lora_a_keys[0]].shape}")
    if lora_b_keys:
        logging.info(f"  - Example LoRA B key: {lora_b_keys[0]}")
        logging.info(f"  - Example LoRA B shape: {state_dict[lora_b_keys[0]].shape}")
    if action_head_keys:
        logging.info(f"  - Example Action Head key: {action_head_keys[0]}")

    # Check LoRA-related keys in policy.model
    model_state_keys = list(policy.model.state_dict().keys())
    model_lora_keys = [k for k in model_state_keys if "lora_A" in k or "lora_B" in k]

    logging.info(f"[DEBUG] Model LoRA related key count: {len(model_lora_keys)}")

    # Print all LoRA keys
    logging.info("=" * 60)
    logging.info("[DEBUG] All LoRA keys ({} total):".format(len(model_lora_keys)))
    logging.info("=" * 60)
    for i, k in enumerate(model_lora_keys):
        logging.info(f"  [{i}] {k}")

    # ==================== Debug info end ====================

    # ==================== Print keys before loading ====================
    logging.info("=" * 60)
    logging.info("[DEBUG] Checkpoint keys (before loading) - all:")
    logging.info("=" * 60)
    for i, key in enumerate(list(state_dict.keys())):
        logging.info(f"  [{i}] {key}")

    # ==================== Fix key mismatch ====================
    # The checkpoint stores policy.state_dict(), so keys carry the "model." prefix
    # Loading uses policy.model.load_state_dict(), so strip the "model." prefix

    transformed_state_dict = {}
    for key, value in state_dict.items():
        # Strip the "model." prefix
        if key.startswith("model."):
            new_key = key[6:]  # Drop the first 6 characters ("model.")
        else:
            new_key = key

        transformed_state_dict[new_key] = value

    logging.info(f"[DEBUG] Key conversion done, converted key count: {len(transformed_state_dict)}")
    # ==================== Conversion end ====================

    # ==================== Print converted keys ====================
    logging.info("=" * 60)
    logging.info("[DEBUG] Converted keys (ready to load) - all:")
    logging.info("=" * 60)
    for i, key in enumerate(list(transformed_state_dict.keys())):
        logging.info(f"  [{i}] {key}")

    # Use strict=False because checkpoints usually omit frozen layers such as the Vision Encoder
    # Only the LoRA branches and Action Head need to be updated
    msg = policy.model.load_state_dict(transformed_state_dict, strict=False)

    if bool(getattr(policy.config, "enable_affordance", False)):
        loaded_state = policy.model.state_dict()
        missing_affordance = []
        mismatched_affordance = []
        for key in affordance_keys:
            model_key = key[6:] if key.startswith("model.") else key
            if model_key not in loaded_state:
                missing_affordance.append(model_key)
                continue
            if not torch.allclose(loaded_state[model_key].cpu(), state_dict[key].cpu()):
                mismatched_affordance.append(model_key)
        if missing_affordance or mismatched_affordance:
            raise RuntimeError(
                "Affordance checkpoint loading verification failed: "
                f"missing={missing_affordance[:5]}, mismatched={mismatched_affordance[:5]}"
            )
        logging.info("Verified %d Affordance tensors loaded from checkpoint", len(affordance_keys))

    # ==================== Print keys after loading ====================
    model_state_after = policy.model.state_dict()
    logging.info("=" * 60)
    logging.info("[DEBUG] Model keys (after loading) - all:")
    logging.info("=" * 60)
    for i, key in enumerate(list(model_state_after.keys())):
        logging.info(f"  [{i}] {key}")

    logging.info("=" * 60)
    logging.info("[DEBUG] Model LoRA keys - all:")
    lora_keys_after = [k for k in model_state_after.keys() if "lora" in k.lower()]
    for i, key in enumerate(lora_keys_after):
        logging.info(f"  [{i}] {key}")

    # ==================== Load result debug ====================
    logging.info("=" * 60)
    logging.info("[DEBUG] Load result:")
    logging.info(f"  - Missing keys : {len(msg.missing_keys)}")
    logging.info(f"  - Unexpected keys : {len(msg.unexpected_keys)}")

    if msg.missing_keys:
        logging.info(f"  - Missing keys (all): {msg.missing_keys}")
    if msg.unexpected_keys:
        logging.info(f"  - Unexpected keys: {msg.unexpected_keys}")

    # ==================== Verify Action Head loaded correctly ====================
    # Note: model.state_dict() returns CPU tensors; move both sides to CPU for comparison
    model_state = policy.model.state_dict()

    logging.info("=" * 60)
    logging.info("[DEBUG] Action Head weight verification (vs checkpoint):")

    # Check action_in_proj
    if "action_in_proj.weight" in transformed_state_dict:
        action_in = model_state.get("action_in_proj.weight", None)
        ckpt_action_in = transformed_state_dict["action_in_proj.weight"]
        if action_in is not None:
            # Force to CPU for comparison to avoid device mismatch
            action_in_cpu = action_in.cpu()
            ckpt_action_in_cpu = ckpt_action_in.cpu()
            model_norm = torch.norm(action_in_cpu).item()
            ckpt_norm = torch.norm(ckpt_action_in_cpu).item()
            match = "✓" if torch.allclose(action_in_cpu, ckpt_action_in_cpu) else "✗"
            logging.info(f"  action_in_proj.weight: model_norm={model_norm:.4f}, ckpt_norm={ckpt_norm:.4f} {match}")
            logging.info(f"    mean: {action_in_cpu.mean().item():.6f}, std: {action_in_cpu.std().item():.6f}")
        else:
            logging.warning("  - action_in_proj.weight not found")

    # Check action_out_proj
    if "action_out_proj.weight" in transformed_state_dict:
        action_out = model_state.get("action_out_proj.weight", None)
        ckpt_action_out = transformed_state_dict["action_out_proj.weight"]
        if action_out is not None:
            action_out_cpu = action_out.cpu()
            ckpt_action_out_cpu = ckpt_action_out.cpu()
            model_norm = torch.norm(action_out_cpu).item()
            ckpt_norm = torch.norm(ckpt_action_out_cpu).item()
            match = "✓" if torch.allclose(action_out_cpu, ckpt_action_out_cpu) else "✗"
            logging.info(f"  action_out_proj.weight: model_norm={model_norm:.4f}, ckpt_norm={ckpt_norm:.4f} {match}")

    # Check action_time_mlp_in
    if "action_time_mlp_in.weight" in transformed_state_dict:
        mlp_in = model_state.get("action_time_mlp_in.weight", None)
        ckpt_mlp_in = transformed_state_dict["action_time_mlp_in.weight"]
        if mlp_in is not None:
            mlp_in_cpu = mlp_in.cpu()
            ckpt_mlp_in_cpu = ckpt_mlp_in.cpu()
            model_norm = torch.norm(mlp_in_cpu).item()
            ckpt_norm = torch.norm(ckpt_mlp_in_cpu).item()
            match = "✓" if torch.allclose(mlp_in_cpu, ckpt_mlp_in_cpu) else "✗"
            logging.info(f"  action_time_mlp_in.weight: model_norm={model_norm:.4f}, ckpt_norm={ckpt_norm:.4f} {match}")

    # Check action_time_mlp_out
    if "action_time_mlp_out.weight" in transformed_state_dict:
        mlp_out = model_state.get("action_time_mlp_out.weight", None)
        ckpt_mlp_out = transformed_state_dict["action_time_mlp_out.weight"]
        if mlp_out is not None:
            mlp_out_cpu = mlp_out.cpu()
            ckpt_mlp_out_cpu = ckpt_mlp_out.cpu()
            model_norm = torch.norm(mlp_out_cpu).item()
            ckpt_norm = torch.norm(ckpt_mlp_out_cpu).item()
            match = "✓" if torch.allclose(mlp_out_cpu, ckpt_mlp_out_cpu) else "✗"
            logging.info(f"  action_time_mlp_out.weight: model_norm={model_norm:.4f}, ckpt_norm={ckpt_norm:.4f} {match}")

    # ==================== Verify LoRA weights loaded successfully ====================
    # Directly compare values in the checkpoint and the model
    # Use CPU for comparison to avoid device mismatch
    logging.info("=" * 60)
    logging.info("[DEBUG] LoRA weight comparison:")

    # Verify act_expert LoRA - multi-layer, multi-position check
    act_lora_keys_to_check = [
        "model.qwen3_vl_with_expert.act_expert.base_model.model.layers.0.mlp.down_proj.lora_A.default.weight",
        "model.qwen3_vl_with_expert.act_expert.base_model.model.layers.0.mlp.down_proj.lora_B.default.weight",
        "model.qwen3_vl_with_expert.act_expert.base_model.model.layers.10.mlp.down_proj.lora_A.default.weight",
        "model.qwen3_vl_with_expert.act_expert.base_model.model.layers.10.mlp.down_proj.lora_B.default.weight",
        "model.qwen3_vl_with_expert.act_expert.base_model.model.layers.20.mlp.down_proj.lora_A.default.weight",
        "model.qwen3_vl_with_expert.act_expert.base_model.model.layers.20.mlp.down_proj.lora_B.default.weight",
    ]

    for ckpt_key in act_lora_keys_to_check:
        if ckpt_key in state_dict:
            ckpt_lora = state_dict[ckpt_key].cpu()
            ckpt_norm = torch.norm(ckpt_lora).item()
            layer_idx = ckpt_key.split(".layers.")[1].split(".")[0]
            lora_type = "lora_A" if "lora_A" in ckpt_key else "lora_B"

            # Find the corresponding key in the model
            found = False
            for model_key in model_state.keys():
                if "act_expert" in model_key and f"layers.{layer_idx}" in model_key and "mlp.down_proj" in model_key and lora_type in model_key:
                    model_lora = model_state[model_key].cpu()
                    model_norm = torch.norm(model_lora).item()
                    match = "✓" if torch.allclose(ckpt_lora, model_lora) else "✗"
                    logging.info(f"  [act_expert] layer={layer_idx}, {lora_type}: ckpt_norm={ckpt_norm:.4f}, model_norm={model_norm:.4f} {match}")
                    found = True
                    break
            if not found:
                logging.warning(f"  [act_expert] layer={layer_idx}, {lora_type}: not found in model")

    # Verify und_expert LoRA - multi-layer, multi-position check
    und_lora_keys_to_check = [
        "model.qwen3_vl_with_expert.und_expert.base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.default.weight",
        "model.qwen3_vl_with_expert.und_expert.base_model.model.model.language_model.layers.0.mlp.down_proj.lora_B.default.weight",
        "model.qwen3_vl_with_expert.und_expert.base_model.model.model.language_model.layers.15.mlp.down_proj.lora_A.default.weight",
        "model.qwen3_vl_with_expert.und_expert.base_model.model.model.language_model.layers.15.mlp.down_proj.lora_B.default.weight",
        "model.qwen3_vl_with_expert.und_expert.base_model.model.model.language_model.layers.30.mlp.down_proj.lora_A.default.weight",
        "model.qwen3_vl_with_expert.und_expert.base_model.model.model.language_model.layers.30.mlp.down_proj.lora_B.default.weight",
    ]

    for ckpt_key in und_lora_keys_to_check:
        if ckpt_key in state_dict:
            ckpt_lora = state_dict[ckpt_key].cpu()
            ckpt_norm = torch.norm(ckpt_lora).item()
            layer_idx = ckpt_key.split(".layers.")[1].split(".")[0]
            lora_type = "lora_A" if "lora_A" in ckpt_key else "lora_B"

            # Find the corresponding key in the model
            found = False
            for model_key in model_state.keys():
                if "und_expert" in model_key and f"layers.{layer_idx}" in model_key and "mlp.down_proj" in model_key and lora_type in model_key:
                    model_lora = model_state[model_key].cpu()
                    model_norm = torch.norm(model_lora).item()
                    match = "✓" if torch.allclose(ckpt_lora, model_lora) else "✗"
                    logging.info(f"  [und_expert] layer={layer_idx}, {lora_type}: ckpt_norm={ckpt_norm:.4f}, model_norm={model_norm:.4f} {match}")
                    found = True
                    break
            if not found:
                logging.warning(f"  [und_expert] layer={layer_idx}, {lora_type}: not found in model")

    # Verify state_proj LoRA
    state_lora_keys = [
        ("state_proj_lora_A.weight", "state_proj", "lora_A"),
        ("state_proj_lora_B.weight", "state_proj", "lora_B"),
    ]
    for ckpt_key, model_type, lora_type in state_lora_keys:
        if ckpt_key in transformed_state_dict:
            ckpt_lora = transformed_state_dict[ckpt_key].cpu()
            ckpt_norm = torch.norm(ckpt_lora).item()
            model_lora = model_state.get(ckpt_key, None)
            if model_lora is not None:
                model_lora = model_lora.cpu()
                model_norm = torch.norm(model_lora).item()
                match = "✓" if torch.allclose(ckpt_lora, model_lora) else "✗"
                logging.info(f"  [{model_type}] {lora_type}: ckpt_norm={ckpt_norm:.4f}, model_norm={model_norm:.4f} {match}")
            else:
                logging.warning(f"  [{model_type}] {lora_type}: not found in model")

    # ==================== Verify pretrained (base model) weights are preserved ====================
    logging.info("=" * 60)
    logging.info("[DEBUG] Pretrained (Base Model) weight verification:")

    # First check whether the checkpoint contains base model weights (without LoRA)
    down_proj_keys = [k for k in state_dict.keys() if "down_proj.weight" in k and "lora" not in k.lower()]
    logging.info(f"[DEBUG] Checkpoint down_proj.weight (non-LoRA) count: {len(down_proj_keys)}")
    if down_proj_keys:
        logging.info(f"[DEBUG] Example down_proj key: {down_proj_keys[0]}")

    # Check base model weights in the model
    model_down_proj_keys = [k for k in model_state.keys() if "down_proj.weight" in k and "lora" not in k.lower()]
    logging.info(f"[DEBUG] Model down_proj.weight (non-LoRA) count: {len(model_down_proj_keys)}")
    if model_down_proj_keys:
        logging.info(f"[DEBUG] Example model down_proj key: {model_down_proj_keys[0]}")

    # Try to match base model weights between checkpoint and model
    if down_proj_keys and model_down_proj_keys:
        # Use the first down_proj key as a test
        ckpt_key = down_proj_keys[0]
        logging.info(f"[DEBUG] Test match: ckpt_key={ckpt_key}")
        ckpt_weight = state_dict[ckpt_key].cpu()
        ckpt_norm = torch.norm(ckpt_weight).item()

        # Strip the "model." prefix, then look it up in the model
        model_key_candidate = ckpt_key[6:] if ckpt_key.startswith("model.") else ckpt_key
        logging.info(f"[DEBUG] Trying model_key: {model_key_candidate}")

        if model_key_candidate in model_state:
            model_weight = model_state[model_key_candidate].cpu()
            model_norm = torch.norm(model_weight).item()
            match = "✓" if torch.allclose(ckpt_weight, model_weight) else "✗"
            logging.info(f"[DEBUG] base model weight comparison: ckpt_norm={ckpt_norm:.4f}, model_norm={model_norm:.4f} {match}")
        else:
            logging.warning(f"[DEBUG] not found in model: {model_key_candidate}")

    # Check act_expert base model weights
    base_model_keys_to_check = [
        # act_expert base model
        "model.qwen3_vl_with_expert.act_expert.base_model.model.layers.0.mlp.down_proj.weight",
        "model.qwen3_vl_with_expert.act_expert.base_model.model.layers.10.mlp.down_proj.weight",
        "model.qwen3_vl_with_expert.act_expert.base_model.model.layers.0.self_attn.q_proj.weight",
        "model.qwen3_vl_with_expert.act_expert.base_model.model.layers.10.self_attn.q_proj.weight",
    ]

    for ckpt_key in base_model_keys_to_check:
        if ckpt_key in state_dict:
            ckpt_weight = state_dict[ckpt_key].cpu()
            ckpt_norm = torch.norm(ckpt_weight).item()

            # Derive the model key from the checkpoint key
            # checkpoint: model.qwen3_vl_with_expert.act_expert.base_model.model.layers.x.xxx
            # model: qwen3_vl_with_expert.act_expert.base_model.model.layers.x.xxx
            model_key = ckpt_key[6:]  # Strip the "model." prefix

            if model_key in model_state:
                model_weight = model_state[model_key].cpu()
                model_norm = torch.norm(model_weight).item()
                match = "✓" if torch.allclose(ckpt_weight, model_weight) else "✗"
                layer_idx = ckpt_key.split(".layers.")[1].split(".")[0]
                component = ckpt_key.split(".layers.")[1].split(".")[1:3]  # e.g., "mlp.down_proj"
                component_str = ".".join(component)
                logging.info(f"  [act_expert] layer={layer_idx}, {component_str}: ckpt_norm={ckpt_norm:.4f}, model_norm={model_norm:.4f} {match}")
            else:
                logging.warning(f"  [act_expert] not found in model: {model_key}")

    # Check und_expert base model weights
    und_base_model_keys = [
        "model.qwen3_vl_with_expert.und_expert.base_model.model.model.language_model.layers.0.mlp.down_proj.weight",
        "model.qwen3_vl_with_expert.und_expert.base_model.model.model.language_model.layers.15.mlp.down_proj.weight",
        "model.qwen3_vl_with_expert.und_expert.base_model.model.model.language_model.layers.0.self_attn.q_proj.weight",
        "model.qwen3_vl_with_expert.und_expert.base_model.model.model.language_model.layers.15.self_attn.q_proj.weight",
    ]

    for ckpt_key in und_base_model_keys:
        if ckpt_key in state_dict:
            ckpt_weight = state_dict[ckpt_key].cpu()
            ckpt_norm = torch.norm(ckpt_weight).item()

            # checkpoint: model.qwen3_vl_with_expert.und_expert.base_model.model.model.language_model.layers.x.xxx
            # model: qwen3_vl_with_expert.und_expert.base_model.model.model.language_model.layers.x.xxx
            model_key = ckpt_key[6:]

            if model_key in model_state:
                model_weight = model_state[model_key].cpu()
                model_norm = torch.norm(model_weight).item()
                match = "✓" if torch.allclose(ckpt_weight, model_weight) else "✗"
                layer_idx = ckpt_key.split(".layers.")[1].split(".")[0]
                component = ckpt_key.split(".layers.")[1].split(".")[1:3]
                component_str = ".".join(component)
                logging.info(f"  [und_expert] layer={layer_idx}, {component_str}: ckpt_norm={ckpt_norm:.4f}, model_norm={model_norm:.4f} {match}")
            else:
                logging.warning(f"  [und_expert] not found in model: {model_key}")

    logging.info("=" * 60)
    # ==================== Debug info end ====================


@dataclass
class InferenceArgs:
    """Configuration arguments for inference."""

    task_idx: int = 0  # Single task index
    task_indices: str = None  # Multiple task indices, e.g. "0,1,2,3" or "0-10"
    task_config: str = "demo_clean"
    instruction_type: str = "unseen"
    seed: int = 0
    ckpt_path: Union[str, Path] = "InternRobotics/InternVLA-A1-3B-RoboTwin"
    base_model_path: Union[str, Path, None] = None  # Required for LoRA checkpoint inference
    stats_key: str = "aloha"
    resize_size: int = 224
    image_history_interval: int = 15
    action_mode: str = "delta"  # delta | abs
    dtype: str = "float32"  # float32 | bfloat16
    video_dir: Path = Path("videos")
    fps: int = 30
    save_videos: bool = True
    decode_image_flag: bool = False
    debug: bool = False
    log_level: str = "WARNING"  # DEBUG | INFO | WARNING | ERROR
    infer_horizon: int = 30
    action_horizon_size: int = 50
    test_num: int = 100
    resume_test_num: int = 0
    resume_success: int = 0
    resume_next_seed: int | None = None
    compile_inference: bool = True
    robot_type: tuple[int, ...] = (6, 1, 6, 1)
    # LoRA+MoE extra parameters (must match training config)
    loramoe_num_experts: int = 4
    loramoe_router_top_k: int = 2
    # AB routing: separate routers for lora_A and lora_B
    loramoe_ab_routing: bool = False
    loramoe_router_top_k_a: int = 2  # top-k for lora_A router
    loramoe_router_top_k_b: int = 2  # top-k for lora_B router
    loramoe_enable_a_experts: bool = False
    loramoe_enable_b_experts: bool = True
    loramoe_share_a_across_experts: bool = True
    use_visual_token_prune: bool = True
    enable_affordance: bool = False
    lambda_affordance: float | None = None
    affordance_dim: int | None = None
    enable_affordance_v2: bool = False
    enable_affordance_v3: bool = False
    affordance_v3_state_router_prior: bool = True
    disable_affordance_router_input: bool = False
    affordance_action_horizon: int | None = None
    affordance_action_decay_end: int | None = None
    affordance_gate_init: float | None = None
    affordance_router_lead_in_fraction: float | None = None
    # PTQ parameters (native expert LLM only)
    use_ptq: bool = False
    ptq_mode: str = "weight_only_int8"
    ptq_backend: str = "naive"
    ptq_experts: tuple[str, ...] = ("und", "gen", "act")
    ptq_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    result_markdown_path: Union[str, Path, None] = None
    action_moe_activation_path: Union[str, Path, None] = None



def infer_once(
    args: InferenceArgs,
    policy=None,
    input_transforms=None,
    unnormalize_fn=None,
    inference_time_history: list[float] | None = None,
    inference_time_window: list[float] | None = None,
):
    """Run inference on a single task.

    Args:
        args: Inference arguments
        policy: Pre-loaded policy (if None, will load new one)
        input_transforms: Pre-loaded input transforms (if None, will create new ones)
        unnormalize_fn: Pre-loaded unnormalize function (if None, will create new one)
    """
    if inference_time_history is None:
        inference_time_history = []
    if inference_time_window is None:
        inference_time_window = []

    task_name = TASK_NAMES[args.task_idx]
    task_args = build_task_args(args.task_config, task_name)
    TASK_ENV = class_decorator(task_args["task_name"])

    # dtype must always be defined
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    # Create a new policy if none is provided
    if policy is None:
        policy, input_transforms, unnormalize_fn = build_policy_and_transforms(
            args.ckpt_path, args.stats_key, args.resize_size, dtype,
            compile_inference=args.compile_inference,
            base_model_path=args.base_model_path,
            loramoe_num_experts=args.loramoe_num_experts,
            loramoe_router_top_k=args.loramoe_router_top_k,
            loramoe_ab_routing=args.loramoe_ab_routing,
            loramoe_router_top_k_a=args.loramoe_router_top_k_a,
            loramoe_router_top_k_b=args.loramoe_router_top_k_b,
            loramoe_enable_a_experts=args.loramoe_enable_a_experts,
            loramoe_enable_b_experts=args.loramoe_enable_b_experts,
            loramoe_share_a_across_experts=args.loramoe_share_a_across_experts,
            use_visual_token_prune=args.use_visual_token_prune,
            enable_affordance=args.enable_affordance,
            lambda_affordance=args.lambda_affordance,
            affordance_dim=args.affordance_dim,
            enable_affordance_v2=args.enable_affordance_v2,
            enable_affordance_v3=args.enable_affordance_v3,
            affordance_v3_state_router_prior=args.affordance_v3_state_router_prior,
            disable_affordance_router_input=args.disable_affordance_router_input,
            affordance_action_horizon=args.affordance_action_horizon,
            affordance_action_decay_end=args.affordance_action_decay_end,
            affordance_gate_init=args.affordance_gate_init,
            affordance_router_lead_in_fraction=args.affordance_router_lead_in_fraction,
            use_ptq=args.use_ptq,
            ptq_mode=args.ptq_mode,
            ptq_backend=args.ptq_backend,
            ptq_experts=args.ptq_experts,
            ptq_target_modules=args.ptq_target_modules,
        )

    logging.info("=" * 80)
    logging.info("Initializing environment...")
    logging.info(f"Task: {task_name}, seed: {args.seed}")

    if args.resume_test_num < 0 or args.resume_test_num > args.test_num:
        raise ValueError("resume_test_num must be between 0 and test_num")
    if args.resume_success < 0 or args.resume_success > args.resume_test_num:
        raise ValueError("resume_success must be between 0 and resume_test_num")

    TASK_ENV.suc = args.resume_success
    TASK_ENV.test_num = args.resume_test_num
    expert_check = True

    now_id = args.resume_test_num
    succ_seed = args.resume_test_num
    seed = args.seed
    st_seed = 100000 * (1 + seed)
    now_seed = args.resume_next_seed if args.resume_next_seed is not None else st_seed + args.resume_test_num
    test_num = args.test_num
    clear_cache_freq = task_args["clear_cache_freq"]
    task_args["eval_mode"] = True
    succ_seeds = list(range(st_seed, st_seed * 2))

    while succ_seed < test_num:
        render_freq = task_args["render_freq"]
        task_args["render_freq"] = 0

        if expert_check:
            try:
                TASK_ENV.setup_demo(
                    now_ep_num=now_id, seed=succ_seeds[now_seed - st_seed], is_test=True, **task_args
                )
                episode_info = TASK_ENV.play_once()
                TASK_ENV.close_env()
            except (UnStableError, Exception):
                TASK_ENV.close_env()
                now_seed += 1
                task_args["render_freq"] = render_freq
                continue

        if (not expert_check) or (TASK_ENV.plan_success and TASK_ENV.check_success()):
            succ_seed += 1
        else:
            now_seed += 1
            task_args["render_freq"] = render_freq
            continue

        task_args["render_freq"] = render_freq

        TASK_ENV.setup_demo(
            now_ep_num=now_id, seed=succ_seeds[now_seed - st_seed], is_test=True, **task_args
        )
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(task_name, episode_info_list, test_num)
        instruction = np.random.choice(results[0][args.instruction_type])
        TASK_ENV.set_instruction(instruction=instruction)

        succ = False
        policy.reset()
        action_plan = deque([], maxlen=args.action_horizon_size)
        step_started_logged = False
        replay_images = []
        head_color_list = []
        left_wrist_color_list = []
        right_wrist_color_list = []
        image_history_interval = args.image_history_interval
        action_dim = sum(args.robot_type)
        left_gripper_idx = sum(args.robot_type[0:2])-1
        right_gripper_idx = sum(args.robot_type[0:4])-1

        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            # Get observation at every step for video recording
            observation = TASK_ENV.get_obs()
            img = observation["observation"]["head_camera"]["rgb"]

            # Record frame at every step (not just when inferring new actions)
            replay_images.append(img.copy())

            if len(action_plan) <= image_history_interval:
                left_wrist_img = observation["observation"]["left_camera"]["rgb"]
                right_wrist_img = observation["observation"]["right_camera"]["rgb"]

                head_color_list.append(torch.as_tensor(img).contiguous().cuda().to(dtype) / 255.0)
                left_wrist_color_list.append(torch.as_tensor(left_wrist_img).contiguous().cuda().to(dtype) / 255.0)
                right_wrist_color_list.append(torch.as_tensor(right_wrist_img).contiguous().cuda().to(dtype) / 255.0)

                while len(head_color_list) > image_history_interval + 1:
                    head_color_list.pop(0)
                    left_wrist_color_list.pop(0)
                    right_wrist_color_list.pop(0)

                past_idx = max(len(head_color_list) - image_history_interval - 1, 0)
                image_head_with_history = torch.stack([head_color_list[past_idx], head_color_list[-1]], dim=0)
                image_hand_left_with_history = torch.stack(
                    [left_wrist_color_list[past_idx], left_wrist_color_list[-1]], dim=0
                )
                image_hand_right_with_history = torch.stack(
                    [                    right_wrist_color_list[past_idx], right_wrist_color_list[-1]], dim=0
                )

            if not action_plan:
                init_action = torch.as_tensor(observation["joint_action"]["vector"][None]).contiguous().cuda()
                state = torch.from_numpy(observation["joint_action"]["vector"]).float().cuda()
                task = TASK_ENV.get_instruction()

                sample = {
                    f"{OBS_IMAGES}.image0": image_head_with_history,
                    f"{OBS_IMAGES}.image1": image_hand_left_with_history,
                    f"{OBS_IMAGES}.image2": image_hand_right_with_history,
                    "observation.state": state,
                    "task": task,
                }
                for key in sample.keys():
                    if OBS_IMAGES in key and "mask" not in key:
                        image = sample[key].permute(0, 3, 1, 2)
                        sample[key] = image

                sample = input_transforms(sample)

                inputs = {}
                for key in sample.keys():
                    if key == "task":
                        inputs[key] = [sample[key]]
                    elif sample[key].dtype == torch.int64:
                        inputs[key] = sample[key][None].cuda()
                    else:
                        inputs[key] = sample[key][None].cuda().to(dtype=dtype)

                inputs.update({
                    f"{OBS_IMAGES}.image0_mask": torch.tensor([True]).cuda(),
                    f"{OBS_IMAGES}.image1_mask": torch.tensor([True]).cuda(),
                    f"{OBS_IMAGES}.image2_mask": torch.tensor([True]).cuda(),
                })

                with torch.no_grad():
                    # Timing start
                    torch.cuda.synchronize()
                    t0 = time.time()

                    if not step_started_logged:
                        print(
                            f"[STEP] Entered action inference for task {args.task_idx}, episode {now_id}",
                            flush=True,
                        )
                        step_started_logged = True
                    action_pred, _ = policy.predict_action_chunk(inputs, decode_image=args.decode_image_flag)

                    # Timing end
                    torch.cuda.synchronize()
                    t1 = time.time()

                    # Print inference time (average of the last 10 calls, every 10 calls)
                    inference_time_history.append(t1 - t0)
                    inference_time_window.append(t1 - t0)
                    if len(inference_time_window) >= 10:
                        avg_time = sum(inference_time_window) / len(inference_time_window)
                        print(f"[DEBUG] Inference time (last 10 avg): {avg_time*1000:.2f} ms")
                        if torch.cuda.is_available():
                            current_allocated = torch.cuda.memory_allocated()
                            current_reserved = torch.cuda.memory_reserved()
                            peak_allocated = torch.cuda.max_memory_allocated()
                            peak_reserved = torch.cuda.max_memory_reserved()
                            print(
                                "[DEBUG] GPU memory | "
                                f"current allocated: {format_cuda_memory_gb(current_allocated)} | "
                                f"current reserved: {format_cuda_memory_gb(current_reserved)} | "
                                f"peak allocated: {format_cuda_memory_gb(peak_allocated)} | "
                                f"peak reserved: {format_cuda_memory_gb(peak_reserved)}"
                            )
                        inference_time_window = []  # Reset window

                action_pred = action_pred[0, : args.infer_horizon, :action_dim]
                action_pred = unnormalize_fn({"action": action_pred})["action"]

                if args.action_mode == "delta":
                    init_action[:, left_gripper_idx] = 0.0
                    init_action[:, right_gripper_idx] = 0.0
                    action_pred += init_action
                action_plan.extend(action_pred.cpu().numpy())

            action = action_plan.popleft()
            action[left_gripper_idx] = 0 if action[left_gripper_idx] < 0.5 else 1
            action[right_gripper_idx] = 0 if action[right_gripper_idx] < 0.5 else 1
            TASK_ENV.take_action(action, action_type="qpos")

            if TASK_ENV.eval_success:
                succ = True
                break

        if succ:
            TASK_ENV.suc += 1
            print("\033[92mSuccess!\033[0m")
        else:
            print("\033[91mFail!\033[0m")
            if TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
                print(
                    "Failure context: "
                    f"step limit reached ({TASK_ENV.take_action_cnt}/{TASK_ENV.step_lim})"
                )
            failure_reasons = TASK_ENV.get_failure_reasons()
            if failure_reasons:
                print("Failure reasons:")
                for reason in failure_reasons:
                    print(f"  - {reason}")

        if args.save_videos:
            args.video_dir.mkdir(parents=True, exist_ok=True)
            suffix = "success" if succ else "failure"
            imageio.mimwrite(
                args.video_dir / f"{suffix}_{succ_seed}.mp4",
                replay_images,  # Already in HWC format (uint8 numpy arrays)
                fps=args.fps,
            )

        now_id += 1
        TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))

        if TASK_ENV.render_freq:
            TASK_ENV.viewer.close()

        TASK_ENV.test_num += 1

        print(
            f"\033[93m{task_name}\033[0m |  \033[92m{task_args['task_config']}\033[0m \033[0m\n"
            f"Success rate: \033[96m{TASK_ENV.suc}/{TASK_ENV.test_num}\033[0m => "
            f"\033[95m{round(TASK_ENV.suc/TASK_ENV.test_num*100, 1)}%\033[0m, "
            f"current seed: \033[90m{now_seed}\033[0m\n"
        )
        now_seed += 1

    # Return result
    return {
        'task_idx': args.task_idx,
        'task_name': task_name,
        'success_rate': TASK_ENV.suc / TASK_ENV.test_num if TASK_ENV.test_num > 0 else 0,
        'num_success': TASK_ENV.suc,
        'num_test': TASK_ENV.test_num,
    }


def main(args: InferenceArgs):
    """Main entry point for inference."""
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    log_level = log_level_map.get(args.log_level.upper(), logging.INFO)
    if args.debug:
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    # Suppress curobo INFO logs
    logging.getLogger("curobo").setLevel(logging.WARNING)

    # Determine which tasks to run
    if args.task_indices is not None:
        # Parse string format: "0,1,2,3" or "0-10"
        task_str = args.task_indices
        if '-' in task_str:
            # Range format: "0-10"
            start, end = task_str.split('-')
            task_indices = list(range(int(start), int(end) + 1))
        else:
            # Comma-separated: "0,1,2,3"
            task_indices = [int(x.strip()) for x in task_str.split(',')]
        logging.info(f"Running multiple tasks: {task_indices}")
    else:
        task_indices = [args.task_idx]
        logging.info(f"Running single task: {args.task_idx}")

    logging.info("=" * 80)
    logging.info("Starting inference...")
    logging.info(f"Debug mode: {args.debug}, Log level: {args.log_level}")
    logging.info(f"Checkpoint: {args.ckpt_path}")
    logging.info(f"PTQ enabled: {args.use_ptq}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # =========================================================================
    # Optimization: load the model once to avoid OOM from reloading per task
    # =========================================================================
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    policy, input_transforms, unnormalize_fn = build_policy_and_transforms(
        args.ckpt_path, args.stats_key, args.resize_size, dtype,
        compile_inference=args.compile_inference,
        base_model_path=args.base_model_path,
        loramoe_num_experts=args.loramoe_num_experts,
        loramoe_router_top_k=args.loramoe_router_top_k,
        loramoe_ab_routing=args.loramoe_ab_routing,
        loramoe_router_top_k_a=args.loramoe_router_top_k_a,
        loramoe_router_top_k_b=args.loramoe_router_top_k_b,
        loramoe_enable_a_experts=args.loramoe_enable_a_experts,
        loramoe_enable_b_experts=args.loramoe_enable_b_experts,
        loramoe_share_a_across_experts=args.loramoe_share_a_across_experts,
        use_visual_token_prune=args.use_visual_token_prune,
        enable_affordance=args.enable_affordance,
        lambda_affordance=args.lambda_affordance,
        affordance_dim=args.affordance_dim,
        enable_affordance_v2=args.enable_affordance_v2,
        enable_affordance_v3=args.enable_affordance_v3,
        affordance_v3_state_router_prior=args.affordance_v3_state_router_prior,
        disable_affordance_router_input=args.disable_affordance_router_input,
        affordance_action_horizon=args.affordance_action_horizon,
        affordance_action_decay_end=args.affordance_action_decay_end,
        affordance_gate_init=args.affordance_gate_init,
        affordance_router_lead_in_fraction=args.affordance_router_lead_in_fraction,
        use_ptq=args.use_ptq,
        ptq_mode=args.ptq_mode,
        ptq_backend=args.ptq_backend,
        ptq_experts=args.ptq_experts,
        ptq_target_modules=args.ptq_target_modules,
    )

    action_moe_counter = None
    if args.action_moe_activation_path is not None:
        if args.compile_inference:
            raise ValueError("Action MoE activation counting requires --args.no-compile-inference")
        action_moe_counter = ActionMoEActivationCounter(
            policy,
            Path(args.action_moe_activation_path),
            args.loramoe_num_experts,
        )
        task_indices = [
            task_idx for task_idx in task_indices if task_idx not in action_moe_counter.completed_tasks
        ]
        logging.info("Action MoE activation tasks remaining: %s", task_indices)

    # Run inference for each task
    all_results = []
    inference_time_history: list[float] = []
    inference_time_window: list[float] = []
    for task_idx in task_indices:
        logging.info("=" * 80)
        logging.info(f"Running task {task_idx}: {TASK_NAMES[task_idx]}")
        args.task_idx = task_idx
        if action_moe_counter is not None:
            action_moe_counter.start_task()
        # Pass the preloaded policy and transforms
        result = infer_once(
            args,
            policy=policy,
            input_transforms=input_transforms,
            unnormalize_fn=unnormalize_fn,
            inference_time_history=inference_time_history,
            inference_time_window=inference_time_window,
        )
        all_results.append(result)

        if args.result_markdown_path is not None:
            update_markdown_with_tasks(
                Path(args.result_markdown_path),
                str(args.ckpt_path),
                args.task_config,
                [result],
            )
            logging.info("Updated result markdown after task %s: %s", task_idx, args.result_markdown_path)

        if action_moe_counter is not None:
            action_moe_counter.completed_tasks.add(task_idx)
            matrix_path = action_moe_counter.save(
                task_idx=task_idx,
                task_name=TASK_NAMES[task_idx],
                checkpoint=str(args.ckpt_path),
                test_num=args.test_num,
            )
            print(
                f"[ActionMoEActivationCounter] saved after task {task_idx}: "
                f"{matrix_path}",
                flush=True,
            )
            logging.info(
                "Updated action MoE activation matrix after task %s: %s",
                task_idx,
                args.action_moe_activation_path,
            )
        # Clear cache after each task
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    # Print summary
    logging.info("=" * 80)
    logging.info("FINAL SUMMARY")
    logging.info("=" * 80)

    # Compute overall success rate
    total_success = sum(r['num_success'] for r in all_results)
    total_test = sum(r['num_test'] for r in all_results)
    overall_success_rate = total_success / total_test if total_test > 0 else 0

    for task_idx, result in zip(task_indices, all_results):
        success_rate = result.get('success_rate', 0)
        logging.info(f"Task {task_idx} ({result.get('task_name', TASK_NAMES[task_idx])}): {success_rate*100:.1f}% ({result['num_success']}/{result['num_test']})")

    logging.info(f"Overall: {overall_success_rate*100:.1f}% ({total_success}/{total_test})")

    if torch.cuda.is_available():
        peak_allocated = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        logging.info(
            "Peak GPU memory | allocated: %s | reserved: %s",
            format_cuda_memory_gb(peak_allocated),
            format_cuda_memory_gb(peak_reserved),
        )

    # Save results to a JSON file
    output_dir = args.video_dir.parent
    result_file = output_dir / "eval_results.json"

    # Prepare data to save
    save_data = {
        'checkpoint': str(args.ckpt_path),
        'num_tasks': len(all_results),
        'overall_success_rate': overall_success_rate,
        'total_success': total_success,
        'total_test': total_test,
        'tasks': []
    }

    for task_idx, result in zip(task_indices, all_results):
        save_data['tasks'].append({
            'task_idx': task_idx,
            'task_name': result.get('task_name', TASK_NAMES[task_idx]),
            'success_rate': result['success_rate'],
            'num_success': result['num_success'],
            'num_test': result['num_test'],
        })

    # Save to file
    with open(result_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    logging.info(f"Results saved to: {result_file}")

    return all_results


if __name__ == "__main__":
    tyro.cli(main)
