#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# RoboTwin Federated Learning + LoRA-MoE Training Script.
#
# Usage:
#   torchrun --nproc_per_node=3 src/lerobot/scripts/lerobot_fl_robotwin_moe.py
#
# Environment variables for FL parameters:
#   FL_NUM_CLIENTS=4
#   FL_LOCAL_STEPS=100
#   FL_NUM_ROUNDS=50
#   FL_AGGREGATION=fedavg
#   FL_PARTITION_STRATEGY=by_task
#   MOE_STEPS=50
#   MOE_K_EXPERTS=2
#
# 5-Phase Training Flow (LoRA-MoE Federated Learning):
#   Phase 1: Local Client Training (Standard LoRA mode)
#   Phase 2: Expert Injection & Sync (Broadcast trained experts to all ranks)
#   Phase 3: MoE Training (optional affordance router lead-in, then joint training)
#   Phase 4: Global Aggregation (AllReduce sync + mean of all experts)
#   Phase 5: Weight Mixing & Distribution (new_local = 0.5 * local + 0.5 * global)
#   Then: Next round

import logging
import os
import sys
import time
from copy import deepcopy
from numbers import Real
from pathlib import Path
from pprint import pformat

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
from accelerate import Accelerator
from safetensors.torch import save_file

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.robotwin_federated_dataset import (
    ROBOTWIN_TASKS,
    create_robottwin_partitioner,
)
from lerobot.datasets.utils import write_json, serialize_dict
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import format_time, AverageMeter, MetricsTracker
from lerobot.utils.loramoe_weight_mapping import (
    inject_standard_to_expert,
    inject_state_to_expert,
    extract_expert_to_standard,
    aggregate_experts_to_global,
    average_complete_lora_adapters,
    get_lora_adapter_scaling,
    sync_expert_weights,
    load_client_state_to_model,
    validate_complete_lora_state_dict,
)
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    save_checkpoint,
)
from lerobot.utils.federated_utils import fedavg_aggregate_state_dicts
from lerobot.utils.fedforesight_utils import path_consensus_expert_weights, smooth_and_impute_pcea_weights
from lerobot.utils.affordance_utils import (
    clear_loramoe_expert_gradients,
    compute_router_lead_in_steps,
    should_accumulate_router_usage,
)
from lerobot.rl.wandb_utils import WandBLogger


def save_trainable_checkpoint(
    checkpoint_dir: Path,
    step: int,
    cfg: TrainPipelineConfig,
    policy,
    data_stats: dict | None = None,
) -> None:
    """Save only trainable parameters (LoRA + Action Head) to checkpoint."""
    import logging
    logger = logging.getLogger()

    pretrained_dir = checkpoint_dir / "pretrained_model"
    pretrained_dir.mkdir(parents=True, exist_ok=True)

    # Extract trainable parameters only
    trainable_state = {}
    action_head_keys = []
    for name, param in policy.named_parameters():
        if param.requires_grad:
            trainable_state[name] = param.data.cpu().clone()
            if "action_in_proj" in name or "action_out_proj" in name or "action_time_mlp" in name:
                action_head_keys.append(name)

    logger.info(f"Saving trainable params: {len(trainable_state)} keys")
    if action_head_keys:
        logger.info(f"  Action head keys: {action_head_keys}")
    else:
        logger.warning(f"  No action head keys found in trainable params!")
    save_file(trainable_state, str(pretrained_dir / "model.safetensors"))

    if hasattr(policy, 'config'):
        policy.config.save_pretrained(pretrained_dir)

    cfg.save_pretrained(pretrained_dir)

    if data_stats is not None:
        write_json(serialize_dict(data_stats), pretrained_dir / 'stats.json')

    from lerobot.utils.train_utils import save_training_state
    save_training_state(checkpoint_dir, step, None, None)


def save_moe_checkpoint(
    checkpoint_dir: Path,
    step: int,
    fl_round: int,
    policy,
    cfg: TrainPipelineConfig,
    data_stats: dict | None = None,
) -> None:
    """Save MoE model checkpoint (router + all expert weights).

    This saves only the MoE-related parameters (router weights and expert LoRA weights).
    """
    import logging
    logger = logging.getLogger()

    moe_dir = checkpoint_dir / "moe_model"
    moe_dir.mkdir(parents=True, exist_ok=True)

    # Extract MoE parameters only (router + expert LoRA weights + action head)
    moe_state = {}
    action_head_keys = []
    state_proj_keys = []
    for name, param in policy.named_parameters():
        # Include router weights (PEFT uses 'lora_router')
        if "lora_router" in name:
            moe_state[name] = param.data.cpu().clone()
        # Include expert LoRA weights (new format: lora_A.{i}.weight, lora_B.{i}.weight)
        elif ".lora_A." in name or ".lora_B." in name:
            moe_state[name] = param.data.cpu().clone()
        # Include state_proj router and expert LoRA weights (ModuleList with integer indices)
        elif "state_proj_router" in name or "state_proj_lora_A_moe" in name or "state_proj_lora_B_moe" in name:
            moe_state[name] = param.data.cpu().clone()
            state_proj_keys.append(name)
        # Include action head full fine-tuning weights (action_in_proj, action_out_proj, action_time_mlp)
        elif any(x in name for x in ["action_in_proj", "action_out_proj", "action_time_mlp"]) and "weight" in name:
            moe_state[name] = param.data.cpu().clone()
            action_head_keys.append(name)
        elif _is_affordance_parameter(name):
            moe_state[name] = param.data.cpu().clone()

    logger.info(f"Saving MoE params: {len(moe_state)} keys")
    if action_head_keys:
        logger.info(f"  Action head keys: {action_head_keys}")

    # Save MoE weights as safetensors
    save_file(moe_state, str(moe_dir / "moe_weights.safetensors"))

    # Save router config (number of experts, top-k, etc.)
    moe_config = {
        "fl_round": fl_round,
        "step": step,
        "num_experts": getattr(cfg.policy, 'loramoe_num_experts', 4),
        "router_top_k": getattr(cfg.policy, 'loramoe_router_top_k', 2),
    }
    write_json(moe_config, moe_dir / "moe_config.json")

    # Save policy config (config.json) - consistent with save_trainable_checkpoint
    if hasattr(policy, 'config'):
        policy.config.save_pretrained(moe_dir)

    # Save train config (train_config.json) - consistent with save_trainable_checkpoint
    cfg.save_pretrained(moe_dir)

    # Save data stats if provided - consistent with save_trainable_checkpoint
    if data_stats is not None:
        write_json(serialize_dict(data_stats), moe_dir / 'stats.json')

    # Save training step
    from lerobot.utils.train_utils import save_training_state
    save_training_state(checkpoint_dir, step, None, None)

    logger.info(f"MoE checkpoint saved to {moe_dir}")


def get_fl_config():
    """Get FL configuration from environment variables."""
    return {
        "training_flow": os.environ.get("FL_TRAINING_FLOW", "lora_moe"),
        "num_clients": int(os.environ.get("FL_NUM_CLIENTS", "4")),
        "num_gpus": int(os.environ.get("NUM_GPUS", "3")),
        "local_steps": int(os.environ.get("FL_LOCAL_STEPS", "100")),
        "local_epochs": int(os.environ.get("FL_LOCAL_EPOCHS", "1")),
        "num_rounds": int(os.environ.get("FL_NUM_ROUNDS", "50")),
        "aggregation_strategy": os.environ.get("FL_AGGREGATION", "fedavg"),
        "fedprox_mu": float(os.environ.get("FL_FEDPROX_MU", "0.01")),
        "seed": int(os.environ.get("FL_SEED", "42")),
        "save_freq": int(os.environ.get("FL_SAVE_FREQ", "10")),
        "log_freq": int(os.environ.get("FL_LOG_FREQ", "200")),
        "output_dir": os.environ.get("FL_OUTPUT_DIR", None),
        "partition_strategy": os.environ.get("FL_PARTITION", "by_task"),
        "held_out_tasks": int(os.environ.get("FL_HELD_OUT_TASKS", "5")),
        "dirichlet_alpha": float(os.environ.get("FL_DIRICHLET_ALPHA", "0.5")),
        # MoE parameters
        "moe_steps": int(os.environ.get("MOE_STEPS", "50")),
        "moe_router_type": os.environ.get("MOE_ROUTER_TYPE", "soft"),
        "moe_k_experts": int(os.environ.get("MOE_K_EXPERTS", "2")),
        "moe_temperature": float(os.environ.get("MOE_TEMPERATURE", "1.0")),
        "moe_loss_weight": float(os.environ.get("MOE_LOSS_WEIGHT", "0.01")),
        "moe_load_balance_weight": float(os.environ.get("MOE_LOAD_BALANCE_WEIGHT", "0.1")),
        "moe_debug_freq": int(os.environ.get("MOE_DEBUG_FREQ", "10")),
        "moe_save_freq": int(os.environ.get("MOE_SAVE_FREQ", "5")),  # MoE checkpoint save frequency (in rounds)
        "moe_batch_size": int(os.environ.get("MOE_BATCH_SIZE", "1")),  # MoE training batch size
        "moe_profiling": os.environ.get("MOE_PROFILING", "false").lower() == "true",  # Enable MoE profiling
        "moe_router_weighted_aggregation": os.environ.get("MOE_ROUTER_WEIGHTED_AGGREGATION", "false").lower() == "true",
        "moe_router_agg_temperature": float(os.environ.get("MOE_ROUTER_AGG_TEMPERATURE", "0.5")),
    }


def assign_clients_to_gpu(local_rank: int, num_gpus: int, num_clients: int) -> list[int]:
    """Assign clients to each GPU (round-robin)."""
    clients = []
    for i in range(local_rank, num_clients, num_gpus):
        clients.append(i)
    return clients


def setup_distributed():
    """Initialize distributed training."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        print(f"==> [PID {os.getpid()}] Pre-binding to GPU {local_rank}")

    if world_size > 1:
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", "29501")
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://{}:{}".format(master_addr, master_port),
            world_size=world_size,
            rank=local_rank,
        )

    return local_rank, world_size


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def broadcast_model(model, src_rank=0):
    """Broadcast model parameters from source rank."""
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return

    for param in model.parameters():
        dist.broadcast(param.data, src=src_rank)


def sync_experts(model, my_client_ids, local_rank=0):
    """Broadcast each client's expert weights to all other ranks.

    After Phase 1, each rank has trained different experts. This function
    broadcasts each trained expert to all other ranks so that all ranks have
    all expert weights (the ones they trained themselves + ones from other ranks).

    This is NOT averaging - it's broadcasting and overwriting.
    """
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Get the inner model (handle QwenA1Policy wrapping)
    if hasattr(model, 'model'):
        target_model = model.model
    else:
        target_model = model

    # Get all expert IDs this rank is responsible for
    my_expert_ids = list(my_client_ids)

    # For each expert this rank is responsible for, broadcast to all other ranks
    for expert_id in my_expert_ids:
        exp_key = str(expert_id)

        # Collect params for this specific expert
        # PEFT LoRA-MoE: uses lora_A/lora_B ModuleDict with string keys
        expert_params = []
        for module in target_model.modules():
            if hasattr(module, 'lora_A') and exp_key in module.lora_A:
                expert_params.append(module.lora_A[exp_key].weight.data)
            if hasattr(module, 'lora_B') and exp_key in module.lora_B:
                expert_params.append(module.lora_B[exp_key].weight.data)

        # Broadcast this expert's params from this rank to all other ranks
        for param in expert_params:
            dist.broadcast(param, src=rank)


def update_policy(policy, batch, grad_clip_norm, local_rank=0):
    """Single training step (no accelerator)."""
    policy.train()

    try:
        loss, output_dict = policy.forward(batch)
        loss.backward()
    except Exception as e:
        print(f"[ERROR rank{local_rank}] Exception in forward/backward: {e}")
        import traceback
        traceback.print_exc()
        raise

    if grad_clip_norm > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    return loss.item(), output_dict, grad_norm.item()


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensors in batch dictionary to device."""
    if not isinstance(batch, dict):
        if hasattr(batch, 'to'):
            return batch.to(device)
        return batch

    moved_batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved_batch[key] = value.to(device, non_blocking=True)
        elif isinstance(value, dict):
            moved_batch[key] = move_batch_to_device(value, device)
        elif isinstance(value, list):
            moved_batch[key] = [
                (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                for v in value
            ]
        else:
            moved_batch[key] = value
    return moved_batch


def move_to_cpu_recursive(obj):
    """Recursively move all tensors in nested structure to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: move_to_cpu_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_cpu_recursive(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_cpu_recursive(v) for v in obj)
    else:
        return obj


def detach_to_cpu_recursive(obj):
    """Recursively detach tensors and move them to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    elif isinstance(obj, dict):
        return {k: detach_to_cpu_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [detach_to_cpu_recursive(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(detach_to_cpu_recursive(v) for v in obj)
    else:
        return obj


def move_to_device_recursive(obj, device):
    """Recursively move all tensors in nested structure to specified device."""
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device_recursive(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device_recursive(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device_recursive(v, device) for v in obj)
    else:
        return obj

# =============================================================================
# LoRA-MoE Helper Functions
# =============================================================================

def set_all_experts_trainable(model):
    """
    Set all experts as trainable (for MoE training).
    Also sets router weights, state_proj LoRA, and full fine-tuning params as trainable.

    Note: Supports PEFT LoRA-MoE with lora_A/lora_B ModuleDict.
    """
    # Get the inner model (handle QwenA1Policy wrapping)
    if hasattr(model, 'model'):
        target_model = model.model
    else:
        target_model = model

    # Set LoRA-MoE specific parameters
    # PEFT LoRA-MoE: uses lora_A/lora_B ModuleDict
    for module in target_model.modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # Set router weights as trainable
            if hasattr(module, 'lora_router') and module.lora_router is not None:
                module.lora_router.requires_grad = True

            # Set all expert LoRA weights as trainable
            for exp_key in module.lora_A:
                for param in module.lora_A[exp_key].parameters():
                    param.requires_grad = True
            for exp_key in module.lora_B:
                for param in module.lora_B[exp_key].parameters():
                    param.requires_grad = True

    # Set state_proj MoE weights as trainable (ModuleList with integer indices)
    if hasattr(target_model, 'state_proj_lora_A_moe'):
        for layer in target_model.state_proj_lora_A_moe:
            for param in layer.parameters():
                param.requires_grad = True
    if hasattr(target_model, 'state_proj_lora_B_moe'):
        for layer in target_model.state_proj_lora_B_moe:
            for param in layer.parameters():
                param.requires_grad = True
    if hasattr(target_model, 'state_proj_router'):
        target_model.state_proj_router.requires_grad = True

    # Set full fine-tuning parameters as trainable (action expert)
    for name, param in model.named_parameters():
        if any(x in name for x in ["action_in_proj", "action_out_proj", "action_time_mlp", "affordance_"]):
            param.requires_grad = True


def set_loramoe_mode(model, mode: str = "lora", active_client_id: int = 0, num_experts: int = None, router_top_k: int = None, temperature: float = 1.0):
    """
    Set LoRA-MoE mode for all LoRA layers in the model.

    Args:
        model: PEFT model with LoRA-MoE layers (QwenA1Policy wrapper)
        mode: "lora" (standard LoRA) or "lora_moe" (Mixture of Experts)
        active_client_id: Which client to use in lora mode
        num_experts: Number of experts (if None, auto-detect from model)
        router_top_k: Number of top experts to select (if None, auto-detect)
        temperature: Temperature for routing
    """
    # Get the inner model (handle QwenA1Policy wrapping)
    if hasattr(model, 'model'):
        target_model = model.model
    else:
        target_model = model

    # Convert mode name for internal use
    internal_mode = "lora_moe" if mode == "moe" else "lora"

    # Use QwenA1's set_lora_mode method
    if hasattr(target_model, 'set_lora_mode'):
        target_model.set_lora_mode(internal_mode)
    else:
        # Fallback: set mode on individual layers
        for module in target_model.modules():
            if hasattr(module, 'set_lora_moe_mode'):
                enabled = (mode == "moe")
                module.set_lora_moe_mode(enabled)


def sync_client_to_expert(policy, client_id: int, adapter_name: str = "default"):
    """
    Sync a client's trained LoRA weights to its expert slot.

    Args:
        policy: PEFT model with LoRA-MoE layers
        client_id: Client ID (0 to num_experts-1)
        adapter_name: Name of the adapter to sync from
    """
    exp_key = str(client_id)

    # Get the inner model (handle QwenA1Policy wrapping)
    if hasattr(policy, 'model'):
        target_model = policy.model
    else:
        target_model = policy

    # PEFT LoRA-MoE: lora_A/lora_B are ModuleDict with string keys
    for name, module in target_model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            if adapter_name in module.lora_A and exp_key in module.lora_A:
                # Copy from adapter_name (e.g., "default") to exp_key (e.g., "0")
                module.lora_A[exp_key].weight.data.copy_(module.lora_A[adapter_name].weight.data)
                module.lora_B[exp_key].weight.data.copy_(module.lora_B[adapter_name].weight.data)


def sync_expert_to_client(policy, client_id: int, adapter_name: str = "default"):
    """Sync an expert's weights back to the adapter."""
    exp_key = str(client_id)

    # Get the inner model (handle QwenA1Policy wrapping)
    if hasattr(policy, 'model'):
        target_model = policy.model
    else:
        target_model = policy

    # PEFT LoRA-MoE: lora_A/lora_B are ModuleDict with string keys
    for name, module in target_model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            if exp_key in module.lora_A and adapter_name in module.lora_A:
                # Copy from exp_key to adapter_name
                module.lora_A[adapter_name].weight.data.copy_(module.lora_A[exp_key].weight.data)
                module.lora_B[adapter_name].weight.data.copy_(module.lora_B[exp_key].weight.data)


def extract_action_head_from_client_states(client_states: dict, num_clients: int) -> dict:
    """
    Extract action head parameters from client_states.

    Action head includes: action_in_proj, action_out_proj, action_time_mlp

    Args:
        client_states: Dict mapping client_id to their state_dict
        num_clients: Total number of clients

    Returns:
        Dict mapping client_id to their action head state dict
    """
    import logging
    logger = logging.getLogger()

    action_head_params = {}

    for client_id in range(num_clients):
        if client_id not in client_states:
            continue

        client_state = client_states[client_id]
        action_head_state = {}

        for name, param in client_state.items():
            # Match action head parameter names
            if "action_in_proj" in name or "action_out_proj" in name or "action_time_mlp" in name:
                action_head_state[name] = param.clone()

        if action_head_state:
            action_head_params[client_id] = action_head_state

    logger.info(f"Extracted action head from {len(action_head_params)} clients")
    if action_head_params:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            sample_client = list(action_head_params.keys())[0]
            logger.info(f"  Sample keys from client {sample_client}: {list(action_head_params[sample_client].keys())[:5]}")

    return action_head_params


def extract_action_head_from_model(policy) -> dict:
    """
    Extract action head parameters from policy model.
    Only extracts weights (not biases).

    Args:
        policy: The policy model

    Returns:
        Dict of action head parameters (weights only)
    """
    action_head_state = {}

    for name, param in policy.named_parameters():
        # Match action head parameter names - ONLY weights (not bias)
        # Only include parameters that end with .weight
        if ("action_in_proj.weight" in name or
            "action_out_proj.weight" in name or
            "action_time_mlp_in.weight" in name or
            "action_time_mlp_out.weight" in name):
            action_head_state[name] = param.data.clone()

    return action_head_state


def aggregate_action_heads(action_head_list: list[dict]) -> dict:
    """
    FedAvg aggregation of action head parameters from multiple clients.

    Args:
        action_head_list: List of action head state dicts from clients

    Returns:
        Aggregated action head state dict
    """
    import logging
    logger = logging.getLogger()

    if not action_head_list:
        return {}

    # Use uniform weights
    num_clients = len(action_head_list)
    weights = [1.0 / num_clients] * num_clients

    # Get all parameter names from first client
    param_names = list(action_head_list[0].keys())

    aggregated = {}
    for param_name in param_names:
        aggregated_param = None

        for client_state, weight in zip(action_head_list, weights):
            if param_name not in client_state:
                continue
            client_param = client_state[param_name].float()

            if aggregated_param is None:
                aggregated_param = weight * client_param
            else:
                aggregated_param += weight * client_param

        if aggregated_param is not None:
            aggregated[param_name] = aggregated_param

    logger.info(f"Aggregated action head: {len(aggregated)} parameters")
    if aggregated:
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            logger.info(f"  Sample keys: {list(aggregated.keys())[:5]}")

    return aggregated


def inject_action_head_to_model(policy, action_head_state: dict):
    """
    Inject action head parameters to policy model.

    Args:
        policy: The policy model to update
        action_head_state: Dict of action head parameters to inject
    """
    import logging
    logger = logging.getLogger()

    if not action_head_state:
        logger.warning("No action head state to inject!")
        return

    updated_count = 0
    skipped_count = 0
    sample_updates = []
    for name, param in policy.named_parameters():
        if name in action_head_state:
            source_data = action_head_state[name]
            # Check source data validity
            if torch.isnan(source_data).any() or torch.isinf(source_data).any():
                logger.warning(f"[inject_action_head] Skipping {name} - contains NaN/Inf")
                skipped_count += 1
                continue
            param.data.copy_(source_data.to(param.device))
            updated_count += 1
            if len(sample_updates) < 3:
                sample_updates.append({
                    'name': name,
                    'norm': param.data.norm().item(),
                    'mean': param.data.mean().item(),
                })

    # Log detailed injection results
    if sample_updates:
        logger.info(f"[inject_action_head] Injected {updated_count} params, skipped {skipped_count}")
        for s in sample_updates:
            logger.info(f"  {s['name']}: norm={s['norm']:.6f}, mean={s['mean']:.6f}")
    else:
        logger.warning(f"[inject_action_head] WARNING: No params were injected! updated={updated_count}, skipped={skipped_count}")

    # Also check what params in policy were NOT in action_head_state
    policy_ah_params = [n for n, _ in policy.named_parameters()
                        if 'action_in_proj' in n or 'action_out_proj' in n or 'action_time_mlp' in n]
    missing_params = [n for n in policy_ah_params if n not in action_head_state]
    if missing_params:
        logger.warning(f"[inject_action_head] Missing params (not in action_head_state): {missing_params[:5]}")



def broadcast_action_head(action_head_state: dict, src_rank: int = 0):
    """
    Broadcast action head parameters across all ranks using distributed communication.

    Args:
        action_head_state: Dict of action head parameters (only valid on src_rank)
        src_rank: Source rank that has the valid action head

    Returns:
        Dict of action head parameters on all ranks
    """
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return action_head_state

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Get the device from the first tensor (or use cuda:0)
    device = torch.device(f"cuda:{rank}")

    # Broadcast each parameter
    for name in action_head_state:
        param = action_head_state[name]

        # Move to GPU for broadcasting
        param_gpu = param.to(device)

        # Broadcast the tensor
        dist.broadcast(param_gpu, src=src_rank)

        # Update the dict (on non-src ranks, this will have the received data)
        action_head_state[name] = param_gpu.cpu()

    return action_head_state


def get_trainable_weights(policy):
    """Get trainable weights while keeping every serialized LoRA adapter complete."""
    import logging
    logger = logging.getLogger()
    trainable_state = {}
    for name, param in policy.named_parameters():
        is_lora_factor = "lora_A" in name or "lora_B" in name
        if param.requires_grad or is_lora_factor:
            trainable_state[name] = param.data.cpu().clone()

    # Debug: print sample parameter names
    if not hasattr(get_trainable_weights, '_logged'):
        get_trainable_weights._logged = True
        lora_keys = [k for k in trainable_state.keys() if 'lora_A' in k or 'lora_B' in k]
        logger.info(f"[DEBUG get_trainable_weights] Total trainable params: {len(trainable_state)}")
        logger.info(f"[DEBUG get_trainable_weights] Sample LoRA keys: {lora_keys[:5]}")
        if lora_keys:
            # Check for base_layer in keys
            has_base_layer = any('.base_layer' in k for k in lora_keys)
            logger.info(f"[DEBUG get_trainable_weights] Keys have .base_layer: {has_base_layer}")

    validate_complete_lora_state_dict(trainable_state, context="client upload")
    return trainable_state


def _is_affordance_parameter(name: str) -> bool:
    return "affordance_" in name


def aggregate_shared_affordance_state(
    server_policy,
    client_states: dict[int, dict],
    local_client_ids: list[int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Uniformly FedAvg the small shared affordance branch across all clients."""
    server_params = {
        name: param
        for name, param in server_policy.named_parameters()
        if _is_affordance_parameter(name)
    }
    if not server_params:
        return {}

    aggregated = {}
    local_count = sum(
        1
        for client_id in local_client_ids
        if client_id in client_states
    )
    count = torch.tensor(float(local_count), device=device, dtype=torch.float32)
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    if float(count.item()) <= 0:
        raise RuntimeError("Affordance FedAvg found no trained client states")

    for name, server_param in server_params.items():
        local_sum = torch.zeros_like(server_param, device=device, dtype=torch.float32)
        for client_id in local_client_ids:
            client_state = client_states.get(client_id, {})
            if name not in client_state:
                raise RuntimeError(f"Client {client_id} is missing shared affordance parameter {name}")
            local_sum.add_(client_state[name].to(device=device, dtype=torch.float32))
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
        averaged = local_sum / count
        server_param.data.copy_(averaged.to(dtype=server_param.dtype))
        aggregated[name] = averaged.detach().cpu()
    return aggregated


def extract_shared_affordance_state(policy) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in policy.named_parameters()
        if _is_affordance_parameter(name)
    }


def distribute_shared_affordance_state(
    client_states: dict[int, dict],
    affordance_state: dict[str, torch.Tensor],
    num_clients: int,
) -> None:
    for client_id in range(num_clients):
        client_states.setdefault(client_id, {})
        for name, value in affordance_state.items():
            client_states[client_id][name] = value.detach().cpu().clone()


def set_trainable_weights(policy, state_dict):
    """Set trainable weights to policy."""
    for name, param in policy.named_parameters():
        if name in state_dict and param.requires_grad:
            param.data.copy_(state_dict[name].to(param.device))


# =============================================================================
# Training Functions
# =============================================================================

def run_local_training(policy, dl_iter, optimizer, cfg, fl_cfg, lr_scheduler=None, local_rank=0, wandb_logger=None, global_start_step=0):
    """Run local training for one federated round."""
    import logging
    logger = logging.getLogger()

    policy.train()

    local_steps = fl_cfg["local_steps"]
    local_epochs = fl_cfg["local_epochs"]
    log_freq = fl_cfg["log_freq"]
    batch_size = cfg.batch_size

    if cfg.policy.type in ["a1", "qwena1"]:
        train_metrics = {
            "loss": AverageMeter("loss", ":.3f"),
            "loss_action": AverageMeter("loss_action", ":.3f"),
            "loss_gen": AverageMeter("loss_gen", ":.3f"),
            "loss_affordance": AverageMeter("loss_aff", ":.3f"),
            "loss_affordance_weighted": AverageMeter("loss_aff_w", ":.3f"),
            "loss_affordance_action_align": AverageMeter("loss_aff_act", ":.3f"),
            "affordance_target_valid": AverageMeter("aff_valid", ":.3f"),
            "affordance_spatial_entropy": AverageMeter("aff_sp_ent", ":.3f"),
            "affordance_spatial_peak": AverageMeter("aff_sp_peak", ":.3f"),
            "affordance_action_gate": AverageMeter("aff_act_gate", ":.3f"),
            "affordance_router_gate": AverageMeter("aff_route_gate", ":.3f"),
            "loss_visual_token_prune_ot": AverageMeter("loss_ot", ":.3f"),
            "grad_norm": AverageMeter("grdn", ":.3f"),
            "lr": AverageMeter("lr", ":0.1e"),
            "update_s": AverageMeter("updt_s", ":.3f"),
            "dataloading_s": AverageMeter("data_s", ":.3f"),
        }
    else:
        train_metrics = {
            "loss": AverageMeter("loss", ":.3f"),
            "grad_norm": AverageMeter("grdn", ":.3f"),
            "lr": AverageMeter("lr", ":0.1e"),
            "update_s": AverageMeter("updt_s", ":.3f"),
            "dataloading_s": AverageMeter("data_s", ":.3f"),
        }

    train_tracker = MetricsTracker(
        batch_size=batch_size,
        num_frames=local_steps * batch_size,
        num_episodes=local_steps,
        metrics=train_metrics,
        initial_step=global_start_step,
    )

    metrics_history = []
    optimizer.zero_grad()

    device = next(policy.parameters()).device

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)

    for _ in range(local_epochs):
        for step in range(local_steps):
            start_time = time.perf_counter()

            batch = next(dl_iter)
            batch = move_batch_to_device(batch, device)

            data_loading_time = time.perf_counter() - start_time

            loss_val, output_dict, grad_norm = update_policy(policy, batch, cfg.optimizer.grad_clip_norm, local_rank)

            optimizer.step()
            optimizer.zero_grad()

            if lr_scheduler is not None:
                lr_scheduler.step()

            update_time = time.perf_counter() - start_time

            train_tracker.loss = loss_val
            train_tracker.grad_norm = grad_norm
            train_tracker.lr = optimizer.param_groups[0]["lr"]
            train_tracker.dataloading_s = data_loading_time
            train_tracker.update_s = update_time
            if "loss_action" in output_dict:
                train_tracker.loss_action = output_dict["loss_action"]
            if "loss_gen" in output_dict:
                train_tracker.loss_gen = output_dict["loss_gen"]
            if "loss_affordance" in output_dict:
                train_tracker.loss_affordance = output_dict["loss_affordance"]
                train_tracker.loss_affordance_weighted = output_dict["loss_affordance_weighted"]
                train_tracker.loss_affordance_action_align = output_dict.get(
                    "loss_affordance_action_align", 0.0
                )
                train_tracker.affordance_target_valid = output_dict.get("affordance_target_valid", 0.0)
                train_tracker.affordance_spatial_entropy = output_dict.get("affordance_spatial_entropy", 0.0)
                train_tracker.affordance_spatial_peak = output_dict.get("affordance_spatial_peak", 0.0)
                train_tracker.affordance_action_gate = output_dict.get("affordance_action_gate", 0.0)
                train_tracker.affordance_router_gate = output_dict.get("affordance_router_gate", 0.0)
            if "loss_visual_token_prune_ot" in output_dict:
                train_tracker.loss_visual_token_prune_ot = output_dict["loss_visual_token_prune_ot"]

            metrics_history.append({
                "loss": loss_val,
                "grad_norm": grad_norm,
                "lr": optimizer.param_groups[0]["lr"],
                "dataloading_s": data_loading_time,
                "update_s": update_time,
                "loss_action": output_dict.get("loss_action"),
                "loss_gen": output_dict.get("loss_gen"),
                "loss_affordance": output_dict.get("loss_affordance", 0.0),
                "loss_affordance_weighted": output_dict.get("loss_affordance_weighted", 0.0),
                "loss_affordance_action_align": output_dict.get("loss_affordance_action_align", 0.0),
                "affordance_target_valid": output_dict.get("affordance_target_valid", 0.0),
                "affordance_spatial_entropy": output_dict.get("affordance_spatial_entropy", 0.0),
                "affordance_spatial_peak": output_dict.get("affordance_spatial_peak", 0.0),
                "affordance_action_gate": output_dict.get("affordance_action_gate", 0.0),
                "affordance_router_gate": output_dict.get("affordance_router_gate", 0.0),
                "loss_visual_token_prune_ot": output_dict.get("loss_visual_token_prune_ot", 0.0),
            })

            train_tracker.step()

            del batch, loss_val, output_dict

    avg_metrics = {
        "loss": sum(m["loss"] for m in metrics_history) / len(metrics_history),
        "grad_norm": sum(m["grad_norm"] for m in metrics_history) / len(metrics_history),
        "lr": sum(m["lr"] for m in metrics_history) / len(metrics_history),
        "dataloading_s": sum(m["dataloading_s"] for m in metrics_history) / len(metrics_history),
        "update_s": sum(m["update_s"] for m in metrics_history) / len(metrics_history),
    }
    if "loss_action" in metrics_history[0] and metrics_history[0]["loss_action"] is not None:
        avg_metrics["loss_action"] = sum(m.get("loss_action", 0) for m in metrics_history) / len(metrics_history)
    if "loss_gen" in metrics_history[0] and metrics_history[0]["loss_gen"] is not None:
        avg_metrics["loss_gen"] = sum(m.get("loss_gen", 0) for m in metrics_history) / len(metrics_history)
    avg_metrics["loss_affordance"] = sum(
        m.get("loss_affordance", 0) for m in metrics_history
    ) / len(metrics_history)
    avg_metrics["loss_affordance_weighted"] = sum(
        m.get("loss_affordance_weighted", 0) for m in metrics_history
    ) / len(metrics_history)
    avg_metrics["loss_affordance_action_align"] = sum(
        m.get("loss_affordance_action_align", 0) for m in metrics_history
    ) / len(metrics_history)
    avg_metrics["affordance_target_valid"] = sum(
        m.get("affordance_target_valid", 0) for m in metrics_history
    ) / len(metrics_history)
    avg_metrics["affordance_spatial_entropy"] = sum(
        m.get("affordance_spatial_entropy", 0) for m in metrics_history
    ) / len(metrics_history)
    avg_metrics["affordance_spatial_peak"] = sum(
        m.get("affordance_spatial_peak", 0) for m in metrics_history
    ) / len(metrics_history)
    avg_metrics["affordance_action_gate"] = sum(
        m.get("affordance_action_gate", 0) for m in metrics_history
    ) / len(metrics_history)
    avg_metrics["affordance_router_gate"] = sum(
        m.get("affordance_router_gate", 0) for m in metrics_history
    ) / len(metrics_history)
    if "loss_visual_token_prune_ot" in metrics_history[0]:
        avg_metrics["loss_visual_token_prune_ot"] = sum(
            m.get("loss_visual_token_prune_ot", 0) for m in metrics_history
        ) / len(metrics_history)

    # Get trainable state
    model_state = get_trainable_weights(policy)

    return {
        "metrics": avg_metrics,
        "model_state": model_state,
    }


def get_moe_params(policy):
    """
    Get MoE-specific parameters for the separate optimizer:
    - Router weights (lora_router)
    - Expert LoRA weights (lora_A.0, lora_B.1, etc.)
    - State projection LoRA weights (state_proj_lora_A_moe)
    - Full fine-tuning parameters (action_in_proj, action_out_proj, action_time_mlp)

    Note: Uses PEFT LoRA-MoE with lora_A/lora_B ModuleDict.
    """
    import logging
    logger = logging.getLogger()

    # Categorize params
    router_params = []
    expert_lora_params = []
    state_proj_params = []
    action_params = []
    affordance_params = []

    for name, param in policy.named_parameters():
        if not param.requires_grad:
            continue
        if _is_affordance_parameter(name):
            affordance_params.append(param)
        # Include router weights
        elif "lora_router" in name:
            router_params.append(param)
        # Include expert LoRA weights (new format: lora_A.0, lora_A.1, etc.)
        elif ".lora_A." in name or ".lora_B." in name:
            expert_lora_params.append(param)
        # Include state_proj LoRA MoE weights (ModuleList with integer indices)
        elif "state_proj_lora_A_moe" in name or "state_proj_lora_B_moe" in name:
            state_proj_params.append(param)
        # Include state_proj router weight
        elif "state_proj_router" in name:
            state_proj_params.append(param)
        # Include full fine-tuning parameters
        elif any(x in name for x in ["action_in_proj", "action_out_proj", "action_time_mlp"]) and "weight" in name:
            action_params.append(param)

    # Calculate total parameters (by count, not numel)
    logger.info(f"[get_moe_params] Router params: {len(router_params)}")
    logger.info(f"[get_moe_params] Expert LoRA params: {len(expert_lora_params)}")
    logger.info(f"[get_moe_params] State proj params: {len(state_proj_params)}")
    logger.info(f"[get_moe_params] Action params: {len(action_params)}")
    logger.info(f"[get_moe_params] Affordance params: {len(affordance_params)}")
    logger.info(
        f"[get_moe_params] Total params: "
        f"{len(router_params) + len(expert_lora_params) + len(state_proj_params) + len(action_params) + len(affordance_params)}"
    )

    # Print sample param names for each category
    if router_params:
        sample_router = [n for n, p in policy.named_parameters() if id(p) in {id(x) for x in router_params}][:2]
        logger.info(f"[get_moe_params] Router sample: {sample_router}")
    if expert_lora_params:
        expert_ids = {id(x) for x in expert_lora_params}
        sample_expert = [n for n, p in policy.named_parameters() if id(p) in expert_ids][:3]
        logger.info(f"[get_moe_params] Expert LoRA sample: {sample_expert}")
    if state_proj_params:
        state_ids = {id(x) for x in state_proj_params}
        sample_state = [n for n, p in policy.named_parameters() if id(p) in state_ids][:3]
        logger.info(f"[get_moe_params] State proj sample: {sample_state}")

    return router_params + expert_lora_params + state_proj_params + action_params + affordance_params


def create_moe_optimizer(policy, optimizer_config):
    """
    Create a separate optimizer for MoE parameters (router + experts).
    Uses the same LR as client optimizer (no 2x).
    """
    moe_params = get_moe_params(policy)
    if not moe_params:
        return None

    # Use same LR as client optimizer (no 2x)
    opt_config = deepcopy(optimizer_config)

    # Build optimizer with MoE params
    optimizer = opt_config.build(moe_params)

    return optimizer


def debug_router_stats(policy, step, log_freq=10):
    """Debug function to print router selection statistics and expert usage distribution.

    This monitors the MoE routing to detect:
    - Expert collapse (one expert dominates)
    - Unbalanced expert usage
    - Router weight instability
    """
    import logging
    logger = logging.getLogger()

    if step % log_freq != 0:
        return

    # Get the inner model (handle QwenA1Policy wrapping)
    if hasattr(policy, 'model'):
        target_model = policy.model
    else:
        target_model = policy

    # Collect expert usage counts from all MoE layers
    all_expert_counts = {}
    total_selections = 0
    num_layers = 0
    num_layers_with_counts = 0

    try:
        # PEFT LoRA-MoE: lora_A/lora_B ModuleDict
        for name, module in target_model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                num_layers += 1
                has_counts_attr = hasattr(module, '_expert_usage_counts')
                counts_attr = getattr(module, '_expert_usage_counts', None)
                if has_counts_attr and counts_attr is not None:
                    num_layers_with_counts += 1
                    counts = counts_attr.cpu()
                    for exp_id in range(len(counts)):
                        if exp_id not in all_expert_counts:
                            all_expert_counts[exp_id] = 0
                        all_expert_counts[exp_id] += counts[exp_id].item()
                        total_selections += counts[exp_id].item()
                else:
                    pass  # No counts yet, normal for first few steps

        # Log debug info
        if num_layers > 0 and total_selections == 0:
            logger.info(f"[MoE Monitor] Step {step}: {num_layers} MoE layers, {num_layers_with_counts} with counts, no selections yet")

        # Always log expert usage if available
        if num_layers > 0:
            logger.info(f"[MoE Monitor] Step {step}: {num_layers} MoE layers")
            if total_selections > 0:
                # Calculate statistics
                num_experts = len(all_expert_counts)
                usage_rates = {exp_id: count / total_selections for exp_id, count in all_expert_counts.items()}
                mean_usage = 1.0 / num_experts if num_experts > 0 else 0

                # Calculate imbalance: variance from uniform distribution
                variance = sum((rate - mean_usage) ** 2 for rate in usage_rates.values()) / num_experts
                std_dev = variance ** 0.5

                # Find most and least used experts
                sorted_usage = sorted(usage_rates.items(), key=lambda x: x[1], reverse=True)
                most_used = sorted_usage[0]
                least_used = sorted_usage[-1]

                # Log summary
                logger.info(f"[MoE Monitor] Step {step}: {num_layers} MoE layers, {total_selections} total selections")
                logger.info(f"[MoE Monitor] Expert usage distribution (uniform={mean_usage*100:.1f}%):")

                # Log each expert's usage
                for exp_id in sorted(all_expert_counts.keys()):
                    count = all_expert_counts[exp_id]
                    rate = usage_rates[exp_id]
                    deviation = (rate - mean_usage) * 100
                    bar = "█" * int(rate * 50)  # Visual bar
                    logger.info(f"  Expert {exp_id}: {rate*100:5.1f}% ({count:6d}) [{bar}] {deviation:+.1f}%")

                logger.info(f"[MoE Monitor] Balance: std={std_dev*100:.2f}%, most={most_used[0]}({most_used[1]*100:.1f}%), least={least_used[0]}({least_used[1]*100:.1f}%)")

                # Warn if severely imbalanced
                if std_dev > 0.2:
                    logger.warning(f"[MoE Monitor] WARNING: Expert usage severely imbalanced! (std={std_dev*100:.1f}%)")
                if most_used[1] > 0.8:
                    logger.warning(f"[MoE Monitor] WARNING: Expert collapse! Expert {most_used[0]} used {most_used[1]*100:.1f}%")
            else:
                logger.info(f"[MoE Monitor] Step {step}: No expert usage counts yet (counts will accumulate)")

        # Also check router weight norms
        if step == 0 or step % (log_freq * 10) == 0:
            router_count = 0
            # PEFT LoRA-MoE: lora_A/lora_B ModuleDict
            for name, module in target_model.named_modules():
                if hasattr(module, 'lora_router') and module.lora_router is not None:
                    router_weight = module.lora_router.data
                    weight_norm = router_weight.norm().item()
                    router_std = router_weight.std().item()
                    router_max = router_weight.abs().max().item()
                    logger.info(f"[MoE Router] Step {step} router_{router_count}: norm={weight_norm:.4f}, std={router_std:.4f}, max={router_max:.4f}")
                    router_count += 1

    except Exception as e:
        import traceback
        logger.warning(f"[MoE Monitor] Error: {e}")


def _unwrap_policy_model(policy):
    if hasattr(policy, 'module'):
        policy = policy.module
    return policy.model if hasattr(policy, 'model') else policy


_ROUTER_AGG_MASK_WARNED: set[str] = set()


def _router_valid_mask(target_model, module_name: str, batch: dict, num_rows: int, device: torch.device):
    if "und_expert" in module_name:
        mask = batch.get("observation.attention_mask")
        if mask is None:
            return None
        mask = mask.to(device=device, dtype=torch.bool).reshape(-1)
    elif "gen_expert" in module_name:
        feature_shape = getattr(target_model, "cosmos_feat_shape", None)
        image_masks = [batch.get(f"observation.images.image{i}_mask") for i in range(3)]
        if feature_shape is None or any(mask is None for mask in image_masks):
            return None
        batch_size, num_views, num_frames, _, height, width = feature_shape
        image_masks = [mask.reshape(batch_size, -1)[:, 0] for mask in image_masks]
        view_mask = torch.stack(image_masks, dim=1).to(device=device, dtype=torch.bool)
        if tuple(view_mask.shape) != (batch_size, num_views):
            return None
        mask = view_mask[:, :, None, None, None].expand(
            batch_size,
            num_views,
            num_frames,
            height,
            width,
        ).reshape(-1)
    elif "act_expert" in module_name:
        return torch.ones((num_rows,), device=device, dtype=torch.bool)
    else:
        return torch.ones((num_rows,), device=device, dtype=torch.bool)

    if mask.numel() != num_rows:
        return None
    return mask


def accumulate_router_expert_usage(policy, usage_sums: dict[str, torch.Tensor], batch: dict) -> None:
    target_model = _unwrap_policy_model(policy)
    with torch.no_grad():
        for name, module in target_model.named_modules():
            observables = getattr(module, '_last_router_observables', None)
            if not observables:
                continue
            probs = observables.get('router', {}).get('p')
            if probs is None:
                continue
            flat_probs = probs.detach().float().reshape(-1, probs.shape[-1])
            valid_mask = _router_valid_mask(target_model, name, batch, flat_probs.shape[0], flat_probs.device)
            if valid_mask is None:
                expert_name = next(
                    (candidate for candidate in ("und_expert", "gen_expert", "act_expert") if candidate in name),
                    "unknown",
                )
                if expert_name not in _ROUTER_AGG_MASK_WARNED:
                    logging.getLogger().warning(
                        f"[RouterAgg] Skipping usage for {name}: valid-mask shape could not be matched "
                        f"to {flat_probs.shape[0]} router rows"
                    )
                    _ROUTER_AGG_MASK_WARNED.add(expert_name)
                continue
            valid_probs = flat_probs[valid_mask]
            if valid_probs.numel() == 0:
                continue
            layer_sum = valid_probs.sum(dim=0).cpu()
            if name not in usage_sums:
                usage_sums[name] = torch.zeros_like(layer_sum)
            usage_sums[name].add_(layer_sum)

        usage = getattr(target_model, '_last_state_proj_router_probs', None)
        if usage is not None:
            layer_sum = usage.detach().float().reshape(-1, usage.shape[-1]).sum(dim=0).cpu()
            if 'model.state_proj' not in usage_sums:
                usage_sums['model.state_proj'] = torch.zeros_like(layer_sum)
            usage_sums['model.state_proj'].add_(layer_sum)


def _normalized_entropy(weights: torch.Tensor) -> torch.Tensor:
    if weights.numel() <= 1:
        return weights.new_ones(())
    entropy = -(weights * weights.clamp_min(1e-12).log()).sum()
    return entropy / torch.log(weights.new_tensor(float(weights.numel())))


def _stabilize_router_aggregation_weights(
    weights: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if weights.numel() <= 1:
        return torch.ones_like(weights)

    temperature = max(float(temperature), 1e-4)
    return torch.softmax(weights.clamp_min(1e-12).log() / temperature, dim=0)


def finalize_router_expert_usage(
    usage_sums: dict[str, torch.Tensor],
    device: torch.device,
    temperature: float = 0.5,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    normalized = {}
    raw_normalized = {}
    for name, usage in usage_sums.items():
        usage = usage.to(device=device, dtype=torch.float32)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(usage, op=dist.ReduceOp.SUM)
        total = usage.sum().clamp_min(1e-12)
        raw_weights = (usage / total).detach().cpu()
        final_weights = _stabilize_router_aggregation_weights(
            raw_weights,
            temperature=temperature,
        )
        raw_normalized[name] = raw_weights
        normalized[name] = final_weights
    return normalized, raw_normalized


def summarize_router_aggregation_usage(
    usage_sums: dict[str, torch.Tensor],
    num_experts: int,
    device: torch.device,
    temperature: float,
) -> tuple[int, torch.Tensor | None]:
    total_usage = torch.zeros(num_experts, device=device, dtype=torch.float32)
    layer_count = torch.zeros((), device=device, dtype=torch.float32)
    for usage in usage_sums.values():
        if usage.numel() != num_experts:
            continue
        total_usage.add_(usage.to(device=device, dtype=torch.float32))
        layer_count.add_(1.0)
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(total_usage, op=dist.ReduceOp.SUM)
        dist.all_reduce(layer_count, op=dist.ReduceOp.SUM)
        layer_count.div_(dist.get_world_size())
    if float(total_usage.sum()) <= 0.0:
        return int(layer_count.item()), None
    raw_weights = total_usage / total_usage.sum()
    weights = _stabilize_router_aggregation_weights(
        raw_weights,
        temperature=temperature,
    )
    return int(layer_count.item()), weights.detach().cpu()


def mean_router_aggregation_weights(
    expert_weights: dict[str, torch.Tensor] | None,
    num_experts: int,
) -> tuple[int, torch.Tensor | None]:
    if not expert_weights:
        return 0, None
    vectors = [
        weights.detach().float().cpu().reshape(-1)
        for weights in expert_weights.values()
        if weights.numel() == num_experts
    ]
    if not vectors:
        return 0, None
    return len(vectors), torch.stack(vectors).mean(dim=0)


def _unwrap_model_for_fedforesight(policy):
    if hasattr(policy, 'module'):
        policy = policy.module
    if hasattr(policy, '_orig_mod'):
        policy = policy._orig_mod
    return policy


def accumulate_pcea_scores(
    policy,
    score_sums: dict[str, torch.Tensor],
    sample_counts: dict[str, int],
) -> None:
    target_policy = _unwrap_model_for_fedforesight(policy)
    score_getter = getattr(target_policy, 'get_last_pcea_scores', None)
    count_getter = getattr(target_policy, 'get_last_pcea_counts', None)
    if score_getter is None or count_getter is None:
        return
    counts = count_getter()
    for name, scores in score_getter().items():
        count = int(counts.get(name, 0))
        if count <= 0:
            continue
        scores = scores.detach().float().cpu().reshape(-1)
        if name not in score_sums:
            score_sums[name] = torch.zeros_like(scores)
        score_sums[name].add_(scores)
        sample_counts[name] = sample_counts.get(name, 0) + count


def clear_fedforesight_caches(policy) -> None:
    target_policy = _unwrap_model_for_fedforesight(policy)
    clearer = getattr(target_policy, 'clear_fedforesight_router_caches', None)
    if clearer is not None:
        clearer()


def finalize_pcea_scores(
    score_sums: dict[str, torch.Tensor],
    sample_counts: dict[str, int],
    num_experts: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    if not score_sums:
        raise RuntimeError("PCEA is enabled but no path-consensus scores were collected")

    if dist.is_initialized() and dist.get_world_size() > 1:
        gathered_keys = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_keys, sorted(score_sums.keys()))
        all_keys = sorted({key for rank_keys in gathered_keys for key in (rank_keys or [])})
    else:
        all_keys = sorted(score_sums.keys())

    if not all_keys:
        raise RuntimeError("PCEA is enabled but distributed score synchronization produced no module keys")

    normalized = {}
    mean_agreements = {}
    for name in all_keys:
        scores = score_sums.get(name)
        if scores is None:
            scores = torch.zeros(num_experts, dtype=torch.float32)
        scores = scores.to(device=device, dtype=torch.float32).reshape(-1)
        if scores.numel() != num_experts:
            raise RuntimeError(
                f"PCEA score shape mismatch for {name}: expected {num_experts}, got {scores.numel()}"
            )
            continue
        count = torch.tensor(float(sample_counts.get(name, 0)), device=device, dtype=torch.float32)
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(scores, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        total_score = scores.sum()
        if (
            float(count.item()) <= 0.0
            or float(total_score.item()) <= 0.0
            or not bool(torch.isfinite(scores).all())
        ):
            raise RuntimeError(
                f"PCEA received invalid consensus evidence for {name}: "
                f"count={float(count.item())}, total_score={float(total_score.item())}"
            )
        weights, mean_agreement = path_consensus_expert_weights(scores, count)
        normalized[name] = weights.detach().cpu()
        mean_agreements[name] = float(mean_agreement.item())

    if not normalized:
        raise RuntimeError("PCEA did not produce any expert aggregation weights")
    return normalized, mean_agreements


def get_pcea_eligible_module_names(policy) -> set[str]:
    """Return U/G/A LoRA-MoE modules that must receive PCEA weights."""
    target_model = _unwrap_policy_model(policy)
    eligible = set()
    for name, module in target_model.named_modules():
        if not any(marker in name for marker in ("und_expert.", "gen_expert.", "act_expert.")):
            continue
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue
        expert_keys = [
            key
            for key in module.lora_A
            if str(key).isdigit() and key in module.lora_B
        ]
        if len(expert_keys) > 1:
            eligible.add(name)
    return eligible


def clear_router_runtime_caches(policy) -> None:
    """Drop non-parameter runtime tensors that can keep CUDA memory alive across phases."""
    target_model = _unwrap_policy_model(policy)

    model_attrs = [
        "_last_state_proj_router_probs",
        "_state_proj_router_observables",
        "_last_phi_features",
        "state_proj_aux_loss",
    ]
    for attr in model_attrs:
        if hasattr(target_model, attr):
            setattr(target_model, attr, None)

    module_attrs = [
        "_last_router_observables",
        "aux_loss",
    ]
    for module in target_model.modules():
        for attr in module_attrs:
            if hasattr(module, attr):
                setattr(module, attr, None)


def sync_router_gradients(model):
    """
    Synchronize router parameter gradients across all ranks (for distributed MoE training).

    Call after the backward pass of the load-balancing loss to ensure every rank
    has consistent router gradients.
    """
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return

    world_size = dist.get_world_size()

    # Get the inner model (handle QwenA1Policy wrapping)
    if hasattr(model, 'model'):
        target_model = model.model
    else:
        target_model = model

    # PEFT LoRA-MoE: lora_A/lora_B ModuleDict
    for module in target_model.modules():
        if hasattr(module, 'lora_router') and module.lora_router is not None:
            if module.lora_router.grad is not None:
                dist.all_reduce(module.lora_router.grad, op=dist.ReduceOp.SUM)
                module.lora_router.grad /= world_size

    # Also sync state_proj_router gradients
    if hasattr(target_model, 'state_proj_router') and target_model.state_proj_router.grad is not None:
        dist.all_reduce(target_model.state_proj_router.grad, op=dist.ReduceOp.SUM)
        target_model.state_proj_router.grad /= world_size


def run_moe_training(policy, dl_iter, optimizer, cfg, fl_cfg, lr_scheduler=None, local_rank=0, wandb_logger=None, global_start_step=0):
    """
    Run MoE training (router + expert weights) as a unique client.

    This trains the router network and expert weights in MoE mode,
    using a separate optimizer for MoE-specific parameters.
    """
    import logging

    logger = logging.getLogger()

    policy.train()

    moe_steps = fl_cfg.get("moe_steps", 50)
    batch_size = cfg.batch_size
    device = next(policy.parameters()).device
    grad_clip_norm = cfg.optimizer.grad_clip_norm
    load_balance_weight = fl_cfg.get("moe_load_balance_weight", 0.1)
    debug_freq = fl_cfg.get("moe_debug_freq", 10)
    lead_in_fraction = float(getattr(cfg.policy, "affordance_router_lead_in_fraction", 0.05))
    lead_in_steps = compute_router_lead_in_steps(
        moe_steps,
        lead_in_fraction,
        enabled=bool(getattr(cfg.policy, "enable_affordance", False)),
    )
    if lead_in_steps > 0:
        logger.info(
            f"[AffordanceRouterLeadIn] Holding LoRA experts fixed for {lead_in_steps}/{moe_steps} "
            "server steps; routers, action head, and affordance parameters remain trainable"
        )

    # Memory debug helper
    def log_memory(prefix=""):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
            logger.info(f"[Memory {prefix}] Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB, Max: {max_allocated:.2f}GB")

    # Log initial memory
    torch.cuda.reset_peak_memory_stats(device)
    log_memory("init")

    # Metrics aligned with lerobot_fl_robotwin
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "loss_action": AverageMeter("loss_action", ":.3f"),
        "loss_gen": AverageMeter("loss_gen", ":.3f"),
        "loss_aux": AverageMeter("loss_lb", ":.3f"),
        "loss_aux_weighted": AverageMeter("loss_lb_w", ":.3f"),
        "loss_affordance": AverageMeter("loss_aff", ":.3f"),
        "loss_affordance_weighted": AverageMeter("loss_aff_w", ":.3f"),
        "loss_affordance_action_align": AverageMeter("loss_aff_act", ":.3f"),
        "affordance_target_valid": AverageMeter("aff_valid", ":.3f"),
        "affordance_spatial_entropy": AverageMeter("aff_sp_ent", ":.3f"),
        "affordance_spatial_peak": AverageMeter("aff_sp_peak", ":.3f"),
        "affordance_action_gate": AverageMeter("aff_act_gate", ":.3f"),
        "affordance_router_gate": AverageMeter("aff_route_gate", ":.3f"),
        "loss_fard": AverageMeter("loss_fard", ":.3f"),
        "loss_fard_weighted": AverageMeter("loss_fard_w", ":.3f"),
        "fard_confidence": AverageMeter("fard_conf", ":.3f"),
        "fard_ug_agreement": AverageMeter("fard_agree", ":.3f"),
        "fard_reliability": AverageMeter("fard_rel", ":.3f"),
        "fard_js_fg_a": AverageMeter("fard_js", ":.3f"),
        "loss_visual_token_prune_ot": AverageMeter("loss_ot", ":.3f"),
        "loss_visual_token_prune_ot_weighted": AverageMeter("loss_ot_w", ":.3f"),
        "visual_token_pruned_pct": AverageMeter("visual_pruned", ":.2f"),
        "visual_token_adapter_pruned_pct": AverageMeter("adapter_pruned", ":.2f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    metrics_history = []
    optimizer.zero_grad()
    log_memory("after_optimizer_zero")

    for step in range(moe_steps):
        start_time = time.perf_counter()
        dataloading_start = start_time

        try:
            # Before data loading
            log_memory(f"step{step}_start")

            batch = next(dl_iter)
            batch = move_batch_to_device(batch, device)

            # After data loading
            log_memory(f"step{step}_after_load")

            dataloading_time = time.perf_counter() - dataloading_start

            # Custom forward with load balancing loss for MoE
            policy.train()

            # Before forward
            log_memory(f"step{step}_before_forward")

            loss, output_dict = policy.forward(batch)

            # After forward
            log_memory(f"step{step}_after_forward")

            # Debug: print router selection statistics
            debug_router_stats(policy, step, debug_freq)

            # The load-balancing loss is now computed inside policy.forward() and merged into loss
            # No need to compute it separately outside

            # After compute loss
            log_memory(f"step{step}_after_compute_loss")

            # forward() already returns total_loss including the load-balancing loss
            # Just call backward
            t3 = time.perf_counter()
            loss.backward()
            t_backward = time.perf_counter() - t3

            lead_in_active = step < lead_in_steps
            if lead_in_active:
                cleared_tensors, cleared_numel = clear_loramoe_expert_gradients(policy.named_parameters())
                if step == 0:
                    if cleared_tensors == 0:
                        raise RuntimeError(
                            "Affordance router lead-in found no LoRA expert gradients to freeze"
                        )
                    logger.info(
                        f"[AffordanceRouterLeadIn] Cleared {cleared_tensors} expert gradients "
                        f"({cleared_numel:,} parameters) after backward"
                    )
            elif step == lead_in_steps and lead_in_steps > 0:
                logger.info("[AffordanceRouterLeadIn] Lead-in complete; restored joint router-expert training")

            # Gradient clipping
            if grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf"), error_if_nonfinite=False)
            grad_norm = grad_norm.item()

            # After gradient clip
            log_memory(f"step{step}_after_gradclip")

            optimizer.step()
            optimizer.zero_grad()

            # Note: PEFT LoRA-MoE doesn't need clear_moe_buffers() call
            # Buffers are managed automatically by PEFT

            if lr_scheduler is not None:
                lr_scheduler.step()

            update_time = time.perf_counter() - start_time

            # Update metrics (aligned with lerobot_fl_robotwin)
            train_metrics["loss"].update(loss.item())
            train_metrics["loss_action"].update(output_dict.get("loss_action", 0.0))
            train_metrics["loss_gen"].update(output_dict.get("loss_gen", 0.0))
            train_metrics["loss_aux"].update(output_dict.get("loss_aux", 0.0))
            train_metrics["loss_aux_weighted"].update(output_dict.get("loss_aux_weighted", 0.0))
            train_metrics["loss_affordance"].update(output_dict.get("loss_affordance", 0.0))
            train_metrics["loss_affordance_weighted"].update(
                output_dict.get("loss_affordance_weighted", 0.0)
            )
            train_metrics["loss_affordance_action_align"].update(
                output_dict.get("loss_affordance_action_align", 0.0)
            )
            train_metrics["affordance_target_valid"].update(
                output_dict.get("affordance_target_valid", 0.0)
            )
            train_metrics["affordance_spatial_entropy"].update(
                output_dict.get("affordance_spatial_entropy", 0.0)
            )
            train_metrics["affordance_spatial_peak"].update(
                output_dict.get("affordance_spatial_peak", 0.0)
            )
            train_metrics["affordance_action_gate"].update(
                output_dict.get("affordance_action_gate", 0.0)
            )
            train_metrics["affordance_router_gate"].update(
                output_dict.get("affordance_router_gate", 0.0)
            )
            train_metrics["loss_fard"].update(output_dict.get("loss_fard", 0.0))
            train_metrics["loss_fard_weighted"].update(output_dict.get("loss_fard_weighted", 0.0))
            train_metrics["fard_confidence"].update(output_dict.get("fard_confidence", 0.0))
            train_metrics["fard_ug_agreement"].update(output_dict.get("fard_ug_agreement", 0.0))
            train_metrics["fard_reliability"].update(output_dict.get("fard_reliability", 0.0))
            train_metrics["fard_js_fg_a"].update(output_dict.get("fard_js_fg_a", 0.0))
            train_metrics["loss_visual_token_prune_ot"].update(output_dict.get("loss_visual_token_prune_ot", 0.0))
            train_metrics["loss_visual_token_prune_ot_weighted"].update(
                output_dict.get("loss_visual_token_prune_ot_weighted", 0.0)
            )
            train_metrics["visual_token_pruned_pct"].update(output_dict.get("visual_token_pruned_pct", 0.0))
            train_metrics["visual_token_adapter_pruned_pct"].update(
                output_dict.get("visual_token_adapter_pruned_pct", 0.0)
            )
            train_metrics["grad_norm"].update(grad_norm)
            train_metrics["lr"].update(optimizer.param_groups[0]["lr"])
            train_metrics["update_s"].update(update_time)
            train_metrics["dataloading_s"].update(dataloading_time)

            metrics_history.append({
                "loss": loss.item(),
                "grad_norm": grad_norm,
                "lr": optimizer.param_groups[0]["lr"],
                "update_s": update_time,
                "loss_action": output_dict.get("loss_action"),
                "loss_gen": output_dict.get("loss_gen"),
                "loss_aux": output_dict.get("loss_aux", 0.0),
                "loss_aux_weighted": output_dict.get("loss_aux_weighted", 0.0),
                "loss_affordance": output_dict.get("loss_affordance", 0.0),
                "loss_affordance_weighted": output_dict.get("loss_affordance_weighted", 0.0),
                "loss_affordance_action_align": output_dict.get("loss_affordance_action_align", 0.0),
                "affordance_target_valid": output_dict.get("affordance_target_valid", 0.0),
                "affordance_spatial_entropy": output_dict.get("affordance_spatial_entropy", 0.0),
                "affordance_spatial_peak": output_dict.get("affordance_spatial_peak", 0.0),
                "affordance_action_gate": output_dict.get("affordance_action_gate", 0.0),
                "affordance_router_gate": output_dict.get("affordance_router_gate", 0.0),
                "loss_fard": output_dict.get("loss_fard", 0.0),
                "loss_fard_weighted": output_dict.get("loss_fard_weighted", 0.0),
                "fard_confidence": output_dict.get("fard_confidence", 0.0),
                "fard_ug_agreement": output_dict.get("fard_ug_agreement", 0.0),
                "fard_reliability": output_dict.get("fard_reliability", 0.0),
                "fard_js_fg_a": output_dict.get("fard_js_fg_a", 0.0),
                "loss_visual_token_prune_ot": output_dict.get("loss_visual_token_prune_ot", 0.0),
                "loss_visual_token_prune_ot_weighted": output_dict.get(
                    "loss_visual_token_prune_ot_weighted", 0.0
                ),
                "visual_token_pruned_pct": output_dict.get("visual_token_pruned_pct", 0.0),
                "visual_token_adapter_pruned_pct": output_dict.get("visual_token_adapter_pruned_pct", 0.0),
            })

            if step % fl_cfg.get("log_freq", 10) == 0:
                logger.info(
                    f"[Rank {local_rank}] MoE step {step}/{moe_steps}: "
                    f"loss={train_metrics['loss'].avg:.4f} "
                    f"loss_action={train_metrics['loss_action'].avg:.4f} "
                    f"phase={'router_lead_in' if lead_in_active else 'joint'} "
                    f"visual_pruned={train_metrics['visual_token_pruned_pct'].avg:.2f}% "
                    f"grdn={train_metrics['grad_norm'].avg:.2f}"
                )

            del batch, loss, output_dict

        except Exception as e:
            logger.warning(f"[Rank {local_rank}] MoE training error at step {step}: {e}")
            break

    # Compute average metrics
    avg_metrics = {
        "loss": train_metrics["loss"].avg,
        "loss_action": train_metrics["loss_action"].avg,
        "loss_gen": train_metrics["loss_gen"].avg,
        "loss_aux": train_metrics["loss_aux"].avg,
        "loss_aux_weighted": train_metrics["loss_aux_weighted"].avg,
        "loss_affordance": train_metrics["loss_affordance"].avg,
        "loss_affordance_weighted": train_metrics["loss_affordance_weighted"].avg,
        "loss_affordance_action_align": train_metrics["loss_affordance_action_align"].avg,
        "affordance_target_valid": train_metrics["affordance_target_valid"].avg,
        "affordance_spatial_entropy": train_metrics["affordance_spatial_entropy"].avg,
        "affordance_spatial_peak": train_metrics["affordance_spatial_peak"].avg,
        "affordance_action_gate": train_metrics["affordance_action_gate"].avg,
        "affordance_router_gate": train_metrics["affordance_router_gate"].avg,
        "loss_fard": train_metrics["loss_fard"].avg,
        "loss_fard_weighted": train_metrics["loss_fard_weighted"].avg,
        "fard_confidence": train_metrics["fard_confidence"].avg,
        "fard_ug_agreement": train_metrics["fard_ug_agreement"].avg,
        "fard_reliability": train_metrics["fard_reliability"].avg,
        "fard_js_fg_a": train_metrics["fard_js_fg_a"].avg,
        "loss_visual_token_prune_ot": train_metrics["loss_visual_token_prune_ot"].avg,
        "loss_visual_token_prune_ot_weighted": train_metrics[
            "loss_visual_token_prune_ot_weighted"
        ].avg,
        "visual_token_pruned_pct": train_metrics["visual_token_pruned_pct"].avg,
        "visual_token_adapter_pruned_pct": train_metrics["visual_token_adapter_pruned_pct"].avg,
        "grad_norm": train_metrics["grad_norm"].avg,
        "lr": train_metrics["lr"].avg,
        "router_lead_in_steps": lead_in_steps,
        "router_lead_in_fraction": lead_in_steps / max(1, moe_steps),
    }

    logger.info(
        f"[Rank {local_rank}] MoE training complete: "
        f"loss={avg_metrics['loss']:.4f} "
        f"loss_action={avg_metrics['loss_action']:.4f} "
        f"loss_aux={avg_metrics['loss_aux']:.4f} "
        f"visual_pruned={avg_metrics['visual_token_pruned_pct']:.2f}%"
    )

    return {"metrics": avg_metrics}


def run_moe_training_distributed(
    policy,
    dataset,  # Pass dataset directly, we will create per-rank data
    optimizer,
    cfg,
    fl_cfg,
    accelerator,
    lr_scheduler=None,
):
    """
    Run distributed MoE training using Accelerator for DDP.

    This trains the router network and expert weights in true distributed mode,
    where all GPUs collaboratively train a single MoE model using data parallelism.

    Args:
        policy: The MoE policy model (will be wrapped by accelerator)
        dataset: The dataset for MoE training (will be sharded across ranks)
        optimizer: Optimizer for MoE parameters
        cfg: Training configuration
        fl_cfg: FL configuration with MoE parameters
        accelerator: Accelerator instance for distributed training
        lr_scheduler: Optional learning rate scheduler

    Returns:
        Dictionary with average metrics
    """
    import logging
    from torch.utils.data import DistributedSampler
    from accelerate.utils import send_to_device
    from lerobot.datasets.utils import cycle

    logger = logging.getLogger()
    is_main_process = accelerator.is_main_process

    policy.train()

    moe_steps = fl_cfg.get("moe_steps", 50)
    moe_batch_size = fl_cfg.get("moe_batch_size", 1)  # Use independent moe_batch_size
    batch_size = moe_batch_size
    grad_clip_norm = cfg.optimizer.grad_clip_norm
    load_balance_weight = fl_cfg.get("moe_load_balance_weight", 0.1)
    debug_freq = fl_cfg.get("moe_debug_freq", 10)
    log_freq = fl_cfg.get("log_freq", 10)
    lead_in_fraction = float(getattr(cfg.policy, "affordance_router_lead_in_fraction", 0.05))
    lead_in_steps = compute_router_lead_in_steps(
        moe_steps,
        lead_in_fraction,
        enabled=bool(getattr(cfg.policy, "enable_affordance", False)),
    )
    if is_main_process and bool(getattr(cfg.policy, "enable_affordance", False)):
        logger.info(
            f"[AffordanceRouterLeadIn] fraction={lead_in_fraction:.3f}, "
            f"effective_steps={lead_in_steps}/{moe_steps}; routers, action head, and "
            "affordance parameters remain trainable"
        )

    # Create DistributedSampler for true data parallelism
    # Each rank gets a different shard of the data
    sampler = DistributedSampler(
        dataset,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
        drop_last=True,  # Prevent deadlock from uneven batch counts across ranks
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        shuffle=False,  # sampler handles shuffle
        pin_memory=True,
        drop_last=True,  # Ensure all ranks have same number of batches
        num_workers=cfg.num_workers if hasattr(cfg, 'num_workers') and cfg.num_workers > 0 else 4,
        prefetch_factor=2,
    )

    # Use cycle to create persistent iterator - prevents StopIteration and deadlocks
    dataloader_iter = iter(cycle(dataloader))

    # Note: The model is already wrapped with DDP outside this function
    # We don't call accelerator.prepare() here

    # Metrics
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "loss_action": AverageMeter("loss_action", ":.3f"),
        "loss_gen": AverageMeter("loss_gen", ":.3f"),
        "loss_aux": AverageMeter("loss_lb", ":.3f"),
        "loss_aux_weighted": AverageMeter("loss_lb_w", ":.3f"),
        "loss_affordance": AverageMeter("loss_aff", ":.3f"),
        "loss_affordance_weighted": AverageMeter("loss_aff_w", ":.3f"),
        "loss_affordance_action_align": AverageMeter("loss_aff_act", ":.3f"),
        "affordance_target_valid": AverageMeter("aff_valid", ":.3f"),
        "affordance_spatial_entropy": AverageMeter("aff_sp_ent", ":.3f"),
        "affordance_spatial_peak": AverageMeter("aff_sp_peak", ":.3f"),
        "affordance_action_gate": AverageMeter("aff_act_gate", ":.3f"),
        "affordance_router_gate": AverageMeter("aff_route_gate", ":.3f"),
        "loss_fard": AverageMeter("loss_fard", ":.3f"),
        "loss_fard_weighted": AverageMeter("loss_fard_w", ":.3f"),
        "fard_confidence": AverageMeter("fard_conf", ":.3f"),
        "fard_ug_agreement": AverageMeter("fard_agree", ":.3f"),
        "fard_reliability": AverageMeter("fard_rel", ":.3f"),
        "fard_js_fg_a": AverageMeter("fard_js", ":.3f"),
        "lambda_fard_effective": AverageMeter("fard_lam", ":.3e"),
        "loss_visual_token_prune_ot": AverageMeter("loss_ot", ":.3f"),
        "loss_visual_token_prune_ot_weighted": AverageMeter("loss_ot_w", ":.3f"),
        "visual_token_pruned_pct": AverageMeter("visual_pruned", ":.2f"),
        "visual_token_adapter_pruned_pct": AverageMeter("adapter_pruned", ":.2f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    metrics_history = []
    optimizer.zero_grad()

    device = accelerator.device
    pcea_enabled = bool(getattr(cfg.policy, 'enable_pcea', False))
    use_router_weighted_aggregation = bool(fl_cfg.get("moe_router_weighted_aggregation", False))
    router_agg_temperature = float(fl_cfg.get("moe_router_agg_temperature", 0.5))
    if use_router_weighted_aggregation and router_agg_temperature <= 0:
        raise ValueError(
            f"MOE_ROUTER_AGG_TEMPERATURE must be positive, got {router_agg_temperature}"
        )
    if use_router_weighted_aggregation and pcea_enabled:
        raise ValueError(
            "Router-weighted aggregation and PCEA are mutually exclusive; disable ENABLE_PCEA"
        )
    if is_main_process and use_router_weighted_aggregation:
        logger.info(
            f"[RouterAgg] Enabled usage-weighted complete-LoRA aggregation with "
            f"temperature={router_agg_temperature:.3f}; usage accumulation starts at "
            f"joint step {lead_in_steps}/{moe_steps}"
        )
    ab_routing_enabled = bool(getattr(cfg.policy, 'loramoe_ab_routing', False))
    if use_router_weighted_aggregation and ab_routing_enabled:
        if is_main_process:
            logger.warning("[RouterAgg] AB routing is enabled; router-weighted aggregation is disabled for this run")
        use_router_weighted_aggregation = False
    router_usage_sums = {} if use_router_weighted_aggregation else None
    router_usage_steps = 0
    pcea_score_sums = {} if pcea_enabled else None
    pcea_sample_counts = {} if pcea_score_sums is not None else None

    # Enable MoE profiling if requested
    moe_profiling = fl_cfg.get("moe_profiling", False)
    if moe_profiling:
        logger.info("[MoE] Enabling profiling in PEFT LoRA-MoE layers")
        from peft.tuners.lora.layer import set_loramoe_config
        set_loramoe_config(profiling_enabled=True)

    for step in range(moe_steps):
        start_time = time.perf_counter()
        lead_in_active = step < lead_in_steps

        # ========== Detailed per-phase timing ==========
        t_dataloader = 0
        t_forward = 0
        t_lb_loss = 0
        t_backward = 0
        t_sync_router = 0
        t_grad_clip = 0
        t_optimizer = 0
        # =================================

        dataloading_start = start_time

        # No need to set_epoch when using cycle iterator
        # sampler.set_epoch(step)

        try:
            # Use persistent iterator - prevents deadlock from re-creating iter
            t0 = time.perf_counter()
            batch = next(dataloader_iter)
            batch = send_to_device(batch, device, non_blocking=True)
            t_dataloader = time.perf_counter() - t0

            # Forward pass
            # Note: the load-balancing loss is now computed inside policy.forward() and merged into loss
            # - MoE mode: forward() computes the load-balancing loss internally
            # - LoRA mode: forward() does not compute the load-balancing loss
            t1 = time.perf_counter()
            policy.train()
            # Use autocast for mixed-precision training, consistent with lerobot_train.py
            with accelerator.autocast():
                loss, output_dict = policy.forward(batch)
            t_forward = time.perf_counter() - t1
            if pcea_score_sums is not None:
                accumulate_pcea_scores(policy, pcea_score_sums, pcea_sample_counts)
            if router_usage_sums is not None and should_accumulate_router_usage(step, lead_in_steps):
                accumulate_router_expert_usage(policy, router_usage_sums, batch)
                router_usage_steps += 1

            # =====================================================================
            # =====================================================================

            # No need to compute it separately outside

            # Backward: a single backward pass
            t3 = time.perf_counter()
            accelerator.backward(loss)
            t_backward = time.perf_counter() - t3
            if lead_in_active:
                cleared_tensors, cleared_numel = clear_loramoe_expert_gradients(policy.named_parameters())
                if step == 0 and cleared_tensors == 0:
                    raise RuntimeError(
                        "Affordance router lead-in found no LoRA expert gradients to freeze"
                    )
                if is_main_process and step == 0:
                    logger.info(
                        f"[AffordanceRouterLeadIn] Cleared {cleared_tensors} expert gradients "
                        f"({cleared_numel:,} parameters) after backward"
                    )
            elif is_main_process and step == lead_in_steps and lead_in_steps > 0:
                logger.info("[AffordanceRouterLeadIn] Lead-in complete; restored joint router-expert training")
            if is_main_process and step % 20 == 0:
                # Check if gradients are being computed
                grad_count = sum(1 for p in policy.parameters() if p.grad is not None)
                logger.info(f"[Backward-Debug] step={step} backward_time={t_backward:.3f}s grads_computed={grad_count}")
            # Note: manual sync_router_gradients removed; DDP handles it automatically
            # sync_router_gradients(policy)
            t_sync_router = 0  # Manual synchronization no longer needed

            # Gradient clipping
            t5 = time.perf_counter()
            if grad_clip_norm > 0:
                grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), float("inf"), error_if_nonfinite=False
                )
            grad_norm = grad_norm.item()
            t_grad_clip = time.perf_counter() - t5

            # Optimizer step
            t6 = time.perf_counter()
            optimizer.step()
            optimizer.zero_grad()
            t_optimizer = time.perf_counter() - t6

            # ========== MoE expert distribution monitoring ==========
            debug_freq = fl_cfg.get("moe_debug_freq", 10)
            if is_main_process and step % debug_freq == 0:
                debug_router_stats(policy, step, debug_freq)
            # ===========================================

            # Note: PEFT LoRA-MoE doesn't need clear_moe_buffers() call
            # Buffers are managed automatically by PEFT

            if lr_scheduler is not None:
                lr_scheduler.step()

            clear_fedforesight_caches(policy)

            update_time = time.perf_counter() - start_time

            # ========== Detailed log output ==========
            if is_main_process and step % 10 == 0:
                logger.info(
                    f"[MoE-Timing] Step {step}: "
                    f"data={t_dataloader:.3f}s "
                    f"forward={t_forward:.3f}s "
                    f"lb_loss={t_lb_loss:.3f}s "
                    f"backward={t_backward:.3f}s "
                    f"sync_router={t_sync_router:.3f}s "
                    f"grad_clip={t_grad_clip:.3f}s "
                    f"optimizer={t_optimizer:.3f}s "
                    f"total={update_time:.3f}s"
                )
            # ==================================

            # Update metrics
            train_metrics["loss"].update(loss.item())
            train_metrics["loss_action"].update(output_dict.get("loss_action", 0.0))
            train_metrics["loss_gen"].update(output_dict.get("loss_gen", 0.0))
            train_metrics["loss_aux"].update(output_dict.get("loss_aux", 0.0))
            train_metrics["loss_aux_weighted"].update(output_dict.get("loss_aux_weighted", 0.0))
            train_metrics["loss_affordance"].update(output_dict.get("loss_affordance", 0.0))
            train_metrics["loss_affordance_weighted"].update(
                output_dict.get("loss_affordance_weighted", 0.0)
            )
            train_metrics["loss_affordance_action_align"].update(
                output_dict.get("loss_affordance_action_align", 0.0)
            )
            train_metrics["affordance_target_valid"].update(
                output_dict.get("affordance_target_valid", 0.0)
            )
            train_metrics["affordance_spatial_entropy"].update(
                output_dict.get("affordance_spatial_entropy", 0.0)
            )
            train_metrics["affordance_spatial_peak"].update(
                output_dict.get("affordance_spatial_peak", 0.0)
            )
            train_metrics["affordance_action_gate"].update(
                output_dict.get("affordance_action_gate", 0.0)
            )
            train_metrics["affordance_router_gate"].update(
                output_dict.get("affordance_router_gate", 0.0)
            )
            train_metrics["loss_fard"].update(output_dict.get("loss_fard", 0.0))
            train_metrics["loss_fard_weighted"].update(output_dict.get("loss_fard_weighted", 0.0))
            train_metrics["fard_confidence"].update(output_dict.get("fard_confidence", 0.0))
            train_metrics["fard_ug_agreement"].update(output_dict.get("fard_ug_agreement", 0.0))
            train_metrics["fard_reliability"].update(output_dict.get("fard_reliability", 0.0))
            train_metrics["fard_js_fg_a"].update(output_dict.get("fard_js_fg_a", 0.0))
            train_metrics["lambda_fard_effective"].update(output_dict.get("lambda_fard_effective", 0.0))
            train_metrics["loss_visual_token_prune_ot"].update(output_dict.get("loss_visual_token_prune_ot", 0.0))
            train_metrics["loss_visual_token_prune_ot_weighted"].update(
                output_dict.get("loss_visual_token_prune_ot_weighted", 0.0)
            )
            train_metrics["visual_token_pruned_pct"].update(output_dict.get("visual_token_pruned_pct", 0.0))
            train_metrics["visual_token_adapter_pruned_pct"].update(
                output_dict.get("visual_token_adapter_pruned_pct", 0.0)
            )
            train_metrics["grad_norm"].update(grad_norm)
            train_metrics["lr"].update(optimizer.param_groups[0]["lr"])
            train_metrics["update_s"].update(update_time)
            train_metrics["dataloading_s"].update(t_dataloader)

            metrics_history.append({
                "loss": loss.item(),
                "grad_norm": grad_norm,
                "lr": optimizer.param_groups[0]["lr"],
                "update_s": update_time,
                "loss_action": output_dict.get("loss_action"),
                "loss_gen": output_dict.get("loss_gen"),
                "loss_aux": output_dict.get("loss_aux", 0.0),
                "loss_aux_weighted": output_dict.get("loss_aux_weighted", 0.0),
                "loss_affordance": output_dict.get("loss_affordance", 0.0),
                "loss_affordance_weighted": output_dict.get("loss_affordance_weighted", 0.0),
                "loss_affordance_action_align": output_dict.get("loss_affordance_action_align", 0.0),
                "affordance_target_valid": output_dict.get("affordance_target_valid", 0.0),
                "affordance_spatial_entropy": output_dict.get("affordance_spatial_entropy", 0.0),
                "affordance_spatial_peak": output_dict.get("affordance_spatial_peak", 0.0),
                "affordance_action_gate": output_dict.get("affordance_action_gate", 0.0),
                "affordance_router_gate": output_dict.get("affordance_router_gate", 0.0),
                "loss_fard": output_dict.get("loss_fard", 0.0),
                "loss_fard_weighted": output_dict.get("loss_fard_weighted", 0.0),
                "fard_confidence": output_dict.get("fard_confidence", 0.0),
                "fard_ug_agreement": output_dict.get("fard_ug_agreement", 0.0),
                "fard_reliability": output_dict.get("fard_reliability", 0.0),
                "fard_js_fg_a": output_dict.get("fard_js_fg_a", 0.0),
                "lambda_fard_effective": output_dict.get("lambda_fard_effective", 0.0),
                "loss_visual_token_prune_ot": output_dict.get("loss_visual_token_prune_ot", 0.0),
                "loss_visual_token_prune_ot_weighted": output_dict.get(
                    "loss_visual_token_prune_ot_weighted", 0.0
                ),
                "visual_token_pruned_pct": output_dict.get("visual_token_pruned_pct", 0.0),
                "visual_token_adapter_pruned_pct": output_dict.get("visual_token_adapter_pruned_pct", 0.0),
                "router_lead_in_active": float(lead_in_active),
            })

            agg_weight_summary = None
            if router_usage_sums is not None and step % log_freq == 0:
                _, agg_weight_summary = summarize_router_aggregation_usage(
                    router_usage_sums,
                    num_experts=int(cfg.policy.loramoe_num_experts),
                    device=device,
                    temperature=router_agg_temperature,
                )

            # Only log on main process
            if is_main_process and step % log_freq == 0:
                agg_weights_text = (
                    [round(float(value), 4) for value in agg_weight_summary.tolist()]
                    if agg_weight_summary is not None
                    else []
                )
                logger.info(
                    f"[MoE-TrainSummary] step={step}/{moe_steps} "
                    f"loss={train_metrics['loss'].avg:.4f} "
                    f"loss_action={train_metrics['loss_action'].avg:.4f} "
                    f"loss_fard={train_metrics['loss_fard'].avg:.4f} "
                    f"phase={'router_lead_in' if lead_in_active else 'joint'} "
                    f"visual_pruned={train_metrics['visual_token_pruned_pct'].avg:.2f}% "
                    f"agg_weight_summary={agg_weights_text} "
                    f"grdn={train_metrics['grad_norm'].avg:.2f}"
                )

            del batch, loss, output_dict

        except Exception as e:
            logger.exception(f"[MoE-Dist] Training error at step {step}; aborting the round: {e}")
            raise

        # Wait for all processes only at logging steps (not every step)
        if step % log_freq == 0:
            accelerator.wait_for_everyone()

    # Compute average metrics (all ranks have same metrics due to sync)
    avg_metrics = {
        "loss": train_metrics["loss"].avg,
        "loss_action": train_metrics["loss_action"].avg,
        "loss_gen": train_metrics["loss_gen"].avg,
        "loss_aux": train_metrics["loss_aux"].avg,
        "loss_aux_weighted": train_metrics["loss_aux_weighted"].avg,
        "loss_affordance": train_metrics["loss_affordance"].avg,
        "loss_affordance_weighted": train_metrics["loss_affordance_weighted"].avg,
        "loss_affordance_action_align": train_metrics["loss_affordance_action_align"].avg,
        "affordance_target_valid": train_metrics["affordance_target_valid"].avg,
        "affordance_spatial_entropy": train_metrics["affordance_spatial_entropy"].avg,
        "affordance_spatial_peak": train_metrics["affordance_spatial_peak"].avg,
        "affordance_action_gate": train_metrics["affordance_action_gate"].avg,
        "affordance_router_gate": train_metrics["affordance_router_gate"].avg,
        "loss_fard": train_metrics["loss_fard"].avg,
        "loss_fard_weighted": train_metrics["loss_fard_weighted"].avg,
        "fard_confidence": train_metrics["fard_confidence"].avg,
        "fard_ug_agreement": train_metrics["fard_ug_agreement"].avg,
        "fard_reliability": train_metrics["fard_reliability"].avg,
        "fard_js_fg_a": train_metrics["fard_js_fg_a"].avg,
        "lambda_fard_effective": train_metrics["lambda_fard_effective"].avg,
        "loss_visual_token_prune_ot": train_metrics["loss_visual_token_prune_ot"].avg,
        "loss_visual_token_prune_ot_weighted": train_metrics[
            "loss_visual_token_prune_ot_weighted"
        ].avg,
        "visual_token_pruned_pct": train_metrics["visual_token_pruned_pct"].avg,
        "visual_token_adapter_pruned_pct": train_metrics["visual_token_adapter_pruned_pct"].avg,
        "grad_norm": train_metrics["grad_norm"].avg,
        "lr": train_metrics["lr"].avg,
        "router_lead_in_steps": lead_in_steps,
        "router_lead_in_fraction": lead_in_steps / max(1, moe_steps),
    }
    router_expert_weights = None
    if router_usage_sums is not None:
        avg_metrics.update({
            "router_agg_usage_start_step": lead_in_steps,
            "router_agg_usage_steps": router_usage_steps,
        })
        router_expert_weights, raw_router_expert_weights = finalize_router_expert_usage(
            router_usage_sums,
            device,
            temperature=router_agg_temperature,
        )
        if router_expert_weights:
            raw_entropies = [float(_normalized_entropy(weights)) for weights in raw_router_expert_weights.values()]
            final_entropies = [float(_normalized_entropy(weights)) for weights in router_expert_weights.values()]
            raw_uniform_l1 = []
            final_uniform_l1 = []
            for name, weights in router_expert_weights.items():
                uniform = torch.full_like(weights, 1.0 / float(weights.numel()))
                raw_uniform_l1.append(float((raw_router_expert_weights[name] - uniform).abs().sum()))
                final_uniform_l1.append(float((weights - uniform).abs().sum()))
            _, mean_router_weights = mean_router_aggregation_weights(
                router_expert_weights,
                int(cfg.policy.loramoe_num_experts),
            )
            avg_metrics.update({
                "router_agg_temperature": router_agg_temperature,
                "router_agg_raw_entropy": sum(raw_entropies) / len(raw_entropies),
                "router_agg_final_entropy": sum(final_entropies) / len(final_entropies),
                "router_agg_raw_uniform_l1": sum(raw_uniform_l1) / len(raw_uniform_l1),
                "router_agg_final_uniform_l1": sum(final_uniform_l1) / len(final_uniform_l1),
                "router_agg_mean_max": float(mean_router_weights.max()) if mean_router_weights is not None else 0.0,
                "router_agg_mean_min": float(mean_router_weights.min()) if mean_router_weights is not None else 0.0,
            })
        if is_main_process and router_expert_weights:
            logger.info(
                f"[RouterAgg] usage_window=[{lead_in_steps}, {moe_steps}), "
                f"accumulated_steps={router_usage_steps}; mean weight sharpness: "
                f"entropy={avg_metrics['router_agg_raw_entropy']:.4f}->"
                f"{avg_metrics['router_agg_final_entropy']:.4f}, "
                f"uniform_l1={avg_metrics['router_agg_raw_uniform_l1']:.4f}->"
                f"{avg_metrics['router_agg_final_uniform_l1']:.4f}, "
                f"mean_max={avg_metrics['router_agg_mean_max']:.4f}, "
                f"mean_min={avg_metrics['router_agg_mean_min']:.4f}"
            )
            for layer_name, weights in list(router_expert_weights.items())[:3]:
                raw_weights = raw_router_expert_weights[layer_name]
                logger.info(
                    f"[RouterAgg] {layer_name}: "
                    f"raw={[round(float(v), 4) for v in raw_weights.tolist()]} "
                    f"final={[round(float(v), 4) for v in weights.tolist()]} "
                    f"l1_shift={float((weights - raw_weights).abs().sum()):.4f} "
                    f"raw_entropy={float(_normalized_entropy(raw_weights)):.4f} "
                    f"final_entropy={float(_normalized_entropy(weights)):.4f}"
                )

    pcea_expert_weights = None
    pcea_mean_agreements = {}
    if pcea_score_sums is not None:
        pcea_expert_weights, pcea_mean_agreements = finalize_pcea_scores(
            pcea_score_sums,
            pcea_sample_counts,
            num_experts=int(cfg.policy.loramoe_num_experts),
            device=device,
        )
        if is_main_process and pcea_expert_weights:
            logger.info(f"[PCEA] Finalized path-consensus expert weights for {len(pcea_expert_weights)} modules")
            for layer_name, weights in list(pcea_expert_weights.items())[:3]:
                uniform = torch.full_like(weights, 1.0 / float(weights.numel()))
                logger.info(
                    f"[PCEA] {layer_name}: agreement={pcea_mean_agreements[layer_name]:.4f} "
                    f"entropy={float(_normalized_entropy(weights)):.4f} "
                    f"uniform_l1={float((weights - uniform).abs().sum()):.4f} "
                    f"weights={[round(float(v), 4) for v in weights.tolist()]}"
                )
        if pcea_mean_agreements:
            avg_metrics["pcea_mean_agreement"] = sum(pcea_mean_agreements.values()) / len(pcea_mean_agreements)
            pcea_entropies = [float(_normalized_entropy(weights)) for weights in pcea_expert_weights.values()]
            pcea_uniform_l1 = []
            for weights in pcea_expert_weights.values():
                uniform = torch.full_like(weights, 1.0 / float(weights.numel()))
                pcea_uniform_l1.append(float((weights - uniform).abs().sum()))
            avg_metrics["pcea_weight_entropy"] = sum(pcea_entropies) / len(pcea_entropies)
            avg_metrics["pcea_uniform_l1"] = sum(pcea_uniform_l1) / len(pcea_uniform_l1)

    if is_main_process:
        aggregation_weights_for_summary = pcea_expert_weights or router_expert_weights
        _, final_agg_weights = mean_router_aggregation_weights(
            aggregation_weights_for_summary,
            num_experts=int(cfg.policy.loramoe_num_experts),
        )
        final_agg_weights_text = (
            [round(float(value), 4) for value in final_agg_weights.tolist()]
            if final_agg_weights is not None
            else []
        )
        logger.info(
            f"[MoE-TrainSummary] complete "
            f"loss={avg_metrics['loss']:.4f} "
            f"loss_action={avg_metrics['loss_action']:.4f} "
            f"loss_aux={avg_metrics['loss_aux']:.4f} "
            f"loss_fard={avg_metrics['loss_fard']:.4f} "
            f"visual_pruned={avg_metrics['visual_token_pruned_pct']:.2f}% "
            f"agg_final_mean={final_agg_weights_text}"
        )

        # Print MoE profiling summary at the end
        if moe_profiling:
            from peft.tuners.lora.layer import get_moe_profiling_stats
            stats = get_moe_profiling_stats(policy)
            total = stats["total"]
            if total["calls"] > 0:
                logger.info("=" * 80)
                logger.info("[MoE-Prof-Summary] Aggregated over all steps:")
                logger.info(f"  Total LoRA-MoE layers: {len(stats['per_layer_type'])}")
                logger.info(f"  Total calls: {total['calls']}")
                logger.info(f"  Total MoE time: {total['total']:.3f}s")
                logger.info(f"    - base_layer:  {total['base']:.3f}s ({100*total['base']/total['total']:.1f}%)")
                logger.info(f"    - router:      {total['router']:.3f}s ({100*total['router']/total['total']:.1f}%)")
                logger.info(f"    - expert_calc: {total['expert']:.3f}s ({100*total['expert']/total['total']:.1f}%)")
                logger.info(f"    - gather:       {total['gather']:.3f}s ({100*total['gather']/total['total']:.1f}%)")
                logger.info(f"  Avg per call: {total['total']/total['calls']*1000:.2f}ms")
                logger.info("=" * 80)

    # Note: After DDP training, model parameters are automatically synchronized across ranks
    # No need for manual sync_all_experts_allreduce

    clear_router_runtime_caches(policy)

    return {
        "metrics": avg_metrics,
        "router_expert_weights": router_expert_weights,
        "pcea_expert_weights": pcea_expert_weights,
        "pcea_mean_agreements": pcea_mean_agreements,
    }


def validate_fedavg_server_config(cfg, fl_cfg, world_size):
    """Preflight the standard-LoRA flow without imposing MoE router constraints."""
    if cfg.resume:
        raise ValueError("lora_fedavg_server does not support federated resume")
    if not world_size == fl_cfg["num_gpus"] == fl_cfg["num_clients"]:
        raise ValueError("lora_fedavg_server requires WORLD_SIZE == NUM_GPUS == FL_NUM_CLIENTS")
    if not cfg.policy.use_lora or cfg.policy.use_lora_moe:
        raise ValueError("lora_fedavg_server requires use_lora=True, use_lora_moe=False")
    incompatible = (
        "use_lora_moe_forced_last", "loramoe_share_a_across_experts", "loramoe_ab_routing",
        "enable_fard", "enable_pcea", "enable_affordance", "enable_affordance_v2",
        "enable_affordance_v3", "enable_tcr", "tcr_enable_loss", "use_visual_token_prune",
        "use_ptq", "train_expert_only", "train_vlm_only", "compile_model",
    )
    enabled = [name for name in incompatible if getattr(cfg.policy, name, False)]
    enabled += [name for name in ("ENABLE_FARD", "ENABLE_PCEA", "MOE_ROUTER_WEIGHTED_AGGREGATION")
                if os.environ.get(name, "false").lower() == "true"]
    if fl_cfg["moe_router_weighted_aggregation"]:
        enabled.append("moe_router_weighted_aggregation")
    if os.environ.get("PEFT_TCR_CACHE_G", "0") not in ("0", "false", "False", ""):
        enabled.append("PEFT_TCR_CACHE_G")
    if enabled:
        raise ValueError(f"lora_fedavg_server incompatible flags: {enabled}")
    if fl_cfg["aggregation_strategy"] not in ("fedavg", "fedavg_uniform"):
        raise ValueError("lora_fedavg_server supports only equal-client FedAvg")
    if not cfg.policy.pretrained_path:
        raise ValueError("lora_fedavg_server requires a pretrained base (or local copy)")
    for name in ("num_rounds", "local_steps", "local_epochs", "moe_steps", "moe_batch_size", "save_freq"):
        if fl_cfg[name] <= 0:
            raise ValueError(f"lora_fedavg_server requires positive {name}")
    if (cfg.dataset.streaming or cfg.dataset.dist_loading or cfg.dataset.episodes is not None
            or len(cfg.dataset.repo_id.split()) != 1):
        raise ValueError("lora_fedavg_server requires one complete map-style dataset on every rank")
    if fl_cfg["partition_strategy"] == "mole_by_task" and fl_cfg["num_clients"] not in (4, 8):
        raise ValueError("MoLe lora_fedavg_server requires four or eight clients")


def validate_fedavg_server_upload(policy, state):
    """Require exactly the trainable pool, including complete, trainable standard A/B."""
    params = dict(policy.named_parameters())
    expected = {name for name, param in params.items() if param.requires_grad}
    if set(state) != expected:
        raise ValueError(f"FedAvg upload keys differ from trainable keys: "
                         f"missing={sorted(expected - set(state))}, extra={sorted(set(state) - expected)}")
    factors = [name for name in params if "lora_A" in name or "lora_B" in name]
    if not factors or any(name not in expected for name in factors):
        raise ValueError("FedAvg requires all standard LoRA A/B factors to be trainable")
    for name in expected:
        is_factor = name in factors
        is_action_weight = any(part in name for part in (
            "action_in_proj", "action_out_proj", "action_time_mlp"
        )) and name.endswith(".weight")
        if not (is_factor or is_action_weight) or "router" in name or "_moe" in name:
            raise ValueError(f"Not a standard LoRA/action-head trainable parameter: {name}")
        for factor in ("lora_A", "lora_B"):
            if f".{factor}." in name and not name.endswith(f".{factor}.default.weight"):
                raise ValueError(f"Not a standard default LoRA adapter: {name}")
        if state[name].shape != params[name].shape or state[name].dtype != params[name].dtype:
            raise ValueError(f"FedAvg upload shape/dtype mismatch: {name}")
    validate_complete_lora_state_dict(state, context="standard LoRA FedAvg upload")


def average_fedavg_server_upload(policy, state, device):
    """Direct equal-client parameter mean, never a mean/factorization of B @ A."""
    error = None
    try:
        validate_fedavg_server_upload(policy, state)
    except ValueError as exc:
        error = str(exc)
    signature = [(name, tuple(value.shape), str(value.dtype)) for name, value in sorted(state.items())]
    reports = [(error, signature)]
    if dist.is_initialized():
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, (error, signature))
    # All ranks fail together before any shape-dependent tensor collective.
    if any(report[0] for report in reports) or any(report[1] != signature for report in reports):
        raise ValueError(f"Invalid FedAvg uploads across ranks: {[report[0] for report in reports]}; "
                         "trainable key/shape/dtype signatures must agree")
    averaged = {}
    for name in sorted(state):
        value = state[name].to(device=device, dtype=torch.float32).contiguous()
        if dist.is_initialized():
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        averaged[name] = (value / len(reports)).to(dtype=state[name].dtype).cpu()
    return averaged


def average_fedavg_server_gradients(policy):
    """Synchronous data-parallel gradients, before clipping, including rank-unused parameters."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    params = [param for param in policy.parameters() if param.requires_grad]
    used = torch.tensor([param.grad is not None for param in params], device=params[0].device)
    used = used.to(torch.int32)
    dist.all_reduce(used, op=dist.ReduceOp.SUM)
    for param, count in zip(params, used.tolist(), strict=True):
        if not count:
            continue  # Preserve None, so AdamW does not decay globally unused parameters.
        grad = (param.grad.detach().float().contiguous() if param.grad is not None
                else torch.zeros_like(param, dtype=torch.float32))
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        param.grad = (grad / dist.get_world_size()).to(dtype=param.dtype)


def move_fedavg_server_policy(policy, optimizer, device):
    """Offload the inactive policy and optimizer without replacing Parameter objects."""
    identities = [id(param) for param in policy.parameters()]
    policy.to(device)
    if identities != [id(param) for param in policy.parameters()]:
        raise RuntimeError("Policy movement replaced parameters; optimizer references would be stale")
    if optimizer is not None:
        for param, state in optimizer.state.items():
            optimizer.state[param] = move_to_device_recursive(state, device)


def gather_fedavg_server_metrics(metrics):
    """Gather small scalar reports, never policy states or per-step histories."""
    reports = [metrics]
    if dist.is_initialized():
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, metrics)
    return reports


def format_server_metrics(moe_metrics, label="MoE"):
    """Original MoE metric formatting, also used by the standard-LoRA server."""
    # The policy-reported loss already includes weighted regularizers.
    total_loss = moe_metrics.get('loss', 0)
    moe_parts = [f"{label}: loss={total_loss:.4f}"]
    if "loss_action" in moe_metrics:
        moe_parts.append(f"loss_action={moe_metrics.get('loss_action', 0):.4f}")
    if "loss_gen" in moe_metrics:
        moe_parts.append(f"loss_gen={moe_metrics.get('loss_gen', 0):.4f}")
    if "grad_norm" in moe_metrics:
        moe_parts.append(f"grdn={moe_metrics.get('grad_norm', 0):.3f}")
    if "lr" in moe_metrics:
        moe_parts.append(f"lr={moe_metrics.get('lr', 0):.1e}")
    if "update_s" in moe_metrics:
        moe_parts.append(f"updt_s={moe_metrics.get('update_s', 0):.3f}")
    if "dataloading_s" in moe_metrics:
        moe_parts.append(f"data_s={moe_metrics.get('dataloading_s', 0):.3f}")
    if "loss_aux" in moe_metrics:
        moe_parts.append(f"loss_aux={moe_metrics.get('loss_aux', 0):.4f}")
    if "loss_aux_weighted" in moe_metrics:
        moe_parts.append(f"loss_aux_w={moe_metrics.get('loss_aux_weighted', 0):.6f}")
    if "loss_affordance" in moe_metrics:
        moe_parts.append(f"loss_aff={moe_metrics.get('loss_affordance', 0):.4f}")
    if "loss_affordance_weighted" in moe_metrics:
        moe_parts.append(f"loss_aff_w={moe_metrics.get('loss_affordance_weighted', 0):.6f}")
    if "loss_affordance_action_align" in moe_metrics:
        moe_parts.append(
            f"loss_aff_act={moe_metrics.get('loss_affordance_action_align', 0):.4f}"
        )
    if "affordance_target_valid" in moe_metrics:
        moe_parts.append(f"aff_valid={moe_metrics.get('affordance_target_valid', 0):.3f}")
    if "affordance_spatial_entropy" in moe_metrics:
        moe_parts.append(
            f"aff_sp_ent={moe_metrics.get('affordance_spatial_entropy', 0):.3f}"
        )
        moe_parts.append(
            f"aff_sp_peak={moe_metrics.get('affordance_spatial_peak', 0):.4f}"
        )
        if "affordance_action_gate" in moe_metrics:
            moe_parts.append(
                f"aff_act_gate={moe_metrics.get('affordance_action_gate', 0):.4f}"
            )
        if "affordance_router_gate" in moe_metrics:
            moe_parts.append(
                f"aff_route_gate={moe_metrics.get('affordance_router_gate', 0):.4f}"
            )
    if "router_lead_in_steps" in moe_metrics:
        moe_parts.append(f"router_lead_steps={int(moe_metrics.get('router_lead_in_steps', 0))}")
        moe_parts.append(
            f"router_lead_frac={moe_metrics.get('router_lead_in_fraction', 0):.3f}"
        )
    if "router_agg_final_entropy" in moe_metrics:
        moe_parts.append(
            f"router_agg_steps={int(moe_metrics.get('router_agg_usage_steps', 0))}"
        )
        moe_parts.append(
            f"router_agg_entropy={moe_metrics.get('router_agg_raw_entropy', 0):.4f}->"
            f"{moe_metrics.get('router_agg_final_entropy', 0):.4f}"
        )
        moe_parts.append(
            f"router_agg_range=[{moe_metrics.get('router_agg_mean_min', 0):.4f},"
            f"{moe_metrics.get('router_agg_mean_max', 0):.4f}]"
        )
    if "loss_fard" in moe_metrics:
        moe_parts.append(f"loss_fard={moe_metrics.get('loss_fard', 0):.4f}")
    if "loss_fard_weighted" in moe_metrics:
        moe_parts.append(f"loss_fard_w={moe_metrics.get('loss_fard_weighted', 0):.6f}")
    if "lambda_fard_effective" in moe_metrics:
        moe_parts.append(f"lambda_fard_eff={moe_metrics.get('lambda_fard_effective', 0):.6f}")
    if "fard_js_fg_a" in moe_metrics:
        moe_parts.append(f"fard_js={moe_metrics.get('fard_js_fg_a', 0):.4f}")
    if "fard_ug_agreement" in moe_metrics:
        moe_parts.append(f"fard_agree={moe_metrics.get('fard_ug_agreement', 0):.4f}")
    if "fard_reliability" in moe_metrics:
        moe_parts.append(f"fard_rel={moe_metrics.get('fard_reliability', 0):.4f}")
    if "pcea_mean_agreement" in moe_metrics:
        moe_parts.append(f"pcea_agree={moe_metrics.get('pcea_mean_agreement', 0):.4f}")
    if "pcea_weight_entropy" in moe_metrics:
        moe_parts.append(f"pcea_entropy={moe_metrics.get('pcea_weight_entropy', 0):.4f}")
    if "pcea_uniform_l1" in moe_metrics:
        moe_parts.append(f"pcea_uniform_l1={moe_metrics.get('pcea_uniform_l1', 0):.4f}")
    if "pcea_imputed_modules" in moe_metrics:
        moe_parts.append(f"pcea_imputed={int(moe_metrics.get('pcea_imputed_modules', 0))}")
    if "pcea_smoothing_l1" in moe_metrics:
        moe_parts.append(f"pcea_smooth_l1={moe_metrics.get('pcea_smoothing_l1', 0):.4f}")
    if "loss_visual_token_prune_ot" in moe_metrics:
        moe_parts.append(f"loss_ot={moe_metrics.get('loss_visual_token_prune_ot', 0):.4f}")
    if "loss_visual_token_prune_ot_weighted" in moe_metrics:
        moe_parts.append(
            f"loss_ot_w={moe_metrics.get('loss_visual_token_prune_ot_weighted', 0):.6f}"
        )
    return " | ".join(moe_parts)


def log_round_metrics(logger, client_metrics, fl_round, server_metrics=None, server_label="MoE"):
    """Original round summary; explicit (client ID, metrics) pairs preserve flow-specific IDs."""
    loss_strs = []
    flat_metrics = []
    for i, m in client_metrics:
        flat_metrics.append(m)
        loss_parts = [f"Client{i}: loss={m.get('loss', 0):.4f}"]
        if "loss_action" in m:
            loss_parts.append(f"loss_action={m.get('loss_action', 0):.4f}")
        if "loss_gen" in m:
            loss_parts.append(f"loss_gen={m.get('loss_gen', 0):.4f}")
        loss_parts.append(f"grdn={m.get('grad_norm', 0):.3f}")
        loss_parts.append(f"lr={m.get('lr', 0):.1e}")
        loss_parts.append(f"updt_s={m.get('update_s', 0):.3f}")
        loss_parts.append(f"data_s={m.get('dataloading_s', 0):.3f}")
        loss_strs.append(" | ".join(loss_parts))
    logger.info("After Aggregation:\n  " + "\n  ".join(loss_strs))

    if flat_metrics:
        min_lr = min(m.get('lr', 0) for m in flat_metrics)
        max_lr = max(m.get('lr', 0) for m in flat_metrics)
        logger.info(f"[Debug] Round {fl_round + 1} LR range: {min_lr:.2e} - {max_lr:.2e}")

    if server_metrics is not None:
        logger.info(format_server_metrics(server_metrics, label=server_label))


def run_lora_fedavg_server(cfg, fl_cfg, local_rank, world_size):
    """One persistent client per rank, direct FedAvg, then an independent standard-LoRA server."""
    from lerobot.datasets.utils import cycle
    from torch.utils.data import DataLoader, DistributedSampler

    validate_fedavg_server_config(cfg, fl_cfg, world_size)
    logger = logging.getLogger()
    set_seed(fl_cfg["seed"])
    device = torch.device(f"cuda:{local_rank}")
    dataset, data_stats = make_dataset(cfg)
    num_clients = fl_cfg["num_clients"]
    if fl_cfg["partition_strategy"] == "mole_by_task":
        from lerobot.datasets.mole_federated_dataset import (
            build_mole_partition_manifest, create_mole_partitioner,
            partition_mole_server, persist_mole_partition_manifest,
        )
        manifest = build_mole_partition_manifest(dataset, num_clients=num_clients)
        partitioner = create_mole_partitioner(manifest=manifest)
        server_dataset = partition_mole_server(dataset, manifest=manifest)
        if local_rank == 0:
            persist_mole_partition_manifest(manifest, cfg.output_dir)
    else:
        partitioner = create_robottwin_partitioner(
            strategy=fl_cfg["partition_strategy"], tasks_per_client=None,
            held_out_tasks=fl_cfg["held_out_tasks"], alpha=fl_cfg["dirichlet_alpha"],
        )
        server_partitioner = (create_robottwin_partitioner(strategy="fedforesight_server")
                              if fl_cfg["partition_strategy"] == "fedforesight_category" else partitioner)
        server_dataset = server_partitioner(dataset, num_clients + 1, num_clients, fl_cfg["seed"])
    client_dataset = partitioner(dataset, num_clients, local_rank, fl_cfg["seed"])
    if not len(client_dataset) or not len(server_dataset):
        raise ValueError("FedAvg client and independent server datasets must be nonempty")
    if not hasattr(client_dataset, "indices") or not hasattr(server_dataset, "indices"):
        raise ValueError("FedAvg requires indexed partitions to verify independent server data")
    if set(client_dataset.indices).intersection(server_dataset.indices):
        raise ValueError("FedAvg server dataset overlaps a client partition")

    client_cfg, server_cfg = deepcopy(cfg), deepcopy(cfg)
    client_cfg.policy.device = server_cfg.policy.device = str(device)
    client_cfg.steps = fl_cfg["num_rounds"] * fl_cfg["local_steps"] * fl_cfg["local_epochs"]
    server_cfg.steps = fl_cfg["num_rounds"] * fl_cfg["moe_steps"]
    server_cfg.batch_size = fl_cfg["moe_batch_size"]
    client_policy = make_policy(cfg=client_cfg.policy)
    broadcast_model(client_policy)
    if dist.is_initialized():
        for buffer in client_policy.buffers():
            dist.broadcast(buffer, src=0)
    validate_fedavg_server_upload(client_policy, get_trainable_weights(client_policy))
    move_fedavg_server_policy(client_policy, None, "cpu")
    server_policy = make_policy(cfg=server_cfg.policy)
    # Copy the entire initialized model once, including frozen base and buffers.
    # Subsequent communication/copies contain only the validated trainable pool.
    server_policy.load_state_dict(client_policy.state_dict(), strict=True)
    with torch.no_grad():
        client_buffers = dict(client_policy.named_buffers())
        for name, buffer in server_policy.named_buffers():
            buffer.copy_(client_buffers[name])
    validate_fedavg_server_upload(server_policy, get_trainable_weights(client_policy))
    move_fedavg_server_policy(server_policy, None, "cpu")
    optimizer, scheduler = make_optimizer_and_scheduler(client_cfg, client_policy)
    # Same optimizer options as the existing server, but include standard state-proj A/B too.
    server_optimizer = server_cfg.optimizer.build([p for p in server_policy.parameters() if p.requires_grad])
    server_scheduler = (server_cfg.scheduler.build(server_optimizer, server_cfg.steps)
                        if server_cfg.scheduler is not None else None)
    set_seed(fl_cfg["seed"] + local_rank)  # Independent local sampling/dropout, identical initial weights.
    loader_options = dict(num_workers=cfg.num_workers, pin_memory=True, drop_last=False)
    client_iter = iter(cycle(DataLoader(client_dataset, batch_size=cfg.batch_size,
                                       shuffle=True, **loader_options)))
    sampler = DistributedSampler(server_dataset, num_replicas=world_size, rank=local_rank,
                                 shuffle=True, seed=fl_cfg["seed"], drop_last=False)
    server_loader = DataLoader(server_dataset, batch_size=fl_cfg["moe_batch_size"],
                               sampler=sampler, **loader_options)

    def server_batches():
        epoch = 0
        while True:
            sampler.set_epoch(epoch)
            yield from server_loader
            epoch += 1

    server_iter = iter(server_batches())
    wandb_logger = None
    if local_rank == 0:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        write_json(fl_cfg, Path(cfg.output_dir) / "fl_training_flow.json")
        if cfg.wandb.enable and cfg.wandb.project:
            wandb_logger = WandBLogger(server_cfg)
    # Models are deliberately unwrapped: explicit gradient all-reduces allow safe
    # CPU/GPU offloading without a DDP reducer retaining old storage/buckets.
    for fl_round in range(fl_cfg["num_rounds"]):
        round_label = f"FL round {fl_round + 1}/{fl_cfg['num_rounds']}"
        if local_rank == 0:
            logger.info(f"[FedAvg] {round_label}: client local training, then server training")
        move_fedavg_server_policy(client_policy, optimizer, device)
        result = run_local_training(
            client_policy, client_iter, optimizer, client_cfg, fl_cfg, lr_scheduler=scheduler,
            local_rank=local_rank, global_start_step=fl_round * fl_cfg["local_steps"] * fl_cfg["local_epochs"],
        )
        client_reports = gather_fedavg_server_metrics({
            "client_id": local_rank,
            "metrics": {key: float(value) for key, value in result["metrics"].items() if value is not None},
        })
        client_logs = {}
        client_metrics = []
        if local_rank == 0:
            for report in sorted(client_reports, key=lambda report: report["client_id"]):
                client_id, metrics = report["client_id"], report["metrics"]
                client_metrics.append((client_id, metrics))
                client_logs.update({f"client_{client_id}/{key}": value for key, value in metrics.items()})
        del client_reports
        averaged = average_fedavg_server_upload(client_policy, result["model_state"], device)
        move_fedavg_server_policy(client_policy, optimizer, "cpu")
        torch.cuda.empty_cache()
        validate_fedavg_server_upload(server_policy, averaged)
        set_trainable_weights(server_policy, averaged)
        del averaged, result
        move_fedavg_server_policy(server_policy, server_optimizer, device)
        server_policy.train()
        window_stats, round_stats = {}, {}
        window_start = 1
        log_freq = max(1, fl_cfg["log_freq"])
        for server_step in range(1, fl_cfg["moe_steps"] + 1):
            start_time = time.perf_counter()
            batch = move_batch_to_device(next(server_iter), device)
            data_loading_time = time.perf_counter() - start_time
            server_optimizer.zero_grad(set_to_none=True)
            loss, output_dict = server_policy.forward(batch)
            loss.backward()
            average_fedavg_server_gradients(server_policy)
            clip = server_cfg.optimizer.grad_clip_norm
            grad_norm = torch.nn.utils.clip_grad_norm_(
                server_policy.parameters(), clip if clip > 0 else float("inf")
            )
            server_optimizer.step()
            if server_scheduler is not None:
                server_scheduler.step()
            update_time = time.perf_counter() - start_time
            metrics = {
                key: float(value.detach().item() if isinstance(value, torch.Tensor) else value)
                for key, value in output_dict.items()
                if isinstance(value, Real) or (isinstance(value, torch.Tensor) and value.numel() == 1)
            }
            metrics.update({
                "loss": loss.detach().item(), "grad_norm": float(grad_norm),
                "lr": server_optimizer.param_groups[0]["lr"],
                "dataloading_s": data_loading_time, "update_s": update_time,
            })
            for key, value in metrics.items():
                total, count = window_stats.get(key, (0.0, 0))
                window_stats[key] = (total + value, count + 1)
            del batch, loss, output_dict
            if server_step % log_freq == 0 or server_step == fl_cfg["moe_steps"]:
                # One collective per window on every rank; optional keys have their own counts.
                global_stats = {}
                for report in gather_fedavg_server_metrics(window_stats):
                    for key, (total, count) in report.items():
                        prev_total, prev_count = global_stats.get(key, (0.0, 0))
                        global_stats[key] = (prev_total + total, prev_count + count)
                window_metrics = {key: total / count for key, (total, count) in global_stats.items()}
                for key, (total, count) in global_stats.items():
                    prev_total, prev_count = round_stats.get(key, (0.0, 0))
                    round_stats[key] = (prev_total + total, prev_count + count)
                step = fl_round * fl_cfg["moe_steps"] + server_step
                if local_rank == 0:
                    logger.info(
                        f"[FedAvg-Server] {round_label} step={server_step}/{fl_cfg['moe_steps']} "
                        f"global_step={step} window={window_start}-{server_step} mean(all ranks): "
                        f"{format_server_metrics(window_metrics, label='Server')}"
                    )
                    log_metrics = {"fl_round": fl_round + 1, "server/step": server_step,
                                   **client_logs,
                                   **{f"server/window/{key}": value for key, value in window_metrics.items()}}
                    if server_step == fl_cfg["moe_steps"]:
                        round_metrics = {key: total / count for key, (total, count) in round_stats.items()}
                        log_round_metrics(logger, client_metrics, fl_round, round_metrics, server_label="Server")
                        log_metrics.update({f"server/{key}": value for key, value in round_metrics.items()})
                    # Clients join the first window; the final window and round share one log call.
                    if wandb_logger is not None:
                        wandb_logger.log_dict(log_metrics, step=step, mode="train")
                    client_logs.clear()
                window_stats.clear()
                window_start = server_step + 1
        server_optimizer.zero_grad(set_to_none=True)
        trained = get_trainable_weights(server_policy)
        validate_fedavg_server_upload(client_policy, trained)
        set_trainable_weights(client_policy, trained)  # Exact replacement, no local/global mixing.
        del trained
        step = (fl_round + 1) * fl_cfg["moe_steps"]
        if local_rank == 0:
            if cfg.save_checkpoint and ((fl_round + 1) % fl_cfg["save_freq"] == 0
                                        or fl_round + 1 == fl_cfg["num_rounds"]):
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, server_cfg.steps, step)
                save_checkpoint(checkpoint_dir, step, server_cfg, server_policy,
                                server_optimizer, server_scheduler, data_stats=data_stats)
                write_json({**fl_cfg, "completed_rounds": fl_round + 1}, checkpoint_dir / "fl_training_flow.json")
        if dist.is_initialized():
            dist.barrier()
        move_fedavg_server_policy(server_policy, server_optimizer, "cpu")
        torch.cuda.empty_cache()


@parser.wrap()
def fl_train(cfg: TrainPipelineConfig):
    """Main federated learning + LoRA-MoE training function."""
    # Initialize distributed
    local_rank, world_size = setup_distributed()

    # Setup logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if local_rank == 0 else logging.WARNING)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO if local_rank == 0 else logging.WARNING)
    formatter = logging.Formatter(f"[rank{local_rank}] %(asctime)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Get FL config
    fl_cfg = get_fl_config()

    if fl_cfg["training_flow"] not in ("lora_moe", "lora_fedavg_server"):
        raise ValueError(f"Unknown FL_TRAINING_FLOW: {fl_cfg['training_flow']}")
    if fl_cfg["training_flow"] == "lora_fedavg_server" and cfg.resume:
        raise ValueError("lora_fedavg_server does not support federated resume")
    if fl_cfg["partition_strategy"] == "intern_real_by_task":
        from lerobot.datasets.intern_real_dataset import validate_intern_real_config
        validate_intern_real_config(cfg)
        if (fl_cfg["training_flow"] != "lora_moe" or fl_cfg["num_clients"] != 4
                or fl_cfg["num_gpus"] != world_size or not 1 <= world_size <= 4
                or int(os.environ.get("FL_CLIENT_GPUS", "1")) != 1):
            raise ValueError("Real-robot FL requires four clients, 1..4 ranks, one GPU per active client")
        if (cfg.policy.loramoe_num_experts != 4 or cfg.policy.loramoe_router_top_k != 2
                or cfg.policy.loramoe_share_a_across_experts or cfg.policy.loramoe_ab_routing
                or not cfg.policy.loramoe_enable_a_experts or not cfg.policy.loramoe_enable_b_experts):
            raise ValueError("Real-robot FL requires four complete A/B experts, top-k 2, no shared A/AB router")
        if (fl_cfg.get("moe_router_weighted_aggregation", False)
                or any(getattr(cfg.policy, key, False) for key in (
                    "enable_affordance", "enable_affordance_v2", "enable_affordance_v3",
                    "enable_tcr", "tcr_enable_loss", "use_visual_token_prune"))):
            raise ValueError("Real-robot baseline disables router-weighted aggregation, affordance, TCR and pruning")
    cfg.validate()
    if fl_cfg["training_flow"] == "lora_fedavg_server":
        try:
            return run_lora_fedavg_server(cfg, fl_cfg, local_rank, world_size)
        finally:
            cleanup_distributed()
    cfg.optimizer = cfg.policy.get_optimizer_preset()
    cfg.scheduler = cfg.policy.get_scheduler_preset()

    num_clients = fl_cfg["num_clients"]
    if fl_cfg["partition_strategy"] == "mole_by_task":
        if fl_cfg["num_gpus"] != world_size or not 1 <= world_size <= 8:
            raise ValueError("MoLe requires NUM_GPUS == WORLD_SIZE in [1, 8] (single node)")
        if num_clients not in (4, 8):
            raise ValueError("MoLe requires four or eight clients")
        if not 1 <= int(cfg.policy.loramoe_router_top_k) <= int(cfg.policy.loramoe_num_experts):
            raise ValueError("MoLe requires 1 <= router top-k <= loramoe_num_experts")
        if (cfg.policy.loramoe_share_a_across_experts or cfg.policy.loramoe_ab_routing
                or not cfg.policy.loramoe_enable_a_experts or not cfg.policy.loramoe_enable_b_experts):
            raise ValueError("MoLe requires complete A/B experts, no shared A or dual A/B router")
        if (cfg.dataset.streaming or cfg.dataset.dist_loading or cfg.dataset.episodes is not None
                or len(cfg.dataset.repo_id.split()) != 1):
            raise ValueError("MoLe requires one complete map-style training dataset on every rank")
    if not bool(getattr(cfg.policy, "use_lora_moe", False)) or bool(
        getattr(cfg.policy, "use_lora_moe_forced_last", False)
    ):
        raise ValueError(
            "The affordance router-weighted trainer requires the original LoRA-MoE mode "
            "(use_lora_moe=True, use_lora_moe_forced_last=False)."
        )
    if int(cfg.policy.loramoe_num_experts) != num_clients:
        raise ValueError(
            "The original federated LoRA-MoE mapping requires one server expert slot per client: "
            f"loramoe_num_experts={cfg.policy.loramoe_num_experts}, num_clients={num_clients}."
        )
    if bool(fl_cfg.get("moe_router_weighted_aggregation", False)):
        router_agg_temperature = float(fl_cfg.get("moe_router_agg_temperature", 0.5))
        if router_agg_temperature <= 0:
            raise ValueError(
                f"MOE_ROUTER_AGG_TEMPERATURE must be positive, got {router_agg_temperature}"
            )
        if bool(getattr(cfg.policy, "enable_pcea", False)):
            raise ValueError(
                "Router-weighted aggregation and PCEA are mutually exclusive; set ENABLE_PCEA=false"
            )

    if local_rank == 0:
        logger.info("=" * 60)
        logger.info("ROBOTWIN FEDERATED LEARNING + LoRA-MoE")
        logger.info("=" * 60)
        logger.info(pformat(fl_cfg))
        logger.info(f"World size: {world_size}, Num clients: {num_clients}")
        logger.info(f"MoE steps per round: {fl_cfg.get('moe_steps', 50)}")

    set_seed(fl_cfg["seed"])
    torch.backends.cudnn.benchmark = True

    # Initialize WandB
    wandb_logger = None
    if local_rank == 0 and cfg.wandb.enable and cfg.wandb.project:
        logger.info("Initializing WandB for FL+MoE training...")
        wandb_logger = WandBLogger(cfg)
        logger.info("WandB initialized successfully!")

    # Dataset loading
    if local_rank == 0:
        logger.info("Creating dataset")

    logger.info(f"[Debug] Rank {local_rank}: calling make_dataset")
    dataset, data_stats = make_dataset(cfg)

    logger.info(f"[Debug] Rank {local_rank}: dataset type = {type(dataset).__name__}")

    if world_size > 1:
        dist.barrier()

    # Merge data_stats
    if world_size > 1:
        if local_rank == 0:
            all_data_stats = [None] * world_size
            dist.gather_object(data_stats, all_data_stats, dst=0)
        else:
            dist.gather_object(data_stats, None, dst=0)
    else:
        all_data_stats = [data_stats]

    if local_rank == 0:
        merged_data_stats = {}
        for rank_stats in all_data_stats:
            if rank_stats is not None:
                merged_data_stats.update(rank_stats)
        data_stats = merged_data_stats
    else:
        data_stats = None

    # Dataset partitioning
    num_gpus = fl_cfg["num_gpus"]
    my_client_ids = assign_clients_to_gpu(local_rank, num_gpus, num_clients)
    logger.info(f"[Rank {local_rank}] Total clients: {num_clients}, Num GPUs: {num_gpus}")
    logger.info(f"[Rank {local_rank}] Responsible for clients: {my_client_ids}")

    if fl_cfg["partition_strategy"] == "intern_real_by_task":
        from lerobot.datasets.intern_real_dataset import partition_intern_real, persist_intern_real_artifacts
        partitioner = partition_intern_real
        if not dist.is_initialized() or dist.get_rank() == 0:
            persist_intern_real_artifacts(dataset, cfg.output_dir)
    elif fl_cfg["partition_strategy"] == "mole_by_task":
        from lerobot.datasets.mole_federated_dataset import (
            build_mole_partition_manifest,
            create_mole_partitioner,
            partition_mole_server,
            persist_mole_partition_manifest,
        )

        mole_manifest = build_mole_partition_manifest(dataset, num_clients=num_clients)
        partitioner = create_mole_partitioner(manifest=mole_manifest)
        if not dist.is_initialized() or dist.get_rank() == 0:
            persist_mole_partition_manifest(mole_manifest, cfg.output_dir)
    else:
        partitioner = create_robottwin_partitioner(
            strategy=fl_cfg["partition_strategy"],
            tasks_per_client=None,
            held_out_tasks=fl_cfg["held_out_tasks"],
            alpha=fl_cfg["dirichlet_alpha"],
        )

    # ===============================================================================
    # Create TWO policies: client_policy (standard LoRA) and server_policy (LoRA-MoE)
    # ===============================================================================
    device = torch.device(f"cuda:{local_rank}")
    cfg.policy.device = str(device)

    # 1. Create client policy (standard LoRA, no MoE)
    # Debug: Log config values before creating policies
    logger.info(f"[DEBUG] Config use_lora_moe = {cfg.policy.use_lora_moe}")
    logger.info(f"[DEBUG] Config loramoe_num_experts = {cfg.policy.loramoe_num_experts}")
    logger.info(f"[DEBUG] Config loramoe_router_top_k = {cfg.policy.loramoe_router_top_k}")

    if local_rank == 0:
        logger.info("Creating client policy (standard LoRA)")

    # Save original config values before modifying
    original_use_lora_moe = cfg.policy.use_lora_moe
    original_loramoe_num_experts = cfg.policy.loramoe_num_experts
    original_loramoe_router_top_k = cfg.policy.loramoe_router_top_k

    logger.info(f"[DEBUG] Saving original values: use_lora_moe={original_use_lora_moe}, num_experts={original_loramoe_num_experts}")

    # Create client policy (standard LoRA, no MoE)
    cfg.policy.use_lora_moe = False
    cfg.policy.use_lora = True  # Ensure standard LoRA is used
    logger.info(f"[DEBUG] Creating client_policy with use_lora_moe={cfg.policy.use_lora_moe}, use_lora={cfg.policy.use_lora}")
    client_policy = make_policy(cfg=cfg.policy)
    client_policy.to(device)

    # Debug: check client_policy LoRA structure
    client_lora_modules = 0
    client_adapter_counts = []
    for name, module in client_policy.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            client_lora_modules += 1
            client_adapter_counts.append(len(module.lora_A))
    logger.info(f"[DEBUG] client_policy LoRA modules: {client_lora_modules}, adapter counts sample: {client_adapter_counts[:5]}")

    # Debug: check action head parameters in client_policy
    action_head_params = []
    for name, param in client_policy.named_parameters():
        if "action_in_proj" in name or "action_out_proj" in name or "action_time_mlp" in name:
            action_head_params.append((name, param.shape, param.requires_grad))
    logger.info(f"[Rank {local_rank}] client_policy action head params: {action_head_params[:6]}")

    # 2. Create server policy (LoRA-MoE) - for MoE training
    # All ranks create the same model (like lerobot_train.py)
    logger.info("Creating server policy (LoRA-MoE)")

    # Restore and set MoE config
    cfg.policy.use_lora_moe = original_use_lora_moe
    cfg.policy.loramoe_num_experts = original_loramoe_num_experts
    cfg.policy.loramoe_router_top_k = original_loramoe_router_top_k

    logger.info(f"[DEBUG] Creating server_policy with use_lora_moe={cfg.policy.use_lora_moe}, num_experts={cfg.policy.loramoe_num_experts}")

    # All ranks create the same server policy (will be wrapped by DDP later)
    server_policy = make_policy(cfg=cfg.policy)
    server_policy.to(device)

    # Debug: check action head parameters in server_policy
    server_action_head_params = []
    for name, param in server_policy.named_parameters():
        if "action_in_proj" in name or "action_out_proj" in name or "action_time_mlp" in name:
            if "weight" in name:
                server_action_head_params.append((name, param.shape, param.requires_grad))
    logger.info(f"[Rank {local_rank}] server_policy action head params: {server_action_head_params}")

    # Check LoRA-MoE is enabled in server
    # PEFT LoRA-MoE: uses lora_A/lora_B ModuleDict with multiple keys
    loramoe_enabled = False
    moe_module_count = 0
    lora_module_details = []
    for name, module in server_policy.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            adapter_count = len(module.lora_A)
            lora_module_details.append(f"{name}: {adapter_count} adapters")
            if adapter_count > 1:  # More than 1 adapter = MoE mode
                loramoe_enabled = True
                moe_module_count += 1

    logger.info(f"[DEBUG] Server policy LoRA-MoE modules found: {moe_module_count}")
    logger.info(f"[DEBUG] LoRA module details (first 10): {lora_module_details[:10]}")
    logger.info(f"[DEBUG] loramoe_enabled = {loramoe_enabled}")

    if loramoe_enabled:
        logger.info("LoRA-MoE is enabled in server policy!")
    else:
        logger.warning("LoRA-MoE NOT enabled! Please set enable_loramoe=True in policy config")

    # Use client_policy as the main policy for now
    policy = client_policy

    # Broadcast initial model (only client_policy needs broadcast, server_policy will use DDP)
    if world_size > 1:
        broadcast_model(policy, src_rank=0)
        # server_policy will be synchronized via DDP in MoE training phase

    # Optimizer and scheduler
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    initial_lr_scheduler_state = lr_scheduler.state_dict() if lr_scheduler is not None else None
    initial_optimizer_state = move_to_cpu_recursive(optimizer.state_dict())

    # Keep a separate optimizer state per client (in CPU memory)
    client_optimizer_states = {}

    # Keep a persistent optimizer state for MoE (only the latest, same as clients)
    moe_optimizer_states = None

    # Create moe_optimizer and moe_lr_scheduler once (like client optimizer), and save initial state
    moe_optimizer = None
    moe_lr_scheduler = None
    initial_moe_optimizer_state = None
    server_policy_ddp = None  # Reuse DDP wrapper across rounds
    moe_accelerator = None
    if loramoe_enabled:
        logger.info(f"[Rank {local_rank}] Creating moe_optimizer and moe_lr_scheduler (global, reused across rounds)")

        # Expert slots are already auto-created during _apply_lora_moe() in server_policy
        # No need to call init_expert_slots - server_policy already has all expert slots initialized

        # Get router_top_k from config
        router_top_k = getattr(cfg.policy, 'loramoe_router_top_k', 2)
        logger.info(f"[Rank {local_rank}] Setting LoRA-MoE mode with router_top_k={router_top_k}")
        # Use policy.model.set_lora_mode for mode switching
        if hasattr(server_policy, 'model'):
            server_policy.model.set_lora_mode("lora_moe")
        else:
            server_policy.set_lora_mode("lora_moe")
        set_all_experts_trainable(server_policy)
        moe_optimizer = create_moe_optimizer(server_policy, cfg.optimizer)
        # Create same lr_scheduler as client (using cfg.scheduler directly)
        moe_lr_scheduler = cfg.scheduler.build(moe_optimizer, cfg.steps) if cfg.scheduler is not None else None
        if moe_optimizer is not None:
            logger.info(f"[Rank {local_rank}] moe_optimizer created with {len(moe_optimizer.param_groups[0]['params'])} params")
            initial_moe_optimizer_state = move_to_cpu_recursive(moe_optimizer.state_dict())
        else:
            logger.warning(f"[Rank {local_rank}] No MoE parameters found!")

    if local_rank == 0:
        total_optimizer_params = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
        logger.info(f"[Optimizer] Trainable params: {total_optimizer_params:,}")

    if world_size > 1:
        broadcast_model(policy, src_rank=0)

    # Create dataloaders for each client
    from lerobot.datasets.utils import cycle
    client_dataloaders = {}

    for client_id in my_client_ids:
        ds = partitioner(dataset, num_clients, client_id, fl_cfg["seed"])

        num_workers = cfg.num_workers if hasattr(cfg, 'num_workers') and cfg.num_workers > 0 else 4
        prefetch_factor = 2 if num_workers > 0 else None
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        client_dataloaders[client_id] = iter(cycle(dl))
        logger.info(f"[Rank {local_rank}] Client {client_id}: {len(ds)} samples initialized")

    # =============================================================================
    # Create MoE dataloader (server-side mixed/residual task data)
    # =============================================================================
    if loramoe_enabled:
        if fl_cfg["partition_strategy"] == "intern_real_by_task":
            moe_dataset = partition_intern_real(dataset)
        elif fl_cfg["partition_strategy"] == "mole_by_task":
            moe_dataset = partition_mole_server(dataset, manifest=mole_manifest)
        elif fl_cfg["partition_strategy"] == "fedforesight_category":
            logger.info(f"[Rank {local_rank}] Creating MoE dataloader (FedForesight residual server tasks)")
            server_partitioner = create_robottwin_partitioner(strategy="fedforesight_server")
            moe_dataset = server_partitioner(dataset, num_clients, 0, fl_cfg["seed"])
        else:
            logger.info(f"[Rank {local_rank}] Creating MoE dataloader ({fl_cfg['partition_strategy']} server holdout)")
            moe_dataset = partitioner(dataset, num_clients + 1, num_clients, fl_cfg["seed"])

        # Use moe_batch_size for MoE training (independent of client batch_size)
        moe_batch_size = fl_cfg.get("moe_batch_size", 1)
        num_workers = cfg.num_workers if hasattr(cfg, 'num_workers') and cfg.num_workers > 0 else 4
        prefetch_factor = 2 if num_workers > 0 else None
        moe_dl = torch.utils.data.DataLoader(
            moe_dataset,
            batch_size=moe_batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        moe_dl_iter = iter(cycle(moe_dl))
        logger.info(f"[Rank {local_rank}] MoE dataloader: {len(moe_dataset)} samples, batch_size={moe_batch_size}")

    if local_rank == 0:
        logger.info(f"Starting FL+LoRA-MoE training: {fl_cfg['num_rounds']} rounds")
        training_start_time = time.perf_counter()

    global_step = 0

    # Initialize client_states before FL loop
    # This preserves weights across rounds for mixing
    client_states = {}

    # =============================================================================
    # FL + LoRA-MoE Training Loop
    # =============================================================================

    for fl_round in range(fl_cfg["num_rounds"]):
        if local_rank == 0:
            logger.info(f"\n{'='*50}")
            logger.info(f"FL+LoRA-MoE Round {fl_round + 1}/{fl_cfg['num_rounds']}")
            logger.info(f"{'='*50}")

        round_lr_scheduler_state = lr_scheduler.state_dict() if lr_scheduler is not None else None

        # Initialize empty dicts for this round (but preserve client_states across rounds)
        # client_states is initialized before the loop and preserved across rounds

        # =========================================================================
        # Phase 1: Client Training (LoRA mode) - using client_policy
        # =========================================================================
        logger.info(f"[Rank {local_rank}] Phase 1: Client training")

        # Move server_policy to CPU to free GPU memory for client training
        clear_router_runtime_caches(server_policy)
        server_policy.to("cpu")
        torch.cuda.empty_cache()
        logger.info(f"[Rank {local_rank}] Moved server_policy to CPU to save GPU memory")

        all_client_metrics = []

        for client_id in my_client_ids:
            logger.info(f"[Rank {local_rank}] Training client {client_id}/{num_clients}")

            # Debug: check client_states before loading
            if client_id in client_states:
                logger.info(f"[Phase1 Debug] client_states[{client_id}] has {len(client_states[client_id])} keys before loading")

                # Debug: print sample client_states keys and client_policy param names
                if local_rank == 0 and client_id == my_client_ids[0]:
                    state_keys = list(client_states[client_id].keys())
                    policy_params = list(client_policy.named_parameters())
                    policy_lora_keys = [n for n, _ in policy_params if 'lora_A' in n or 'lora_B' in n]
                    logger.info(f"[Phase1 Debug] Sample state_keys (first 3): {state_keys[:3]}")
                    logger.info(f"[Phase1 Debug] Sample policy_lora_keys (first 3): {policy_lora_keys[:3]}")
                    # Try direct matching
                    direct_matches = sum(1 for k in state_keys if k in [n for n, _ in policy_params])
                    logger.info(f"[Phase1 Debug] Direct matches: {direct_matches}/{len(state_keys)}")

            # Load client weights from previous round (if available)
            # This ensures each client starts with their own weights (after mixing)

            # Save a sample parameter value before loading for verification
            if client_id in client_states and local_rank == 0:
                sample_name = None
                sample_before = None
                for name, param in client_policy.named_parameters():
                    if 'lora_A.default.weight' in name and 'language_model.layers.0.self_attn.q_proj' in name:
                        sample_name = name
                        sample_before = param.data.clone()
                        break

            if client_id in client_states:
                validate_complete_lora_state_dict(
                    client_states[client_id], context=f"Phase1 client {client_id}"
                )
            load_client_state_to_model(client_policy, client_states, client_id)

            # Debug: verify loaded weights
            if client_id in client_states and local_rank == 0:
                loaded_count = sum(1 for name, _ in client_policy.named_parameters()
                                if name in client_states[client_id] and ('lora_A.default.weight' in name or 'lora_B.default.weight' in name))
                total_lora_params = sum(1 for name, _ in client_policy.named_parameters()
                                        if 'lora_A.default.weight' in name or 'lora_B.default.weight' in name)
                logger.info(f"[Phase1 Debug] Loaded {loaded_count} LoRA params from state_dict, model has {total_lora_params} LoRA params")

                # Check if a sample param changed
                if sample_name and sample_before is not None:
                    for name, param in client_policy.named_parameters():
                        if name == sample_name:
                            changed = not torch.allclose(sample_before, param.data)
                            logger.info(f"[Phase1 Debug] Sample param {sample_name.split('.')[-3:]}: changed={changed}")
                            break

            # Use client_policy for training
            if lr_scheduler is not None and round_lr_scheduler_state is not None:
                lr_scheduler.load_state_dict(round_lr_scheduler_state)

            # Clear stale optimizer state
            optimizer.state.clear()

            # Load this client's optimizer state (if any)
            if client_id in client_optimizer_states:
                # Move CPU state back to GPU
                opt_state_on_gpu = move_to_device_recursive(
                    client_optimizer_states[client_id],
                    device
                )
                optimizer.load_state_dict(opt_state_on_gpu)
                logger.info(f"  Loaded optimizer state for client {client_id}")
            else:
                # Round 1: use the initial state (moved to GPU)
                opt_state_on_gpu = move_to_device_recursive(
                    initial_optimizer_state,
                    device
                )
                optimizer.load_state_dict(opt_state_on_gpu)
                logger.info(f"  Using initial optimizer state for client {client_id}")

            # Train client
            train_result = run_local_training(
                policy=client_policy,
                dl_iter=client_dataloaders[client_id],
                optimizer=optimizer,
                cfg=cfg,
                fl_cfg=fl_cfg,
                lr_scheduler=lr_scheduler,
                local_rank=client_id,
                wandb_logger=None,
                global_start_step=0,
            )

            all_client_metrics.append(train_result["metrics"])

            # Save to client_states
            client_states[client_id] = get_trainable_weights(client_policy)
            # =====================================================================

            logger.info(f"[Rank {local_rank}] Client {client_id} training complete, LR={optimizer.param_groups[0]['lr']:.2e}")

            # Save this client's optimizer state to CPU memory
            client_optimizer_states[client_id] = move_to_cpu_recursive(optimizer.state_dict())

            # Clear optimizer internal state
            optimizer.state.clear()

        # Move server_policy back to GPU for Phase 2 (Expert Injection)
        server_policy.to(device)
        logger.info(f"[Rank {local_rank}] Moved server_policy back to GPU for Phase 2")

        # =========================================================================
        # Phase 2: Expert Injection & Sync
        # =========================================================================
        # Inject client LoRA weights to server MoE expert slots
        # All ranks must have the same server_policy with ALL experts loaded

        if loramoe_enabled:
            logger.info(f"[Rank {local_rank}] Phase 2: Expert injection from client to server")

            # First, each rank injects its own clients' weights
            for client_id in my_client_ids:
                inject_state_to_expert(server_policy, client_states, client_id)

            # IMPORTANT: Call _stack_lora_weights() ONCE after all injections
            # This is more efficient than calling it inside inject_state_to_expert for each client
            target_model = server_policy.model if hasattr(server_policy, 'model') else server_policy
            for name, module in target_model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, '_stack_lora_weights'):
                    module._stack_lora_weights()
            logger.info(f"[Rank {local_rank}] Called _stack_lora_weights() after all expert injections")

            # Sync: broadcast all experts to all ranks
            # This ensures all ranks have the same server_policy with all expert weights
            sync_expert_weights(server_policy, num_clients, world_size)
            for module in target_model.modules():
                if hasattr(module, '_stack_lora_weights'):
                    module._stack_lora_weights()
            logger.info(f"[Rank {local_rank}] Restacked LoRA experts after distributed synchronization")

            logger.info(f"[Rank {local_rank}] Expert injection complete (LoRA weights only, no action head)")
            # =====================================================================

            if bool(getattr(cfg.policy, "enable_affordance", False)):
                aggregated_affordance = aggregate_shared_affordance_state(
                    server_policy,
                    client_states,
                    my_client_ids,
                    device,
                )
                distribute_shared_affordance_state(client_states, aggregated_affordance, num_clients)
                logger.info(
                    f"[Rank {local_rank}] FedAvg aggregated {len(aggregated_affordance)} "
                    "shared affordance parameters"
                )

            # =====================================================================
            # Action Head FedAvg: Collect from all clients across ranks, aggregate
            # =====================================================================
            # Each rank only has its local clients in client_states.
            # Use AllReduce to compute FedAvg across all clients from all ranks.
            # IMPORTANT: Don't do local aggregation first - we need to average ALL clients, not average per rank!

            # Step 1: Each rank extracts ALL its local clients' action heads (NOT aggregated)
            # We keep them separate so AllReduce can correctly sum all clients
            local_action_heads_list = []  # List of dicts, one per client
            for client_id in my_client_ids:
                if client_id in client_states:
                    client_state = client_states[client_id]
                    action_head_state = {}
                    for name, param in client_state.items():
                        # Only include weights (not bias)
                        if ("action_in_proj.weight" in name or
                            "action_out_proj.weight" in name or
                            "action_time_mlp_in.weight" in name or
                            "action_time_mlp_out.weight" in name):
                            action_head_state[name] = param.clone()
                    if action_head_state:
                        local_action_heads_list.append(action_head_state)

            logger.info(f"[Rank {local_rank}] Local clients: {len(local_action_heads_list)}, clients: {my_client_ids}")
            if local_action_heads_list:
                logger.info(f"[Rank {local_rank}] Action head sample keys: {list(local_action_heads_list[0].keys())}")

            # Step 2: If multiple clients on this rank, average them locally first
            # This is equivalent to: each client gets weight 1/num_clients_on_this_rank
            if len(local_action_heads_list) > 1:
                # Average the local clients to get one "representative" for this rank
                # But we need to scale it so that after AllReduce, we get correct FedAvg
                # Actually, let's do: AllReduce(sum of all clients) / num_clients
                local_aggregated = aggregate_action_heads(local_action_heads_list)
            elif len(local_action_heads_list) == 1:
                local_aggregated = local_action_heads_list[0]
            else:
                # Create zero tensors with proper shape (will be used if no local clients)
                # Get shape from server_policy
                dummy_action_head = {}
                for name, param in server_policy.named_parameters():
                    if ("action_in_proj.weight" in name or
                        "action_out_proj.weight" in name or
                        "action_time_mlp_in.weight" in name or
                        "action_time_mlp_out.weight" in name):
                        dummy_action_head[name] = torch.zeros_like(param.data)
                local_aggregated = dummy_action_head

            logger.info(f"[Rank {local_rank}] Local aggregated: {len(local_aggregated)} params")

            # Step 3: AllReduce across all ranks
            # Each rank contributes its local average, but we need to scale appropriately
            # If rank has n clients, its local average = sum(n clients) / n
            # After AllReduce: sum of all local averages
            # But we want: sum of ALL clients / num_clients
            # So we need: (local_avg * n_clients_on_this_rank) summed, then / num_clients
            if world_size > 1 and local_aggregated:
                # Scale by number of local clients so that AllReduce gives correct sum
                num_local_clients = len(local_action_heads_list) if local_action_heads_list else 1
                for name in local_aggregated:
                    # Scale by number of clients on this rank
                    local_aggregated[name] = local_aggregated[name] * num_local_clients
                    # AllReduce to sum across ranks
                    tensor = local_aggregated[name].float().cuda()
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    # Divide by total number of clients to get FedAvg
                    local_aggregated[name] = (tensor / num_clients).cpu()

                logger.info(f"[Rank {local_rank}] FedAvg action head via AllReduce from {num_clients} clients")
                aggregated_action_head = local_aggregated
            elif world_size == 1:
                # Single GPU: local_aggregated is already the correct FedAvg
                # (aggregate_action_heads already divided by num_clients)
                # Do NOT divide again!
                aggregated_action_head = local_aggregated
                logger.info(f"[Rank {local_rank}] Using aggregated action head from {len(local_action_heads_list)} clients")
            else:
                aggregated_action_head = {}

            # Step 4: Inject aggregated action head to server_policy for MoE training
            if aggregated_action_head:
                inject_action_head_to_model(server_policy, aggregated_action_head)
                logger.info(f"[Rank {local_rank}] Injected FedAvg action head to server_policy for MoE training")
            else:
                logger.info(f"[Rank {local_rank}] No action head to inject, using pretrained from server_policy")

        # =========================================================================
        # Phase 3: MoE Router Training - Distributed (using Accelerator)
        # =========================================================================
        if loramoe_enabled:
            logger.info(f"[Rank {local_rank}] Phase 3: MoE distributed training")

            # Free GPU memory by moving client_policy to CPU (not needed during MoE training)
            client_policy.to("cpu")
            torch.cuda.empty_cache()
            logger.info(f"[Rank {local_rank}] Moved client_policy to CPU to free GPU memory")

            # Switch to LoRA-MoE mode using policy.model.set_lora_mode
            router_top_k = getattr(cfg.policy, 'loramoe_router_top_k', 2)
            logger.info(f"[Rank {local_rank}] Setting LoRA-MoE mode with router_top_k={router_top_k}")
            if hasattr(server_policy, 'config'):
                server_policy.config.fard_current_round = fl_round
            cfg.policy.fard_current_round = fl_round
            if hasattr(server_policy, 'model'):
                server_policy.model.set_lora_mode("lora_moe")
            else:
                server_policy.set_lora_mode("lora_moe")

            # Unfreeze all experts (MoE trains all experts + router)
            set_all_experts_trainable(server_policy)

            # Restore MoE optimizer state (reuse optimizer from initialization)
            if moe_optimizer is None:
                logger.warning(f"[Rank {local_rank}] No MoE optimizer found, skipping MoE training")
            else:
                if fl_round > 0 and moe_optimizer_states is not None:
                    # Restore from previous round (latest state)
                    moe_state_on_gpu = move_to_device_recursive(
                        moe_optimizer_states,
                        device
                    )
                    moe_optimizer.load_state_dict(moe_state_on_gpu)
                    logger.info(f"[Rank {local_rank}] Restored MoE optimizer state from previous round")
                else:
                    # Round 0: use initial state (already created in initialization)
                    if initial_moe_optimizer_state is not None:
                        moe_state_on_gpu = move_to_device_recursive(
                            initial_moe_optimizer_state,
                            device
                        )
                        moe_optimizer.load_state_dict(moe_state_on_gpu)
                        logger.info(f"[Rank {local_rank}] Using initial MoE optimizer state for round 0")

            # Use Accelerator for distributed MoE training
            # The distributed environment is already initialized via setup_distributed()
            from accelerate import Accelerator
            from accelerate.utils import DistributedDataParallelKwargs

            # Always create a NEW Accelerator/DDP wrapper each round to avoid DDP internal state issues
            # Reusing DDP wrapper across rounds causes:
            # 1. "lora_router_weight has been marked as ready twice" error
            # 2. "Expected to have finished reduction in the prior iteration" error
            # Note: MoE only activates top-k experts each time, so some params are unused
            # find_unused_parameters=True lets DDP skip unused params
            # Note: static_graph disabled because MoE sparse activation changes the graph
            if server_policy_ddp is not None:
                # Clean up old DDP wrapper before creating new one
                del server_policy_ddp
                torch.cuda.empty_cache()

            # Create Accelerator for MoE training
            # Note: We don't use accelerator.prepare() for the model because we want to use
            # the existing DDP wrapper from previous rounds. Instead, we pass the accelerator
            # to the training function for backward/sync operations.
            moe_accelerator = Accelerator(
                step_scheduler_with_optimizer=False,
                kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
            )

            # Wrap model with DDP using Accelerator
            # Note: We need to prepare the model, optimizer, and dataloader for proper DDP training
            # First, move model to device and prepare
            server_policy = server_policy.to(f"cuda:{local_rank}")

            # Optional: Apply torch.compile for optimization
            # torch_compile mode: "default", "reduce-overhead", "max-autotune"
            torch_compile_enabled = fl_cfg.get("moe_torch_compile", False)
            torch_compile_mode = fl_cfg.get("moe_torch_compile_mode", "reduce-overhead")
            compiled_model = None

            if torch_compile_enabled and local_rank == 0:
                logger.info(f"[Rank {local_rank}] Applying torch.compile with mode='{torch_compile_mode}'")

            if torch_compile_enabled:
                # Only compile on first round to save time (subsequent rounds reuse compiled graph)
                if fl_round == 0:
                    # Compile the model before DDP wrapping
                    # Use dynamo_cache to allow recompilation if needed
                    compiled_model = torch.compile(
                        server_policy,
                        mode=torch_compile_mode,
                        backend="inductor",
                        fullgraph=False,  # Allow non-full graph for MoE dynamic routing
                        dynamic=False,    # Disable dynamic shapes for stability
                    )
                    server_policy = compiled_model
                    if local_rank == 0:
                        logger.info(f"[Rank {local_rank}] torch.compile applied successfully")
                else:
                    # Reuse the compiled model from previous round
                    if hasattr(server_policy, '_orig_mod'):
                        # DDP wrapped model
                        server_policy = server_policy._orig_mod
                    if local_rank == 0:
                        logger.info(f"[Rank {local_rank}] Reusing compiled model from round {fl_round}")

            # Prepare with Accelerator - this handles DDP wrapping automatically
            server_policy_ddp, moe_optimizer, moe_lr_scheduler = moe_accelerator.prepare(
                server_policy, moe_optimizer, moe_lr_scheduler
            )

            logger.info(f"[Rank {local_rank}] Created Accelerator with DDP wrapper (find_unused_parameters=True)")

            # Distributed MoE training with true data parallelism
            # Each rank gets a different shard of moe_dataset via DistributedSampler
            moe_result = run_moe_training_distributed(
                policy=server_policy_ddp,
                dataset=moe_dataset,
                optimizer=moe_optimizer,
                cfg=cfg,
                fl_cfg=fl_cfg,
                accelerator=moe_accelerator,
                lr_scheduler=moe_lr_scheduler,
            )

            # Note: Keep DDP wrapper for reuse in next round (don't delete)
            # Get back the original model from DDP for non-DDP operations
            # With Accelerator, use unwrap_model to get the original model
            server_policy = moe_accelerator.unwrap_model(server_policy_ddp)
            clear_router_runtime_caches(server_policy)
            moe_accelerator.wait_for_everyone()
            del server_policy_ddp
            server_policy_ddp = None
            # Clean up Accelerator for next round
            # Note: Do NOT call end_training() as it destroys the distributed process group
            del moe_accelerator
            torch.cuda.empty_cache()

            logger.info(
                f"[Rank {local_rank}] MoE distributed training complete: "
                f"loss={moe_result['metrics']['loss']:.4f} "
                f"loss_action={moe_result['metrics']['loss_action']:.4f}"
            )

            # Note: do NOT clear optimizer state here!
            # It is saved after Phase 5
            # if moe_optimizer is not None:
            #     moe_optimizer.state.clear()

            # =====================================================================
            # Action Head Distribution: Extract trained action head and distribute to all clients
            # =====================================================================
            # After MoE training, extract the updated action head from server_policy
            # and distribute it to all clients
            trained_action_head = extract_action_head_from_model(server_policy)
            if trained_action_head:
                logger.info(f"[Rank {local_rank}] Extracted trained action head: {len(trained_action_head)} parameters")

                # Broadcast to all ranks if needed
                if world_size > 1:
                    trained_action_head = broadcast_action_head(trained_action_head, src_rank=0)

                # Save the trained action head to client_states for all clients
                # This will be used in the next round
                trained_action_head = detach_to_cpu_recursive(trained_action_head)
                for client_id in range(num_clients):
                    if client_id not in client_states:
                        client_states[client_id] = {}
                    for name, param in trained_action_head.items():
                        client_states[client_id][name] = detach_to_cpu_recursive(param)
                logger.info(f"[Rank {local_rank}] Distributed trained action head to all {num_clients} clients")
            else:
                logger.warning(f"[Rank {local_rank}] No trained action head found in server_policy!")

            if bool(getattr(cfg.policy, "enable_affordance", False)):
                trained_affordance = extract_shared_affordance_state(server_policy)
                if world_size > 1:
                    trained_affordance = broadcast_action_head(trained_affordance, src_rank=0)
                distribute_shared_affordance_state(client_states, trained_affordance, num_clients)
                logger.info(
                    f"[Rank {local_rank}] Distributed {len(trained_affordance)} trained shared "
                    f"affordance parameters to {num_clients} clients"
                )

            # Save MoE checkpoint after training (only on rank 0, with moe_save_freq)
            moe_save_freq = fl_cfg.get("moe_save_freq", 5)
            if local_rank == 0 and (fl_round + 1) % moe_save_freq == 0:
                moe_checkpoint_dir = get_step_checkpoint_dir(
                    cfg.output_dir, fl_cfg["num_rounds"], fl_round + 1
                )
                logger.info(f"Saving MoE checkpoint to {moe_checkpoint_dir}")
                save_moe_checkpoint(
                    checkpoint_dir=moe_checkpoint_dir,
                    step=global_step,
                    fl_round=fl_round,
                    policy=server_policy,
                    cfg=cfg,
                    data_stats=data_stats,
                )

        # =========================================================================
        # Phase 4: Global Aggregation
        # =========================================================================
        # Note: With DDP distributed training, model parameters are already synchronized
        # across all ranks. We only need to aggregate experts to create global weights.
        if loramoe_enabled:
            logger.info(f"[Rank {local_rank}] Phase 4: Global aggregation")

            expert_aggregation_weights = None
            router_expert_weights = moe_result.get("router_expert_weights") if 'moe_result' in locals() else None
            if bool(getattr(cfg.policy, 'enable_pcea', False)):
                pcea_expert_weights = moe_result.get("pcea_expert_weights") if 'moe_result' in locals() else None
                if pcea_expert_weights:
                    eligible_modules = get_pcea_eligible_module_names(server_policy)
                    raw_pcea_weights = dict(pcea_expert_weights)
                    pcea_expert_weights, missing_modules, global_pcea_weights = smooth_and_impute_pcea_weights(
                        pcea_expert_weights,
                        eligible_modules,
                        moe_result.get("pcea_mean_agreements"),
                    )
                    if missing_modules:
                        logger.warning(
                            f"[Phase4] Imputed {len(missing_modules)} eligible modules from the global "
                            f"PCEA consensus; sample={missing_modules[:8]}"
                        )
                    observed_modules = sorted(raw_pcea_weights)
                    smoothing_l1 = sum(
                        float((pcea_expert_weights[name] - raw_pcea_weights[name]).abs().sum())
                        for name in observed_modules
                    ) / len(observed_modules)
                    final_entropies = []
                    final_uniform_l1 = []
                    for weights in pcea_expert_weights.values():
                        uniform = torch.full_like(weights, 1.0 / float(weights.numel()))
                        final_entropies.append(float(_normalized_entropy(weights)))
                        final_uniform_l1.append(float((weights - uniform).abs().sum()))
                    moe_result["metrics"].update({
                        "pcea_imputed_modules": len(missing_modules),
                        "pcea_smoothing_l1": smoothing_l1,
                        "pcea_weight_entropy": sum(final_entropies) / len(final_entropies),
                        "pcea_uniform_l1": sum(final_uniform_l1) / len(final_uniform_l1),
                    })
                    logger.info(
                        f"[Phase4] Hierarchical PCEA smoothing: "
                        f"mean_l1_shift={smoothing_l1:.4f} "
                        f"global_weights={[round(float(v), 4) for v in global_pcea_weights.tolist()]}"
                    )
                    expert_aggregation_weights = pcea_expert_weights
                    logger.info(
                        f"[Phase4] Using standalone PCEA aggregation for all "
                        f"{len(pcea_expert_weights)} eligible U/G/A modules; state_proj uses mean aggregation"
                    )
                else:
                    raise RuntimeError("PCEA is enabled but Phase 3 returned no expert aggregation weights")

            elif fl_cfg.get("moe_router_weighted_aggregation", False):
                if bool(getattr(cfg.policy, 'loramoe_ab_routing', False)):
                    logger.warning("[Phase4] AB routing enabled; using mean aggregation instead of router-weighted aggregation")
                elif expert_aggregation_weights is None:
                    if router_expert_weights:
                        expert_aggregation_weights = router_expert_weights
                        logger.info(
                            f"[Phase4] Using router-weighted expert aggregation for "
                            f"{len(router_expert_weights)} layers"
                        )
                    else:
                        logger.warning("[Phase4] Router-weighted aggregation requested but no usage stats found; falling back to mean")

            if expert_aggregation_weights is None:
                logger.info("[Phase4] Using uniform expert aggregation")

            # Debug: check aggregated_weights for duplicates
            aggregated_weights = detach_to_cpu_recursive(
                aggregate_experts_to_global(
                    server_policy,
                    expert_weights=expert_aggregation_weights,
                )
            )
            if local_rank == 0:
                total_keys = len(aggregated_weights)
                # Count base layers (before .lora_A/.lora_B)
                base_layers = set()
                for k in aggregated_weights:
                    base = k.split('.lora_')[0] if '.lora_' in k else k
                    base_layers.add(base)
                logger.info(f"[Phase4] aggregated_weights has {total_keys} keys, {len(base_layers)} unique base layers")
                if total_keys != len(base_layers):
                    logger.warning(f"[Phase4] DUPLICATE LAYERS FOUND!")

        # =========================================================================
        # Phase 5: Weight Mixing & Distribution
        # =========================================================================
        # Mix each packed complete LoRA as 0.5 * local + 0.5 * global.
        # (NOT to client_policy, which only has single adapter and would cause pollution)

        if loramoe_enabled:
            logger.info(f"[Rank {local_rank}] Phase 5: Weight mixing & distribution")

            # Build module lookup table ONCE (outside client loop, eliminates O(8 * n^2) redundancy)
            target_model = server_policy.model if hasattr(server_policy, 'model') else server_policy
            lora_module_map = {}  # base_name -> module
            for mod_name, module in target_model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B') and hasattr(module, '_stack_lora_weights'):
                    base = mod_name.replace('.base_layer', '')
                    lora_module_map[base] = module

            for client_id in my_client_ids:
                logger.info(f"[Phase5] Processing client {client_id}")

                # Extract all layer weights for this client expert
                client_layer_weights = {}
                exp_key = str(client_id)
                for base_name, module in lora_module_map.items():
                    if exp_key in module.lora_A and exp_key in module.lora_B:
                        client_layer_weights[base_name] = {
                            "lora_A": module.lora_A[exp_key].weight.data.clone(),
                            "lora_B": module.lora_B[exp_key].weight.data.clone(),
                            "scaling": get_lora_adapter_scaling(module, exp_key),
                        }

                # Handle state_proj (ModuleList with integer indices)
                if hasattr(target_model, 'state_proj_lora_A_moe') and hasattr(target_model, 'state_proj_lora_B_moe'):
                    num_experts = len(target_model.state_proj_lora_A_moe)
                    if client_id < num_experts:
                        client_layer_weights["model.state_proj"] = {
                            "lora_A": target_model.state_proj_lora_A_moe[client_id].weight.data.clone(),
                            "lora_B": target_model.state_proj_lora_B_moe[client_id].weight.data.clone(),
                            "scaling": float(getattr(target_model, "state_proj_lora_scaling", 1.0)),
                        }

                if client_layer_weights and aggregated_weights:
                    # Debug: key count check only on first client
                    if local_rank == 0 and client_id == my_client_ids[0]:
                        matches = sum(1 for k in aggregated_weights if k in client_layer_weights)
                        logger.info(f"[Phase5] Will mix {matches} layers")
                        logger.info(f"[Phase5] client_layer_weights: {len(client_layer_weights)}, aggregated: {len(aggregated_weights)}")

                    modules_touched = set()
                    mixed_layers = 0
                    skipped_layers = 0

                    for layer_idx, (layer_name, global_weights) in enumerate(aggregated_weights.items()):
                        if layer_name not in client_layer_weights:
                            skipped_layers += 1
                            continue

                        local_weights = client_layer_weights[layer_name]
                        local_scaling = float(local_weights["scaling"])
                        local_device = local_weights["lora_A"].device

                        if layer_name == "model.state_proj":
                            module = None
                        else:
                            module = lora_module_map.get(layer_name)
                            if module is None:
                                skipped_layers += 1
                                continue

                        global_adapter = {
                            "lora_A": global_weights["lora_A"].to(device=local_device, dtype=torch.float32),
                            "lora_B": global_weights["lora_B"].to(device=local_device, dtype=torch.float32),
                            "scaling": float(global_weights.get("scaling", local_scaling)),
                        }
                        mixed_adapter = average_complete_lora_adapters(
                            [local_weights, global_adapter],
                            weights=torch.tensor([0.5, 0.5], device=local_device),
                        )

                        if local_rank == 0 and client_id == my_client_ids[0] and layer_idx < 3:
                            logger.info(
                                f"[Phase5] Layer {layer_name}: "
                                f"local_A={tuple(local_weights['lora_A'].shape)}, "
                                f"local_B={tuple(local_weights['lora_B'].shape)}, "
                                f"global_A={tuple(global_adapter['lora_A'].shape)}, "
                                f"global_B={tuple(global_adapter['lora_B'].shape)}"
                            )

                        mixed_layers += 1

                        # Direct assignment instead of inject_weights_to_expert (avoids O(n) search)
                        if layer_name == "model.state_proj":
                            # state_proj uses ModuleList
                            if hasattr(target_model, 'state_proj_lora_A_moe') and client_id < len(target_model.state_proj_lora_A_moe):
                                target_model.state_proj_lora_A_moe[client_id].weight.data.copy_(mixed_adapter["lora_A"])
                                target_model.state_proj_lora_B_moe[client_id].weight.data.copy_(mixed_adapter["lora_B"])
                            else:
                                skipped_layers += 1
                                continue
                        else:
                            # Regular LoRA-MoE uses ModuleDict
                            module.lora_A[exp_key].weight.data.copy_(mixed_adapter["lora_A"])
                            module.lora_B[exp_key].weight.data.copy_(mixed_adapter["lora_B"])
                            modules_touched.add(module)

                        # Save mixed weights to client_states for next round's Phase 1
                        if client_id not in client_states:
                            client_states[client_id] = {}

                        # Build save key (standard LoRA format)
                        if layer_name == "model.state_proj":
                            lora_A_key = "model.state_proj_lora_A.weight"
                            lora_B_key = "model.state_proj_lora_B.weight"
                        else:
                            save_key_base = layer_name
                            if save_key_base.endswith('.base_layer'):
                                save_key_base = save_key_base[:-len('.base_layer')]
                            if not save_key_base.startswith('model.'):
                                save_key_base = f"model.{save_key_base}"
                            lora_A_key = f"{save_key_base}.lora_A.default.weight"
                            lora_B_key = f"{save_key_base}.lora_B.default.weight"

                        client_states[client_id][lora_A_key] = mixed_adapter["lora_A"].detach().cpu().clone()
                        client_states[client_id][lora_B_key] = mixed_adapter["lora_B"].detach().cpu().clone()

                    validate_complete_lora_state_dict(
                        client_states[client_id], context=f"Phase5 client {client_id}"
                    )
                    # Call _stack_lora_weights ONCE per unique module (instead of 588 times per client)
                    for module in modules_touched:
                        module._stack_lora_weights()
                    logger.info(f"[Phase5] Called _stack_lora_weights() for {len(modules_touched)} modules")
                    logger.info(
                        f"[Phase5] Client {client_id}: mixed_layers={mixed_layers}, skipped={skipped_layers}"
                    )

            # Debug: print client_states keys after saving
            if local_rank == 0 and my_client_ids:
                first_client = my_client_ids[0]
                if first_client in client_states:
                    keys = list(client_states[first_client].keys())
                    logger.info(f"[Phase5] client_states[{first_client}] has {len(keys)} keys")
                    # Check for state_proj keys
                    state_proj_keys = [k for k in keys if 'state_proj' in k]
                    if state_proj_keys:
                        logger.info(f"[Phase5] state_proj keys found: {state_proj_keys}")
                    # Print all unique keys (grouped by prefix)
                    prefixes = {}
                    for k in keys:
                        prefix = k.split('.')[0] + '.' + k.split('.')[1] if '.' in k else k
                        prefixes[prefix] = prefixes.get(prefix, 0) + 1
                    logger.info(f"[Phase5] Key prefixes: {prefixes}")
                    logger.info(f"[Phase5] Sample keys: {keys[:10]}")

            logger.info(f"[Rank {local_rank}] Phase 5 complete: saved mixed weights to client_states")

            # Save MoE optimizer state (latest only, same as client optimizer)
            if loramoe_enabled and moe_optimizer is not None:
                moe_optimizer_states = move_to_cpu_recursive(moe_optimizer.state_dict())
                logger.info(f"[Rank {local_rank}] Saved MoE optimizer state for round {fl_round}")

            # Move client_policy back to GPU for next round (server_policy stays on GPU for DDP reuse)
            if loramoe_enabled:
                client_policy.to(device)
                # server_policy stays on GPU for DDP wrapper reuse in next round
                logger.info(f"[Rank {local_rank}] Moved client_policy to GPU, server_policy stays on GPU")

        # =========================================================================
        # Gather all client metrics from all GPUs (aligned with lerobot_fl_robotwin)
        # =========================================================================
        if world_size > 1:
            if local_rank == 0:
                all_metrics = [None] * world_size
                dist.gather_object(all_client_metrics, all_metrics, dst=0)
            else:
                dist.gather_object(all_client_metrics, None, dst=0)
        else:
            all_metrics = [all_client_metrics]

        # =========================================================================
        # Logging and Checkpointing
        # =========================================================================

        if local_rank == 0:
            # Flatten nested lists: [[C1, C2], [C3, C4]] -> [C1, C2, C3, C4]
            flat_metrics = [client_m for gpu_list in all_metrics for client_m in gpu_list]

            log_round_metrics(
                logger, enumerate(flat_metrics), fl_round,
                moe_result['metrics']
                if loramoe_enabled and moe_optimizer is not None and 'metrics' in moe_result else None,
            )

            # WandB logging (aligned with lerobot_fl_robotwin)
            if wandb_logger:
                # Compute the average metrics over all clients (using flat_metrics)
                avg_all_clients = {}
                if flat_metrics:
                    for key in flat_metrics[0].keys():
                        avg_all_clients[key] = sum(m.get(key, 0) for m in flat_metrics) / len(flat_metrics)

                wandb_log_dict = {
                    "steps": global_step,
                    "loss": avg_all_clients.get("loss", 0),
                    "grad_norm": avg_all_clients.get("grad_norm", 0),
                    "lr": avg_all_clients.get("lr", 0),
                }
                if "loss_action" in avg_all_clients:
                    wandb_log_dict["loss_action"] = avg_all_clients["loss_action"]
                if "loss_gen" in avg_all_clients:
                    wandb_log_dict["loss_gen"] = avg_all_clients["loss_gen"]
                if loramoe_enabled and moe_optimizer is not None and 'metrics' in moe_result:
                    # MoE metrics: the policy-reported total already includes all weighted regularizers.
                    moe_metrics = moe_result['metrics']
                    moe_loss_action = moe_metrics.get('loss_action', 0)
                    moe_loss_gen = moe_metrics.get('loss_gen', 0)
                    moe_loss_aux = moe_metrics.get('loss_aux', 0)
                    moe_loss_aux_weighted = moe_metrics.get('loss_aux_weighted', 0)
                    moe_loss_affordance = moe_metrics.get('loss_affordance', 0)
                    moe_loss_affordance_weighted = moe_metrics.get('loss_affordance_weighted', 0)
                    moe_loss_affordance_action_align = moe_metrics.get('loss_affordance_action_align', 0)
                    moe_affordance_target_valid = moe_metrics.get('affordance_target_valid', 0)
                    moe_affordance_spatial_entropy = moe_metrics.get('affordance_spatial_entropy', 0)
                    moe_affordance_spatial_peak = moe_metrics.get('affordance_spatial_peak', 0)
                    moe_affordance_action_gate = moe_metrics.get('affordance_action_gate', 0)
                    moe_affordance_router_gate = moe_metrics.get('affordance_router_gate', 0)
                    moe_router_lead_in_steps = moe_metrics.get('router_lead_in_steps', 0)
                    moe_router_lead_in_fraction = moe_metrics.get('router_lead_in_fraction', 0)
                    moe_router_agg_usage_start_step = moe_metrics.get('router_agg_usage_start_step', 0)
                    moe_router_agg_usage_steps = moe_metrics.get('router_agg_usage_steps', 0)
                    moe_router_agg_temperature = moe_metrics.get('router_agg_temperature', 0)
                    moe_router_agg_raw_entropy = moe_metrics.get('router_agg_raw_entropy', 0)
                    moe_router_agg_final_entropy = moe_metrics.get('router_agg_final_entropy', 0)
                    moe_router_agg_raw_uniform_l1 = moe_metrics.get('router_agg_raw_uniform_l1', 0)
                    moe_router_agg_final_uniform_l1 = moe_metrics.get('router_agg_final_uniform_l1', 0)
                    moe_router_agg_mean_max = moe_metrics.get('router_agg_mean_max', 0)
                    moe_router_agg_mean_min = moe_metrics.get('router_agg_mean_min', 0)
                    moe_loss_fard = moe_metrics.get('loss_fard', 0)
                    moe_loss_fard_weighted = moe_metrics.get('loss_fard_weighted', 0)
                    moe_lambda_fard_effective = moe_metrics.get('lambda_fard_effective', 0)
                    moe_fard_confidence = moe_metrics.get('fard_confidence', 0)
                    moe_fard_agreement = moe_metrics.get('fard_ug_agreement', 0)
                    moe_fard_reliability = moe_metrics.get('fard_reliability', 0)
                    moe_fard_js = moe_metrics.get('fard_js_fg_a', 0)
                    moe_pcea_agreement = moe_metrics.get('pcea_mean_agreement', 0)
                    moe_pcea_entropy = moe_metrics.get('pcea_weight_entropy', 0)
                    moe_pcea_uniform_l1 = moe_metrics.get('pcea_uniform_l1', 0)
                    moe_pcea_imputed_modules = moe_metrics.get('pcea_imputed_modules', 0)
                    moe_pcea_smoothing_l1 = moe_metrics.get('pcea_smoothing_l1', 0)
                    moe_loss_ot = moe_metrics.get('loss_visual_token_prune_ot', 0)
                    moe_loss_ot_weighted = moe_metrics.get('loss_visual_token_prune_ot_weighted', 0)
                    moe_total_loss = moe_metrics.get('loss', 0)
                    wandb_log_dict["moe_loss"] = moe_total_loss
                    wandb_log_dict["moe_loss_action"] = moe_loss_action
                    wandb_log_dict["moe_loss_gen"] = moe_loss_gen
                    wandb_log_dict["moe_loss_aux"] = moe_loss_aux
                    wandb_log_dict["moe_loss_aux_weighted"] = moe_loss_aux_weighted
                    wandb_log_dict["moe_loss_affordance"] = moe_loss_affordance
                    wandb_log_dict["moe_loss_affordance_weighted"] = moe_loss_affordance_weighted
                    wandb_log_dict[
                        "moe_loss_affordance_action_align"
                    ] = moe_loss_affordance_action_align
                    wandb_log_dict["moe_affordance_target_valid"] = moe_affordance_target_valid
                    wandb_log_dict["moe_affordance_spatial_entropy"] = moe_affordance_spatial_entropy
                    wandb_log_dict["moe_affordance_spatial_peak"] = moe_affordance_spatial_peak
                    wandb_log_dict["moe_affordance_action_gate"] = moe_affordance_action_gate
                    wandb_log_dict["moe_affordance_router_gate"] = moe_affordance_router_gate
                    wandb_log_dict["moe_router_lead_in_steps"] = moe_router_lead_in_steps
                    wandb_log_dict["moe_router_lead_in_fraction"] = moe_router_lead_in_fraction
                    wandb_log_dict["moe_router_agg_usage_start_step"] = moe_router_agg_usage_start_step
                    wandb_log_dict["moe_router_agg_usage_steps"] = moe_router_agg_usage_steps
                    wandb_log_dict["moe_router_agg_temperature"] = moe_router_agg_temperature
                    wandb_log_dict["moe_router_agg_raw_entropy"] = moe_router_agg_raw_entropy
                    wandb_log_dict["moe_router_agg_final_entropy"] = moe_router_agg_final_entropy
                    wandb_log_dict["moe_router_agg_raw_uniform_l1"] = moe_router_agg_raw_uniform_l1
                    wandb_log_dict["moe_router_agg_final_uniform_l1"] = moe_router_agg_final_uniform_l1
                    wandb_log_dict["moe_router_agg_mean_max"] = moe_router_agg_mean_max
                    wandb_log_dict["moe_router_agg_mean_min"] = moe_router_agg_mean_min
                    wandb_log_dict["moe_loss_fard"] = moe_loss_fard
                    wandb_log_dict["moe_loss_fard_weighted"] = moe_loss_fard_weighted
                    wandb_log_dict["moe_lambda_fard_effective"] = moe_lambda_fard_effective
                    wandb_log_dict["moe_fard_confidence"] = moe_fard_confidence
                    wandb_log_dict["moe_fard_ug_agreement"] = moe_fard_agreement
                    wandb_log_dict["moe_fard_reliability"] = moe_fard_reliability
                    wandb_log_dict["moe_fard_js_fg_a"] = moe_fard_js
                    wandb_log_dict["moe_pcea_mean_agreement"] = moe_pcea_agreement
                    wandb_log_dict["moe_pcea_weight_entropy"] = moe_pcea_entropy
                    wandb_log_dict["moe_pcea_uniform_l1"] = moe_pcea_uniform_l1
                    wandb_log_dict["moe_pcea_imputed_modules"] = moe_pcea_imputed_modules
                    wandb_log_dict["moe_pcea_smoothing_l1"] = moe_pcea_smoothing_l1
                    wandb_log_dict["moe_loss_visual_token_prune_ot"] = moe_loss_ot
                    wandb_log_dict["moe_loss_visual_token_prune_ot_weighted"] = moe_loss_ot_weighted
                wandb_logger.log_dict(wandb_log_dict, step=global_step, mode="train")

        # Save checkpoint
        if local_rank == 0:
            if (fl_round + 1) % fl_cfg["save_freq"] == 0:
                checkpoint_dir = get_step_checkpoint_dir(
                    cfg.output_dir, fl_cfg["num_rounds"], fl_round + 1
                )
                logger.info(f"Saving checkpoint to {checkpoint_dir}")

                save_trainable_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=global_step,
                    cfg=cfg,
                    policy=client_policy,
                    data_stats=data_stats,
                )

        global_step += fl_cfg["local_steps"]

    if local_rank == 0:
        elapsed = time.perf_counter() - training_start_time
        logger.info("FL+LoRA-MoE training completed!")
        logger.info(f"Total training time: {format_time(elapsed)}")

    if world_size > 1:
        dist.destroy_process_group()


def main():
    register_third_party_plugins()
    fl_train()


if __name__ == "__main__":
    main()
