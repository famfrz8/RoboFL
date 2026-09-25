#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# RoboTwin Federated Learning Training Script.
#
# Usage:
#   torchrun --nproc_per_node=3 src/lerobot/scripts/lerobot_fl_robotwin.py
#
# Environment variables for FL parameters:
#   FL_NUM_CLIENTS=3
#   FL_LOCAL_STEPS=100
#   FL_NUM_ROUNDS=50
#   FL_AGGREGATION=fedavg
#   FL_PARTITION_STRATEGY=by_task

import logging
import os
import sys
import time
from pathlib import Path
from pprint import pformat

import torch
import torch.distributed as dist
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
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    save_checkpoint,
)
from lerobot.utils.federated_utils import fedavg_aggregate_state_dicts
from lerobot.rl.wandb_utils import WandBLogger


def save_trainable_checkpoint(
    checkpoint_dir: Path,
    step: int,
    cfg: TrainPipelineConfig,
    policy,
    data_stats: dict | None = None,
) -> None:
    """Save only trainable parameters (LoRA/LoRA-MoE + Action Head) to checkpoint.

    This is useful for LoRA fine-tuning where we only want to save the
    trainable adapter weights instead of the full model.
    """
    import logging
    logger = logging.getLogger()

    pretrained_dir = checkpoint_dir / "pretrained_model"
    pretrained_dir.mkdir(parents=True, exist_ok=True)

    # Extract trainable parameters only
    trainable_state = {}
    for name, param in policy.named_parameters():
        if param.requires_grad:
            trainable_state[name] = param.data.cpu().clone()

    logger.info(f"Saving trainable params: {len(trainable_state)} keys")

    # Save trainable weights as safetensors (use same filename as save_checkpoint)
    # This is consistent with save_checkpoint which saves to model.safetensors
    save_file(trainable_state, str(pretrained_dir / "model.safetensors"))

    # Save policy config (config.json) - consistent with save_checkpoint
    # This contains the active LoRA or LoRA-MoE configuration.
    if hasattr(policy, 'config'):
        policy.config.save_pretrained(pretrained_dir)

    # Save train config (train_config.json) - consistent with save_checkpoint
    cfg.save_pretrained(pretrained_dir)

    # Save data stats if provided - consistent with save_checkpoint
    if data_stats is not None:
        write_json(serialize_dict(data_stats), pretrained_dir / 'stats.json')

    # Save training step - consistent with save_checkpoint
    from lerobot.utils.train_utils import save_training_state
    save_training_state(checkpoint_dir, step, None, None)


def get_fl_config():
    """Get FL configuration from environment variables."""
    return {
        "num_clients": int(os.environ.get("FL_NUM_CLIENTS", "50")),  # Total number of clients
        "num_gpus": int(os.environ.get("NUM_GPUS", "3")),            # Number of GPUs actually used
        "local_steps": int(os.environ.get("FL_LOCAL_STEPS", "100")),
        "local_epochs": int(os.environ.get("FL_LOCAL_EPOCHS", "1")),
        # Client MoE training can need a smaller batch than the model's
        # nominal training batch because every expert participates in forward.
        "client_batch_size": int(os.environ.get("CLIENT_BATCH_SIZE", "0")),
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
        "vanilla_lora_moe": os.environ.get("ENABLE_VANILLA_LORA_MOE", "false").lower()
        in {"1", "true", "yes", "on"},
    }


def configure_vanilla_lora_moe(cfg: TrainPipelineConfig, fl_cfg: dict, logger: logging.Logger) -> None:
    """Force the plain client/server LoRA-MoE + FedAvg configuration."""
    if not fl_cfg.get("vanilla_lora_moe", False):
        return

    if fl_cfg.get("aggregation_strategy") != "fedavg":
        logger.warning(
            "ENABLE_VANILLA_LORA_MOE=true forces FL_AGGREGATION=fedavg; ignoring %s",
            fl_cfg.get("aggregation_strategy"),
        )
    fl_cfg["aggregation_strategy"] = "fedavg"

    policy_cfg = cfg.policy
    policy_cfg.use_lora = False
    policy_cfg.use_lora_moe = True
    policy_cfg.use_lora_moe_forced_last = False
    policy_cfg.loramoe_enable_a_experts = True
    policy_cfg.loramoe_enable_b_experts = True
    policy_cfg.loramoe_share_a_across_experts = False
    policy_cfg.loramoe_ab_routing = False

    # Keep this mode intentionally plain: no auxiliary method changes beyond
    # the standard LoRA-MoE load-balancing loss already controlled by lambda_aux.
    policy_cfg.enable_affordance = False
    policy_cfg.enable_affordance_v2 = False
    policy_cfg.enable_fard = False
    policy_cfg.enable_pcea = False
    policy_cfg.enable_tcr = False
    policy_cfg.tcr_enable_loss = False
    policy_cfg.use_visual_token_prune = False
    policy_cfg.use_ptq = False


def is_parameter_efficient_policy(policy) -> bool:
    """Return whether FL should communicate only trainable adapter parameters."""
    config = getattr(policy, "config", policy)
    return any(
        bool(getattr(config, field, False))
        for field in ("use_lora", "use_lora_moe", "use_lora_moe_forced_last")
    )


def apply_fedavg_trainable_state(policy, summed_state: dict[str, torch.Tensor], total_clients: int) -> None:
    """Apply equal-client FedAvg to the trainable policy parameters."""
    if total_clients <= 0:
        raise ValueError(f"total_clients must be positive, got {total_clients}")
    divisor = float(total_clients)
    for name, parameter in policy.named_parameters():
        if name in summed_state:
            parameter.data.copy_(summed_state[name].to(parameter.device) / divisor)


def validate_vanilla_lora_moe_policy(policy, fl_cfg: dict) -> None:
    """Fail fast unless both client copies and the global model are plain LoRA-MoE."""
    if not fl_cfg.get("vanilla_lora_moe", False):
        return

    config = policy.config
    invalid = (
        bool(getattr(config, "use_lora", False))
        or not bool(getattr(config, "use_lora_moe", False))
        or bool(getattr(config, "use_lora_moe_forced_last", False))
        or bool(getattr(config, "loramoe_ab_routing", False))
        or bool(getattr(config, "loramoe_share_a_across_experts", False))
        or not bool(getattr(config, "loramoe_enable_a_experts", False))
        or not bool(getattr(config, "loramoe_enable_b_experts", False))
        or fl_cfg.get("aggregation_strategy") != "fedavg"
    )
    if invalid:
        raise RuntimeError("Vanilla LoRA-MoE configuration invariants were not preserved")
    if not bool(getattr(policy.model, "_moe_mode", False)):
        raise RuntimeError("Vanilla LoRA-MoE was requested, but the policy model is not in MoE mode")

    trainable_names = [name for name, parameter in policy.named_parameters() if parameter.requires_grad]
    if not any("router" in name for name in trainable_names):
        raise RuntimeError("Vanilla LoRA-MoE has no trainable router parameters")
    if not any("lora_A" in name for name in trainable_names):
        raise RuntimeError("Vanilla LoRA-MoE has no trainable lora_A expert parameters")
    if not any("lora_B" in name for name in trainable_names):
        raise RuntimeError("Vanilla LoRA-MoE has no trainable lora_B expert parameters")


def assign_clients_to_gpu(local_rank: int, num_gpus: int, num_clients: int) -> list[int]:
    """
    Assign clients to each GPU.

    Example: 50 clients, 3 GPUs
    - GPU 0: clients [0, 3, 6, 9, ..., 48] (17 clients)
    - GPU 1: clients [1, 4, 7, 10, ..., 49] (17 clients)
    - GPU 2: clients [2, 5, 8, 11, ..., 47] (16 clients)
    """
    clients = []
    for i in range(local_rank, num_clients, num_gpus):
        clients.append(i)
    return clients


def setup_distributed():
    """Initialize distributed training."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # [Core change] Bind the GPU before any CUDA operation
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


def update_policy(policy, batch, grad_clip_norm, local_rank=0):
    """Single training step (no accelerator)."""
    policy.train()

    try:
        # Match the mixed-precision path used by the distributed MoE trainer.
        # This is especially important for client-side LoRA-MoE, where the
        # expert einsums otherwise create a larger fp32 activation peak.
        policy_config = getattr(policy, "config", None)
        use_bfloat16_autocast = (
            torch.cuda.is_available()
            and next(policy.parameters()).device.type == "cuda"
            and getattr(policy_config, "dtype", None) == "bfloat16"
        )
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=use_bfloat16_autocast,
        ):
            loss, output_dict = policy.forward(batch)

        # Backward pass
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


def run_local_training(policy, dl_iter, optimizer, cfg, fl_cfg, lr_scheduler=None, local_rank=0, wandb_logger=None, global_start_step=0):
    """Run local training for one federated round.

    Args:
        dl_iter: Pre-created iterator (from cycle(dataloader))
    """
    import logging
    logger = logging.getLogger()

    policy.train()

    local_steps = fl_cfg["local_steps"]
    local_epochs = fl_cfg["local_epochs"]
    log_freq = fl_cfg["log_freq"]
    batch_size = fl_cfg.get("client_batch_size") or cfg.batch_size

    # Create MetricsTracker (same as lerobot_train.py)
    if cfg.policy.type in ["a1", "qwena1"]:
        train_metrics = {
            "loss": AverageMeter("loss", ":.3f"),
            "loss_action": AverageMeter("loss_action", ":.3f"),
            "loss_gen": AverageMeter("loss_gen", ":.3f"),
            "loss_aux": AverageMeter("loss_aux", ":.3f"),
            "loss_aux_weighted": AverageMeter("loss_aux_w", ":.3f"),
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

    # Use MetricsTracker (same as lerobot_train.py)
    # FL has no real batch concept; use local_steps as reference
    train_tracker = MetricsTracker(
        batch_size=batch_size,
        num_frames=local_steps * batch_size,  # Estimated value
        num_episodes=local_steps,  # Estimated value
        metrics=train_metrics,
        initial_step=global_start_step,
    )

    metrics_history = []
    optimizer.zero_grad()

    device = next(policy.parameters()).device

    # Reset peak memory stats before training loop
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)

    for _ in range(local_epochs):
        for step in range(local_steps):
            start_time = time.perf_counter()

            batch = next(dl_iter)

            # Move batch to device
            batch = move_batch_to_device(batch, device)

            data_loading_time = time.perf_counter() - start_time

            loss_val, output_dict, grad_norm = update_policy(policy, batch, cfg.optimizer.grad_clip_norm, local_rank)

            optimizer.step()
            optimizer.zero_grad()

            # Step lr_scheduler
            if lr_scheduler is not None:
                lr_scheduler.step()

            update_time = time.perf_counter() - start_time

            # Record metrics with MetricsTracker (same as lerobot_train.py)
            train_tracker.loss = loss_val
            train_tracker.grad_norm = grad_norm
            train_tracker.lr = optimizer.param_groups[0]["lr"]
            train_tracker.dataloading_s = data_loading_time
            train_tracker.update_s = update_time
            if "loss_action" in output_dict:
                train_tracker.loss_action = output_dict["loss_action"]
            if "loss_gen" in output_dict:
                train_tracker.loss_gen = output_dict["loss_gen"]
            if "loss_aux" in output_dict:
                train_tracker.loss_aux = output_dict["loss_aux"]
                train_tracker.loss_aux_weighted = output_dict.get("loss_aux_weighted", 0.0)

            metrics_history.append({
                "loss": loss_val,
                "grad_norm": grad_norm,
                "lr": optimizer.param_groups[0]["lr"],
                "dataloading_s": data_loading_time,
                "update_s": update_time,
                "loss_action": output_dict.get("loss_action"),
                "loss_gen": output_dict.get("loss_gen"),
                "loss_aux": output_dict.get("loss_aux", 0.0),
                "loss_aux_weighted": output_dict.get("loss_aux_weighted", 0.0),
            })

            # ============================================================
            # Update step counter
            # ============================================================
            train_tracker.step()  # Update step counter

            # Free memory
            del batch, loss_val, output_dict
            # torch.cuda.empty_cache()

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
    avg_metrics["loss_aux"] = sum(m.get("loss_aux", 0) for m in metrics_history) / len(metrics_history)
    avg_metrics["loss_aux_weighted"] = sum(
        m.get("loss_aux_weighted", 0) for m in metrics_history
    ) / len(metrics_history)

    # ============================================================
    # For LoRA FL: Only aggregate trainable parameters (requires_grad=True)
    # This automatically includes: LoRA params + Action Head + any other trainable layers
    # Using named_parameters() avoids copying the entire frozen base model (~6GB)
    # ============================================================
    use_lora = is_parameter_efficient_policy(policy)

    if use_lora:
        # Use requires_grad to identify trainable parameters
        # This automatically captures: LoRA layers, Action Head, and any other trainable params
        trainable_state = {}
        total_size = 0
        lora_count = 0
        action_head_count = 0

        for name, param in policy.named_parameters():
            if param.requires_grad:
                trainable_state[name] = param.data.cpu().clone()
                param_size = param.numel() * param.element_size()
                total_size += param_size
                param_dtype = param.dtype  # Record the original dtype

                if "lora_" in name or ".lora_" in name:
                    lora_count += 1
                if "action_in_proj" in name or "action_out_proj" in name:
                    action_head_count += 1

        logger.info(f"[Client {local_rank}] Trainable params: {len(trainable_state)} keys, "
                   f"size: {total_size / 1e6:.2f} MB (dtype: {param_dtype}) "
                   f"(LoRA layers: {lora_count}, Action Head layers: {action_head_count})")
        model_state = trainable_state
    else:
        # Full model state for non-LoRA training
        model_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}

    return {
        "metrics": avg_metrics,
        "model_state": model_state,
    }


@parser.wrap()
def fl_train(cfg: TrainPipelineConfig):
    """Main federated learning training function."""
    # Initialize distributed
    local_rank, world_size = setup_distributed()

    # Setup logging - use root logger with rank-specific formatting
    logger = logging.getLogger()
    logger.setLevel(logging.INFO if local_rank == 0 else logging.WARNING)

    # Clear any existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler with rank-specific format
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO if local_rank == 0 else logging.WARNING)
    formatter = logging.Formatter(f"[rank{local_rank}] %(asctime)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Get FL config
    fl_cfg = get_fl_config()

    # The standalone switch normalizes all potentially conflicting policy and
    # aggregation settings before config validation and model construction.
    configure_vanilla_lora_moe(cfg, fl_cfg, logger)

    # Validate and initialize configs
    cfg.validate()
    cfg.optimizer = cfg.policy.get_optimizer_preset()
    cfg.scheduler = cfg.policy.get_scheduler_preset()

    num_clients = fl_cfg["num_clients"]

    if local_rank == 0:
        logger.info("=" * 60)
        logger.info("ROBOTWIN FEDERATED LEARNING")
        logger.info("=" * 60)
        logger.info(pformat(fl_cfg))
        logger.info(f"World size: {world_size}, Num clients: {num_clients}")

    set_seed(fl_cfg["seed"])

    torch.backends.cudnn.benchmark = True

    if local_rank == 0:
        logger.info(f"RoboTwin Tasks: {len(ROBOTWIN_TASKS)} tasks")
        logger.info(f"Starting FL with {num_clients} clients")

    # ============================================================
    # Initialize WandB (only on rank 0)
    # ============================================================
    wandb_logger = None
    if local_rank == 0 and cfg.wandb.enable and cfg.wandb.project:
        logger.info("Initializing WandB for FL training...")
        wandb_logger = WandBLogger(cfg)
        logger.info("WandB initialized successfully!")

    # ============================================================
    # Dataset loading - follow lerobot_train.py pattern
    # ============================================================

    # Main process creates dataset first (to avoid race conditions)
    if local_rank == 0:
        logger.info("Creating dataset")

    # Create full dataset (same as lerobot_train.py)
    # Note: make_dataset handles rank-specific repo assignment internally
    logger.info(f"[Debug] Rank {local_rank}: calling make_dataset")
    dataset, data_stats = make_dataset(cfg)

    logger.info(f"[Debug] Rank {local_rank}: dataset type = {type(dataset).__name__}")
    logger.info(f"[Debug] Rank {local_rank}: num_episodes = {dataset.num_episodes}")
    logger.info(f"[Debug] Rank {local_rank}: num_frames = {dataset.num_frames}")

    # Debug: Check dataset structure and weights
    if hasattr(dataset, 'dataset_weights'):
        logger.info(f"[Debug] Rank {local_rank}: dataset_weights = {dataset.dataset_weights}")
    if hasattr(dataset, 'datasets'):
        for i, ds in enumerate(dataset.datasets):
            logger.info(f"[Debug] Rank {local_rank}: dataset[{i}] repo_id={getattr(ds, 'repo_id', 'N/A')}, num_episodes={ds.num_episodes}, num_frames={ds.num_frames}")

    # Synchronize all ranks after dataset creation
    if world_size > 1:
        dist.barrier()

    # Merge data_stats from all ranks (same as lerobot_train.py)
    # Note: Since FL uses torch.distributed directly (not accelerate),
    # we use dist.gather_object instead of gather_object(accelerator)
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

    # ============================================================
    # Federated Learning Dataset Partitioning
    # ============================================================

    # Each GPU handles its assigned clients using assign_clients_to_gpu
    num_gpus = fl_cfg["num_gpus"]
    my_client_ids = assign_clients_to_gpu(local_rank, num_gpus, num_clients)
    logger.info(f"[Rank {local_rank}] Total clients: {num_clients}, Num GPUs: {num_gpus}")
    logger.info(f"[Rank {local_rank}] Responsible for clients: {my_client_ids}")

    # Create partitioner (used for each client during training)
    partitioner = create_robottwin_partitioner(
        strategy=fl_cfg["partition_strategy"],
        tasks_per_client=None,
        held_out_tasks=fl_cfg["held_out_tasks"],
        alpha=fl_cfg["dirichlet_alpha"],
    )

    logger.info(f"[Debug] Partitioner type: {type(partitioner).__name__}")


    # Create policy
    if local_rank == 0:
        logger.info("Creating policy")

    # Explicitly set device so the model is created on the right GPU
    device = torch.device(f"cuda:{local_rank}")
    cfg.policy.device = str(device)  # Pass to config
    policy = make_policy(cfg=cfg.policy)
    policy.to(device)
    validate_vanilla_lora_moe_policy(policy, fl_cfg)
    logger.info(f"==> [Rank {local_rank}] Model created on {device}")
    if fl_cfg.get("vanilla_lora_moe", False):
        logger.info(
            "[VanillaLoRAMoE] Client copies and global server model use the same %d-expert, top-%d LoRA-MoE; aggregation=FedAvg",
            cfg.policy.loramoe_num_experts,
            cfg.policy.loramoe_router_top_k,
        )

    # Broadcast initial model from rank 0
    if world_size > 1:
        broadcast_model(policy, src_rank=0)

    # Create single optimizer and scheduler (native lerobot)
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # Save the initial lr_scheduler state (reset before each client trains)
    initial_lr_scheduler_state = lr_scheduler.state_dict() if lr_scheduler is not None else None

    # Save the initial optimizer state (used in round 1)
    initial_optimizer_state = move_to_cpu_recursive(optimizer.state_dict())

    # Log optimizer info
    if local_rank == 0:
        total_optimizer_params = sum(p.numel() for group in optimizer.param_groups for p in group['params'])
        logger.info(f"[Optimizer] Trainable params: {total_optimizer_params:,} ({total_optimizer_params/1e6:.2f}M)")
        logger.info(f"[Optimizer] Clients per GPU: {len(my_client_ids)}")

    # Sync before training
    if world_size > 1:
        broadcast_model(policy, src_rank=0)

    # ============================================================
    # FL training loop - Serial Multi-Client per GPU
    # ============================================================

    num_rounds = fl_cfg["num_rounds"]
    local_steps = fl_cfg["local_steps"]

    # Keep a separate optimizer state per client (in CPU memory)
    client_optimizer_states = {}

    # ============================================================
    # Bug2 fix: create persistent iterators per client outside the fl_round loop
    # so each round resumes instead of restarting from the beginning
    # ============================================================
    from lerobot.datasets.utils import cycle
    client_dataloaders = {}
    for client_id in my_client_ids:
        # Create the dataset with a fixed seed
        ds = partitioner(dataset, num_clients, client_id, fl_cfg["seed"])
        client_batch_size = fl_cfg.get("client_batch_size") or cfg.batch_size

        # Follow the DataLoader config in lerobot_train.py
        num_workers = cfg.num_workers if hasattr(cfg, 'num_workers') and cfg.num_workers > 0 else 4
        prefetch_factor = 2 if num_workers > 0 else None
        dl = torch.utils.data.DataLoader(
            ds,
            batch_size=client_batch_size,
            shuffle=True,
            pin_memory=True,           # Set to True to speed up CPU->GPU transfer
            drop_last=False,
            num_workers=num_workers,  # Multi-process data loading
            prefetch_factor=prefetch_factor,
        )
        # Create persistent iterators
        client_dataloaders[client_id] = iter(cycle(dl))
        logger.info(
            f"[Rank {local_rank}] Client {client_id}: {len(ds)} samples initialized "
            f"(batch_size={client_batch_size})"
        )

    if local_rank == 0:
        logger.info(f"Starting FL training: {num_rounds} rounds, {local_steps} local steps per round")
        logger.info(f"Each GPU will train {len(my_client_ids)} clients SERIALLY per round")
        training_start_time = time.perf_counter()

    # Total accumulated step
    global_step = 0

    for fl_round in range(num_rounds):
        if local_rank == 0:
            logger.info(f"\n{'='*50}")
            logger.info(f"FL Round {fl_round + 1}/{num_rounds}")
            logger.info(f"{'='*50}")

        # ============================================================
        # Bug fix: save the lr_scheduler state at the start of the round
        # so every client starts from the same lr, which decays across rounds
        # ============================================================
        round_lr_scheduler_state = lr_scheduler.state_dict() if lr_scheduler is not None else None

        # ============================================================
        # Bug1 fix: back up the global model state at round start (trainable params only)
        # to avoid later clients using a model already updated by earlier clients
        # ============================================================
        use_lora = is_parameter_efficient_policy(policy)
        if use_lora:
            # Back up only trainable params (LoRA + Action Head)
            global_model_state = {
                k: v.cpu().clone()
                for k, v in policy.named_parameters()
                if v.requires_grad
            }
        else:
            # Full fine-tuning: back up all params
            global_model_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}

        # ============================================================
        # Train each client SERIALLY on this GPU
        # ============================================================

        all_client_metrics = []

        # 1. Prepare the local accumulation buffer (on GPU); only accumulate requires_grad params (LoRA + Action Head)
        use_lora = is_parameter_efficient_policy(policy)
        if use_lora:
            local_sum_buffer = {
                name: torch.zeros_like(param.data, device="cpu")
                for name, param in policy.named_parameters() if param.requires_grad
            }
        else:
            local_sum_buffer = {
                name: torch.zeros_like(param.data, device="cpu")
                for name, param in policy.named_parameters()
            }

        for client_id in my_client_ids:
            logger.info(f"[Rank {local_rank}] Training client {client_id}/{num_clients}")

            # ============================================================
            # Bug2 fix: reset to the global model before each client trains
            # ============================================================
            if use_lora:
                # LoRA: load only trainable params with strict=False
                policy.load_state_dict(global_model_state, strict=False)
            else:
                # Full fine-tuning: load all params
                policy.load_state_dict(global_model_state)
            policy.to(local_rank)

            # ========== Reset lr_scheduler ==========
            # Every client starts training from the same lr
            if lr_scheduler is not None and round_lr_scheduler_state is not None:
                lr_scheduler.load_state_dict(round_lr_scheduler_state)

            # Clear stale optimizer state
            optimizer.state.clear()
            # torch.cuda.empty_cache()

            # Load this client's optimizer state (if any)
            if client_id in client_optimizer_states:
                # Move CPU state back to GPU
                opt_state_on_gpu = move_to_device_recursive(
                    client_optimizer_states[client_id],
                    torch.cuda.current_device()
                )
                optimizer.load_state_dict(opt_state_on_gpu)
                logger.info(f"  Loaded optimizer state for client {client_id}")
            else:
                # Round 1: use the initial state (moved to GPU)
                opt_state_on_gpu = move_to_device_recursive(
                    initial_optimizer_state,
                    torch.cuda.current_device()
                )
                optimizer.load_state_dict(opt_state_on_gpu)
                logger.info(f"  Using initial optimizer state for client {client_id}")

            # Use the persistent iterator (created outside the loop)
            logger.info(f"[Rank {local_rank}] Client {client_id}: using persistent iterator")

            # Train this client with single optimizer and scheduler
            current_wandb_logger = wandb_logger if local_rank == 0 else None
            train_result = run_local_training(
                policy=policy,
                dl_iter=client_dataloaders[client_id],
                optimizer=optimizer,
                cfg=cfg,
                fl_cfg=fl_cfg,
                lr_scheduler=lr_scheduler,
                local_rank=client_id,
                wandb_logger=current_wandb_logger,
                global_start_step=0,
            )

            # Store metrics
            all_client_metrics.append(train_result["metrics"])

            # 2. Add the current client's params to the local accumulation buffer
            for name, param in policy.named_parameters():
                if use_lora and not param.requires_grad:
                    continue
                local_sum_buffer[name].add_(param.detach().cpu())

            logger.info(f"[Rank {local_rank}] Client {client_id} training complete, LR={optimizer.param_groups[0]['lr']:.2e}")

            # Save this client's optimizer state to CPU memory
            client_optimizer_states[client_id] = move_to_cpu_recursive(optimizer.state_dict())

            # Clear optimizer internal state
            optimizer.state.clear()
            # torch.cuda.empty_cache()

        # ============================================================
        # Use All-Reduce for cross-GPU aggregation (bypasses the 2GB limit)
        # ============================================================

        # 3. Cross-GPU global aggregation (All-Reduce)
        if world_size > 1:
            for name in local_sum_buffer:
                # NCCL cannot reduce CPU tensors. Move only one aggregation
                # tensor at a time instead of keeping the full FedAvg buffer
                # on every GPU throughout client training.
                reduced = local_sum_buffer[name].to(device, non_blocking=True)
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
                local_sum_buffer[name] = reduced.cpu()
                del reduced

        # 4. Compute the global average and load it back into the model
        total_clients = fl_cfg["num_clients"]
        apply_fedavg_trainable_state(policy, local_sum_buffer, total_clients)

        # 5. All GPUs are synchronized; use policy directly, no extra broadcast needed

        # Compute the average metrics over clients on this GPU
        avg_my_metrics = {}
        if len(all_client_metrics) > 0:
            for key in all_client_metrics[0].keys():
                avg_my_metrics[key] = sum(m[key] for m in all_client_metrics) / len(all_client_metrics)

        # Gather metrics (small; gather_object is fine)
        if world_size > 1:
            if local_rank == 0:
                all_metrics = [None] * world_size
                dist.gather_object(all_client_metrics, all_metrics, dst=0)
            else:
                dist.gather_object(all_client_metrics, None, dst=0)
        else:
            all_metrics = [all_client_metrics]

        # ============================================================
        # Log (after all_reduce all GPU models are synchronized)
        # ============================================================

        if local_rank == 0:
            # Flatten nested lists: [[C1, C2], [C3, C4]] -> [C1, C2, C3, C4]
            flat_metrics = [client_m for gpu_list in all_metrics for client_m in gpu_list]

            loss_strs = []
            for i, m in enumerate(flat_metrics):
                # Here m is the actual dict
                loss_parts = [f"Client{i}: loss={m['loss']:.3f}"]
                if "loss_action" in m:
                    loss_parts.append(f"loss_action={m['loss_action']:.3f}")
                if "loss_gen" in m:
                    loss_parts.append(f"loss_gen={m['loss_gen']:.3f}")
                if "loss_aux" in m:
                    loss_parts.append(f"loss_aux={m['loss_aux']:.3f}")
                    loss_parts.append(f"loss_aux_w={m.get('loss_aux_weighted', 0):.6f}")
                loss_parts.append(f"grdn={m['grad_norm']:.3f}")
                loss_parts.append(f"lr={m['lr']:.1e}")
                loss_parts.append(f"updt_s={m['update_s']:.3f}")
                loss_parts.append(f"data_s={m['dataloading_s']:.3f}")
                loss_strs.append(" | ".join(loss_parts))
            logger.info("After Aggregation:\n  " + "\n  ".join(loss_strs))
            logger.info(f"[Debug] Round {fl_round + 1} LR range: {min(m['lr'] for m in flat_metrics):.2e} - {max(m['lr'] for m in flat_metrics):.2e}")

            # ============================================================
            # Log to WandB
            # ============================================================
            if wandb_logger:
                # Compute the average metrics over all clients
                avg_all_clients = {}
                for key in flat_metrics[0].keys():
                    avg_all_clients[key] = sum(m[key] for m in flat_metrics) / len(flat_metrics)

                wandb_log_dict = {
                    "steps": global_step,
                    "loss": avg_all_clients["loss"],
                    "grad_norm": avg_all_clients["grad_norm"],
                    "lr": avg_all_clients["lr"],
                }
                if "loss_action" in avg_all_clients:
                    wandb_log_dict["loss_action"] = avg_all_clients["loss_action"]
                if "loss_gen" in avg_all_clients:
                    wandb_log_dict["loss_gen"] = avg_all_clients["loss_gen"]
                if "loss_aux" in avg_all_clients:
                    wandb_log_dict["loss_aux"] = avg_all_clients["loss_aux"]
                    wandb_log_dict["loss_aux_weighted"] = avg_all_clients["loss_aux_weighted"]

                wandb_logger.log_dict(wandb_log_dict, step=global_step, mode="train")

        # ============================================================
        # Save checkpoint (rank 0 only)
        # ============================================================
        if local_rank == 0:
            if (fl_round + 1) % fl_cfg["save_freq"] == 0:
                checkpoint_dir = get_step_checkpoint_dir(
                    cfg.output_dir, fl_cfg["num_rounds"], fl_round + 1
                )
                total_training_steps = (fl_round + 1) * local_steps * len(my_client_ids)
                logger.info(f"Saving aggregated checkpoint to {checkpoint_dir}")

                # Check if using LoRA
                use_lora = is_parameter_efficient_policy(policy)

                if use_lora:
                    # Only save trainable parameters (LoRA + Action Head)
                    save_trainable_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        step=total_training_steps,
                        cfg=cfg,
                        policy=policy,
                        data_stats=data_stats,
                    )
                else:
                    # Save full model for non-LoRA training
                    save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        step=total_training_steps,
                        cfg=cfg,
                        policy=policy,
                        optimizer=None,
                        scheduler=None,
                        data_stats=data_stats,
                    )
        # Accumulate global_step by fl_round, which is better for WandB
        global_step += local_steps

        # After all_reduce all GPU models are synchronized; no extra broadcast needed
        # torch.cuda.empty_cache()

    if local_rank == 0:
        elapsed = time.perf_counter() - training_start_time
        logger.info("RoboTwin FL completed!")
        logger.info(f"Total training time: {format_time(elapsed)}")
    if world_size > 1:
        dist.destroy_process_group()

def main():
    register_third_party_plugins()
    fl_train()

if __name__ == "__main__":
    main()
