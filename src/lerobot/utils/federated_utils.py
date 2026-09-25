#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""
Federated Learning Utilities for InternVLA-A1.

This module provides functions for:
- Model parameter aggregation (FedAvg, FedProx, SCAFFOLD)
- Model weight serialization/deserialization
- Client sampling and selection
- Gradient tracking for SCAFFOLD
"""

import copy
import logging
import math
import random
from collections import defaultdict
from typing import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def fedavg_aggregate(
    client_models: list[nn.Module],
    client_weights: list[float] | None = None,
) -> nn.Module:
    """
    Federated Averaging (FedAvg) aggregation.

    Aggregates model parameters from multiple clients using weighted averaging.

    Args:
        client_models: List of client models to aggregate.
        client_weights: Optional list of weights (proportional to sample counts).
            If None, uses uniform averaging.

    Returns:
        Aggregated model with averaged parameters.

    Reference:
        McMahan et al., "Communication-Efficient Learning of Deep Networks
        from Decentralized Data", AISTATS 2017.
    """
    if not client_models:
        raise ValueError("No client models provided for aggregation")

    # Use uniform weights if not provided
    if client_weights is None:
        client_weights = [1.0 / len(client_models)] * len(client_models)

    # Validate weights sum to approximately 1
    weight_sum = sum(client_weights)
    if abs(weight_sum - 1.0) > 1e-6:
        client_weights = [w / weight_sum for w in client_weights]

    # Create a deep copy of the first model as the base
    global_model = copy.deepcopy(client_models[0])

    # Get all state dict keys (includes parameters and buffers)
    state_dict = global_model.state_dict()
    param_names = list(state_dict.keys())

    for param_name in param_names:
        # Initialize aggregated parameter
        aggregated_param = None

        for client_model, weight in zip(client_models, client_weights):
            client_state = client_model.state_dict()
            if param_name not in client_state:
                # Skip keys that don't exist in client model
                continue
            client_param = client_state[param_name].float()

            if aggregated_param is None:
                aggregated_param = weight * client_param
            else:
                aggregated_param += weight * client_param

        # Update global model
        if aggregated_param is not None:
            state_dict[param_name].copy_(aggregated_param)

    return global_model


def fedavg_uniform_aggregate(client_models: list[nn.Module]) -> nn.Module:
    """
    Uniform FedAvg aggregation (equal weight for each client).

    Simplified FedAvg where each client contributes equally regardless of
    the number of samples they trained on.

    Args:
        client_models: List of client models to aggregate.

    Returns:
        Aggregated model with uniformly averaged parameters.
    """
    return fedavg_aggregate(client_models, client_weights=None)


def fedavg_aggregate_state_dicts(
    client_states: list[dict],
    client_weights: list[float] | None = None,
) -> dict:
    """
    Federated Averaging (FedAvg) aggregation on state dicts.

    Aggregates model parameters from multiple clients using weighted averaging.

    Args:
        client_states: List of state_dicts to aggregate.
        client_weights: Optional list of weights (proportional to sample counts).
            If None, uses uniform averaging.

    Returns:
        Aggregated state_dict with averaged parameters.

    Reference:
        McMahan et al., "Communication-Efficient Learning of Deep Networks
        from Decentralized Data", AISTATS 2017.
    """
    if not client_states:
        raise ValueError("No client states provided for aggregation")

    # Use uniform weights if not provided
    if client_weights is None:
        client_weights = [1.0 / len(client_states)] * len(client_states)

    # Validate weights sum to approximately 1
    weight_sum = sum(client_weights)
    if abs(weight_sum - 1.0) > 1e-6:
        client_weights = [w / weight_sum for w in client_weights]

    # Get all keys from first state dict
    param_names = list(client_states[0].keys())

    # Debug: log key info
    print(f"[fedavg_aggregate] Number of clients: {len(client_states)}")
    print(f"[fedavg_aggregate] Weights: {client_weights}")
    print(f"[fedavg_aggregate] Total keys in state_dict: {len(param_names)}")

    # Separate parameters from buffers
    param_keys = [k for k in param_names if not k.startswith('_')]
    buffer_keys = [k for k in param_names if k.startswith('_')]
    print(f"[fedavg_aggregate] Parameter keys: {len(param_keys)}, Buffer keys: {len(buffer_keys)}")

    # Initialize aggregated state dict
    aggregated_state = {}

    for param_name in param_names:
        # Initialize aggregated parameter
        aggregated_param = None

        for client_state, weight in zip(client_states, client_weights):
            if param_name not in client_state:
                continue
            client_param = client_state[param_name].float()

            if aggregated_param is None:
                aggregated_param = weight * client_param
            else:
                aggregated_param += weight * client_param

        if aggregated_param is not None:
            aggregated_state[param_name] = aggregated_param

    # Debug: check aggregated result
    print(f"[fedavg_aggregate] Aggregated {len(aggregated_state)} parameters")
    sample_param = list(aggregated_state.keys())[0] if aggregated_state else None
    if sample_param:
        sample_val = aggregated_state[sample_param]
        print(f"[fedavg_aggregate] Sample: {sample_param} -> shape={sample_val.shape}")

    return aggregated_state


def fedprox_aggregate(
    client_models: list[nn.Module],
    global_model: nn.Module,
    client_weights: list[float] | None = None,
    mu: float = 0.01,
) -> nn.Module:
    """
    FedProx aggregation with proximal term.

    Aggregates client models while penalizing large deviations from the
    global model to ensure stability.

    Args:
        client_models: List of client models to aggregate.
        global_model: The previous global model (before aggregation).
        client_weights: Optional list of weights for each client.
        mu: Proximal term coefficient.

    Returns:
        Aggregated model with FedProx-adjusted parameters.

    Reference:
        Li et al., "Federated Optimization in Heterogeneous Networks", MLSys 2020.
    """
    if not client_models:
        raise ValueError("No client models provided for aggregation")

    if client_weights is None:
        client_weights = [1.0 / len(client_models)] * len(client_models)

    weight_sum = sum(client_weights)
    if abs(weight_sum - 1.0) > 1e-6:
        client_weights = [w / weight_sum for w in client_weights]

    # Create base model
    global_model_copy = copy.deepcopy(global_model)
    global_state = global_model.state_dict()
    global_params = {k: v.clone() for k, v in global_state.items()}

    # Get all state dict keys (includes parameters and buffers)
    param_names = list(global_state.keys())

    for param_name in param_names:
        aggregated_param = None

        for client_model, weight in zip(client_models, client_models):
            client_state = client_model.state_dict()
            if param_name not in client_state:
                continue
            client_param = client_state[param_name].float()

            # Proximal term: add mu * (client_param - global_param)
            proximal_term = mu * (client_param - global_params[param_name])
            adjusted_param = client_param + proximal_term

            if aggregated_param is None:
                aggregated_param = weight * adjusted_param
            else:
                aggregated_param += weight * adjusted_param

        if aggregated_param is not None:
            global_model_copy.state_dict()[param_name].copy_(aggregated_param)

    return global_model_copy


class ScaffoldAggregator:
    """
    SCAFFOLD aggregator for variance reduction in federated learning.

    SCAFFOLD maintains control variates (client learning rates) to correct
    for client drift during aggregation.

    Reference:
        Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging
        for Federated Learning", ICML 2020.
    """

    def __init__(self, model: nn.Module, num_clients: int, device: str = "cpu"):
        """
        Initialize SCAFFOLD aggregator.

        Args:
            model: Reference model architecture.
            num_clients: Total number of clients.
            device: Device to store control variates.
        """
        self.num_clients = num_clients
        self.device = device

        # Global control variates (c)
        self.global_c = {name: torch.zeros_like(param, device=device)
                         for name, param in model.named_parameters()}

        # Client control variates (c_i) - stored per client
        self.client_cs: dict[int, dict[str, torch.Tensor]] = {}

        # Initialize client control variates to global values
        for client_id in range(num_clients):
            self.client_cs[client_id] = {
                name: param.clone().zero_()
                for name, param in model.named_parameters()
            }

    def get_global_control_variates(self) -> dict[str, torch.Tensor]:
        """Get a copy of global control variates."""
        return {name: c.clone() for name, c in self.global_c.items()}

    def get_client_control_variates(self, client_id: int) -> dict[str, torch.Tensor]:
        """Get control variates for a specific client."""
        if client_id not in self.client_cs:
            raise ValueError(f"Unknown client_id: {client_id}")
        return {name: c.clone() for name, c in self.client_cs[client_id].items()}

    def set_client_control_variates(self, client_id: int, cs: dict[str, torch.Tensor]):
        """Set control variates for a specific client."""
        self.client_cs[client_id] = {
            name: c.to(self.device) for name, c in cs.items()
        }

    def aggregate(
        self,
        client_models: list[nn.Module],
        client_deltas: list[dict[str, torch.Tensor]],
        client_weights: list[float] | None = None,
    ) -> nn.Module:
        """
        Perform SCAFFOLD aggregation.

        Args:
            client_models: List of client models after local training.
            client_deltas: List of control variate updates for each client.
            client_weights: Optional weights for each client.

        Returns:
            Aggregated global model.
        """
        if not client_models:
            raise ValueError("No client models provided")

        if len(client_models) != len(client_deltas):
            raise ValueError("Number of models must match number of control variate updates")

        if client_weights is None:
            client_weights = [1.0 / len(client_models)] * len(client_models)

        weight_sum = sum(client_weights)
        if abs(weight_sum - 1.0) > 1e-6:
            client_weights = [w / weight_sum for w in client_weights]

        # Create base model from first client
        global_model = copy.deepcopy(client_models[0])

        # Get global control variates for reference
        global_c = self.global_c

        param_names = [name for name, _ in global_model.named_parameters()]

        # Compute new global control variates
        new_global_c = {}
        for param_name in param_names:
            # c = sum(n_i * c_i) / sum(n_i)
            weighted_sum = sum(
                w * delta[param_name]
                for w, delta in zip(client_weights, client_deltas)
            )
            new_global_c[param_name] = global_c[param_name] + weighted_sum

        # Update global control variates
        self.global_c = new_global_c

        # Aggregate model parameters
        for param_name in param_names:
            aggregated_param = None

            for client_model, weight, delta in zip(
                client_models, client_weights, client_deltas
            ):
                client_param = client_model.state_dict()[param_name].float()
                c_i = delta[param_name]

                # Apply correction using control variates
                corrected_param = client_param + (global_c[param_name] - c_i)

                if aggregated_param is None:
                    aggregated_param = weight * corrected_param
                else:
                    aggregated_param += weight * corrected_param

            global_model.state_dict()[param_name].copy_(aggregated_param)

        return global_model

    def compute_client_delta(
        self,
        client_model: nn.Module,
        global_model: nn.Module,
        client_id: int,
    ) -> dict[str, torch.Tensor]:
        """
        Compute control variate update for a client.

        Args:
            client_model: Client model after local training.
            global_model: Global model before local training.
            client_id: Client identifier.

        Returns:
            Control variate update delta for this client.
        """
        c = self.client_cs[client_id]
        global_c = self.global_c

        delta = {}
        for name, param in client_model.named_parameters():
            delta[name] = global_c[name] - c[name] - param.grad.float()

        return delta


def select_clients(
    num_clients: int,
    sampling_ratio: float,
    exclude: list[int] | None = None,
    seed: int | None = None,
) -> list[int]:
    """
    Randomly select a subset of clients for participation.

    Args:
        num_clients: Total number of available clients.
        sampling_ratio: Fraction of clients to select (0.0 to 1.0).
        exclude: List of client IDs to exclude from selection.
        seed: Random seed for reproducibility.

    Returns:
        List of selected client IDs.
    """
    exclude = exclude or []
    available = [i for i in range(num_clients) if i not in exclude]

    num_to_select = max(1, int(num_clients * sampling_ratio))
    num_to_select = min(num_to_select, len(available))

    if seed is not None:
        random.seed(seed)

    selected = random.sample(available, num_to_select)
    return selected


def compute_sample_weights(
    client_sample_counts: list[int],
    strategy: str = "proportional",
) -> list[float]:
    """
    Compute weights for client aggregation based on sample counts.

    Args:
        client_sample_counts: Number of samples each client trained on.
        strategy: Weighting strategy ("proportional", "uniform", "sqrt_proportional").

    Returns:
        List of weights for each client.
    """
    if strategy == "uniform":
        return [1.0 / len(client_sample_counts)] * len(client_sample_counts)

    if strategy == "sqrt_proportional":
        sqrt_counts = [math.sqrt(c) for c in client_sample_counts]
        total = sum(sqrt_counts)
        return [c / total for c in sqrt_counts]

    # Default: proportional to sample count
    total = sum(client_sample_counts)
    if total == 0:
        return [1.0 / len(client_sample_counts)] * len(client_sample_counts)
    return [c / total for c in client_sample_counts]


def save_global_model(
    model_or_state,
    path: str,
    round_num: int,
    metadata: dict | None = None,
) -> None:
    """
    Save global model checkpoint.

    Args:
        model_or_state: Model or state_dict to save.
        path: Directory path to save checkpoint.
        round_num: Current federated learning round number.
        metadata: Optional metadata to save with checkpoint.
    """
    import os
    from pathlib import Path

    Path(path).mkdir(parents=True, exist_ok=True)

    # Handle both model and state_dict
    if isinstance(model_or_state, dict):
        state_dict = {k: v.cpu() for k, v in model_or_state.items()}
    else:
        state_dict = {k: v.cpu() for k, v in model_or_state.state_dict().items()}

    checkpoint = {
        "model_state_dict": state_dict,
        "round": round_num,
    }
    if metadata:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, os.path.join(path, "global_model_round_{}.pt".format(round_num)))
    torch.save(checkpoint, os.path.join(path, "global_model_latest.pt"))


def load_global_model(
    model: nn.Module,
    path: str,
) -> tuple[nn.Module, int]:
    """
    Load global model checkpoint.

    Args:
        model: Model architecture to load weights into.
        path: Path to checkpoint file or directory.

    Returns:
        Tuple of (loaded model, round number).
    """
    import os
    from pathlib import Path

    checkpoint_path = Path(path)
    if checkpoint_path.is_dir():
        # Try to load latest checkpoint
        latest_path = checkpoint_path / "global_model_latest.pt"
        if latest_path.exists():
            checkpoint_path = latest_path
        else:
            # Find the latest round checkpoint
            checkpoints = list(checkpoint_path.glob("global_model_round_*.pt"))
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoint found in {path}")
            checkpoint_path = max(checkpoints, key=lambda p: int(p.stem.split("_")[-1]))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])
    round_num = checkpoint.get("round", 0)

    return model, round_num


def get_model_diff(
    model1: nn.Module,
    model2: nn.Module,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Compute model parameter differences (model1 - model2).

    Useful for efficient communication in federated learning.

    Args:
        model1: First model.
        model2: Second model.
        device: Device to place tensors on.

    Returns:
        Dictionary of parameter name to difference tensor.
    """
    diff = {}
    for name, param in model1.named_parameters():
        diff[name] = (param.float() - model2.state_dict()[name].float()).to(device)
    return diff


def apply_model_diff(
    model: nn.Module,
    diff: dict[str, torch.Tensor],
    scale: float = 1.0,
) -> None:
    """
    Apply model difference to a model in-place.

    Args:
        model: Model to update.
        diff: Dictionary of parameter differences.
        scale: Scaling factor for the difference.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in diff:
                param.add_(diff[name].to(param.device) * scale)


def broadcast_model(
    model: nn.Module,
    src_rank: int = 0,
    world_size: int = 1,
) -> None:
    """
    Broadcast model parameters from source rank to all other ranks.

    Useful for synchronizing the global model after aggregation.

    Args:
        model: Model to broadcast (modified in-place on all ranks).
        src_rank: Source rank that has the correct parameters.
        world_size: Total number of processes.
    """
    import torch.distributed as dist

    if world_size <= 1:
        return

    for param in model.parameters():
        dist.broadcast(param.data, src=src_rank)

