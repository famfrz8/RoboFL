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
Federated Dataset Partitioning for InternVLA-A1.

This module provides utilities for partitioning datasets across federated
learning clients with various distribution strategies (IID, non-IID, etc.).
"""

import logging
import math
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Subset


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def partition_dataset_iid(
    dataset,
    num_clients: int,
    client_id: int,
    seed: int = 42,
) -> Subset:
    """
    Partition dataset using IID (Independent and Identically Distributed) strategy.

    The dataset is randomly shuffled and divided into num_clients equal parts.
    Each client receives one part.

    Args:
        dataset: The full dataset to partition.
        num_clients: Total number of clients.
        client_id: ID of the client requesting its data partition.
        seed: Random seed for shuffling.

    Returns:
        Subset of data for the specified client.
    """
    set_seed(seed)

    num_samples = len(dataset)
    indices = list(range(num_samples))
    random.shuffle(indices)

    # Split indices into num_clients parts
    split_size = num_samples // num_clients
    splits = []

    for i in range(num_clients):
        start = i * split_size
        end = start + split_size if i < num_clients - 1 else num_samples
        splits.append(indices[start:end])

    if client_id >= num_clients:
        raise ValueError(f"client_id {client_id} >= num_clients {num_clients}")

    client_indices = splits[client_id]
    logging.info(f"[FederatedDataset] IID partition: client {client_id} gets {len(client_indices)} samples")

    return Subset(dataset, client_indices)


def partition_dataset_non_iid_dirichlet(
    dataset,
    num_clients: int,
    client_id: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> Subset:
    """
    Partition dataset using Dirichlet non-IID distribution.

    Each client receives samples from a random subset of classes, with
    the Dirichlet distribution controlling the heterogeneity.

    Args:
        dataset: The full dataset (must have targets/labels attribute).
        num_clients: Total number of clients.
        client_id: ID of the client requesting its data partition.
        alpha: Dirichlet concentration parameter.
            - Smaller alpha = more heterogeneous (clients have fewer classes).
            - Larger alpha = more homogeneous (clients have more balanced classes).
        seed: Random seed for reproducibility.

    Returns:
        Subset of data for the specified client.
    """
    set_seed(seed)

    # Get targets/labels from dataset
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
    elif hasattr(dataset, 'labels'):
        targets = dataset.labels
    elif hasattr(dataset, '_data') and isinstance(dataset._data, tuple):
        targets = dataset._data[1]
    else:
        raise ValueError(
            "Dataset must have 'targets' or 'labels' attribute for non-IID partitioning"
        )

    num_samples = len(targets)
    unique_classes = torch.unique(torch.tensor(targets)).tolist()
    num_classes = len(unique_classes)

    logging.info(
        f"[FederatedDataset] Dirichlet partition: {num_classes} classes, alpha={alpha}"
    )

    # Create class-to-sample mapping
    class_to_indices = {c: [] for c in unique_classes}
    for idx, target in enumerate(targets):
        class_to_indices[target].append(idx)

    # Sample from Dirichlet for each class
    # This determines what fraction of each class goes to each client
    proportions = np.random.dirichlet([alpha] * num_clients, num_classes)

    client_indices = []
    for c_idx, c in enumerate(unique_classes):
        class_indices = class_to_indices[c]
        random.shuffle(class_indices)

        # Get the slice for this client
        n_class_samples = len(class_indices)
        start_idx = 0
        for client in range(num_clients):
            n_for_client = int(proportions[c_idx, client] * n_class_samples)
            if client == client_id:
                client_indices.extend(class_indices[start_idx:start_idx + n_for_client])
            start_idx += n_for_client

    logging.info(
        f"[FederatedDataset] Dirichlet partition: client {client_id} gets {len(client_indices)} samples"
    )

    return Subset(dataset, client_indices)


def partition_dataset_by_robot(
    dataset,
    num_clients: int,
    client_id: int,
    seed: int = 42,
) -> Subset:
    """
    Partition dataset by robot type (natural non-IID).

    This is useful when different robots have different workspaces,
    end-effectors, or action spaces.

    Args:
        dataset: The full dataset.
        num_clients: Total number of clients (should match number of robot types).
        client_id: ID of the client (corresponds to robot type).
        seed: Random seed.

    Returns:
        Subset of data for the specified client/robot.
    """
    set_seed(seed)

    # Get robot type from dataset metadata
    if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'robot_type'):
        robot_type = dataset.meta.robot_type
    elif hasattr(dataset, 'robot_type'):
        robot_type = dataset.robot_type
    else:
        raise ValueError(
            "Dataset must have 'robot_type' attribute for by-robot partitioning"
        )

    # This is a special case where each client gets a specific robot type
    # We'll handle this at a higher level in the dataset factory
    logging.info(
        f"[FederatedDataset] By-robot partition: client {client_id} assigned to robot type"
    )

    # Return full dataset - actual partitioning should happen at factory level
    return dataset


def partition_dataset_shards(
    dataset,
    num_clients: int,
    client_id: int,
    num_shards: int = 2,
    seed: int = 42,
    shuffle: bool = True,
) -> Subset:
    """
    Partition dataset using shard-based non-IID strategy.

    Each client receives num_shards random shards of data. This creates
    controlled heterogeneity where each client has different data subsets.

    Args:
        dataset: The full dataset.
        num_clients: Total number of clients.
        client_id: ID of the client requesting its data partition.
        num_shards: Number of shards per client.
        seed: Random seed.
        shuffle: Whether to shuffle samples within shards.

    Returns:
        Subset of data for the specified client.
    """
    set_seed(seed)

    num_samples = len(dataset)
    indices = list(range(num_samples))

    # Shard the dataset
    num_total_shards = num_clients * num_shards
    shard_size = num_samples // num_total_shards
    shards = []

    if shuffle:
        random.shuffle(indices)

    for i in range(num_total_shards):
        start = i * shard_size
        end = start + shard_size if i < num_total_shards - 1 else num_samples
        shards.append(indices[start:end])

    # Assign shards to client
    start_shard = client_id * num_shards
    end_shard = start_shard + num_shards
    client_indices = []
    for shard in shards[start_shard:end_shard]:
        client_indices.extend(shard)

    logging.info(
        f"[FederatedDataset] Shard partition: client {client_id} gets {len(client_indices)} samples "
        f"({num_shards} shards)"
    )

    return Subset(dataset, client_indices)


def create_federated_dataset_partitioner(
    strategy: Literal["iid", "non_iid_dirichlet", "non_iid_shard", "by_robot"] = "iid",
    **kwargs,
) -> callable:
    """
    Factory function to create a dataset partitioner.

    Args:
        strategy: Partitioning strategy.
        **kwargs: Strategy-specific arguments.

    Returns:
        A partitioner function that takes (dataset, num_clients, client_id, seed).
    """
    if strategy == "iid":
        return partition_dataset_iid
    elif strategy == "non_iid_dirichlet":
        return lambda ds, nc, cid, s: partition_dataset_non_iid_dirichlet(
            ds, nc, cid, alpha=kwargs.get("alpha", 0.5), seed=s
        )
    elif strategy == "non_iid_shard":
        return lambda ds, nc, cid, s: partition_dataset_shards(
            ds, nc, cid, num_shards=kwargs.get("num_shards", 2), seed=s
        )
    elif strategy == "by_robot":
        return partition_dataset_by_robot
    else:
        raise ValueError(f"Unknown partition strategy: {strategy}")


def get_client_data_stats(
    dataset: Subset,
    stats: dict,
    robot_type: str,
) -> dict:
    """
    Get data statistics for a federated client.

    Args:
        dataset: The client's subset of data.
        stats: Full dataset statistics.
        robot_type: Robot type identifier.

    Returns:
        Updated or subset statistics for the client.
    """
    # For federated learning, we typically use global stats
    # to ensure consistent normalization across clients
    return stats
