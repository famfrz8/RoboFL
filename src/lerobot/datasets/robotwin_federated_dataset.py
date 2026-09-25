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
RoboTwin Federated Dataset Partitioning.

This module provides specialized dataset partitioning for RoboTwin's 52 manipulation
tasks, supporting realistic federated learning scenarios including:
- Task-based partitioning (each client gets a subset of tasks)
- Source-based partitioning (each client gets data from different robots)
- Mixed partitioning (balanced task distribution)
- Dirichlet-based non-IID partitioning for task heterogeneity
"""

import logging
import random
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from torch.utils.data import Subset


# RoboTwin Task List (50 tasks)
ROBOTWIN_TASKS = [
    # Grasp tasks
    "grab_roller",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "place_object_basket",
    # Place tasks
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
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    # Stack tasks
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    # Manipulation tasks
    "adjust_bottle",
    "beat_block_hammer",
    "dump_bin_bigbin",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    # Button/Switch tasks
    "click_alarmclock",
    "click_bell",
    "press_stapler",
    "turn_switch",
    # Open/Close tasks
    "open_laptop",
    "open_microwave",
    "rotate_qrcode",
    # Other tasks
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stamp_seal",
]


def get_robottwin_tasks_by_category() -> dict[str, list[str]]:
    """Organize RoboTwin tasks by semantic category."""
    return {
        "grasp": [
            "grab_roller", "pick_diverse_bottles", "pick_dual_bottles",
            "put_bottles_dustbin", "put_object_cabinet",
        ],
        "place": [
            "place_object_basket", "place_a2b_left", "place_a2b_right",
            "place_bread_basket", "place_bread_skillet", "place_burger_fries",
            "place_can_basket", "place_cans_plasticbox", "place_container_plate",
            "place_dual_shoes", "place_empty_cup", "place_fan", "place_mouse_pad",
            "place_object_scale", "place_object_stand", "place_phone_stand",
            "place_shoe",
        ],
        "stack": [
            "stack_blocks_three", "stack_blocks_two", "stack_bowls_three", "stack_bowls_two",
        ],
        "manipulation": [
            "adjust_bottle", "beat_block_hammer", "dump_bin_bigbin", "handover_block",
            "handover_mic", "hanging_mug", "lift_pot", "move_can_pot",
            "move_pillbottle_pad", "move_playingcard_away", "move_stapler_pad",
        ],
        "button_switch": [
            "click_alarmclock", "click_bell", "press_stapler", "turn_switch",
            "open_laptop", "open_microwave", "rotate_qrcode",
        ],
        "other": [
            "blocks_ranking_rgb", "blocks_ranking_size", "scan_object",
            "shake_bottle", "shake_bottle_horizontally", "stamp_seal",
        ],
    }


ROBOTWIN_FEDFORESIGHT_CLIENT_TASKS: dict[int, list[str]] = {
    0: [
        "grab_roller", "pick_diverse_bottles", "pick_dual_bottles",
        "put_bottles_dustbin", "put_object_cabinet",
    ],
    1: [
        "place_object_basket", "place_bread_basket", "place_bread_skillet",
        "place_can_basket", "place_cans_plasticbox", "place_container_plate",
        "place_empty_cup",
    ],
    2: [
        "place_a2b_left", "place_a2b_right", "place_fan", "place_mouse_pad",
        "place_object_scale", "place_object_stand", "place_phone_stand",
        "place_shoe",
    ],
    3: [
        "stack_blocks_three", "stack_blocks_two", "stack_bowls_three",
        "stack_bowls_two", "blocks_ranking_rgb", "blocks_ranking_size",
    ],
    4: [
        "adjust_bottle", "lift_pot", "move_pillbottle_pad",
        "move_playingcard_away", "move_stapler_pad",
    ],
    5: [
        "beat_block_hammer", "press_stapler", "stamp_seal",
    ],
    6: [
        "click_alarmclock", "click_bell", "turn_switch", "open_laptop",
        "open_microwave", "rotate_qrcode",
    ],
    7: [
        "scan_object", "shake_bottle", "shake_bottle_horizontally",
    ],
}


ROBOTWIN_FEDFORESIGHT_SERVER_TASKS: list[str] = [
    "place_burger_fries", "place_dual_shoes", "dump_bin_bigbin",
    "handover_block", "handover_mic", "hanging_mug", "move_can_pot",
]


def validate_fedforesight_manifest() -> None:
    """Validate the 8-client plus residual-server FedForesight task manifest."""
    expected = set(ROBOTWIN_TASKS)
    all_tasks: list[str] = []
    for client_id in sorted(ROBOTWIN_FEDFORESIGHT_CLIENT_TASKS):
        all_tasks.extend(ROBOTWIN_FEDFORESIGHT_CLIENT_TASKS[client_id])
    all_tasks.extend(ROBOTWIN_FEDFORESIGHT_SERVER_TASKS)

    duplicate_tasks = sorted({task for task in all_tasks if all_tasks.count(task) > 1})
    if duplicate_tasks:
        raise ValueError(f"FedForesight manifest contains duplicate tasks: {duplicate_tasks}")

    actual = set(all_tasks)
    unknown_tasks = sorted(actual - expected)
    if unknown_tasks:
        raise ValueError(f"FedForesight manifest contains unknown tasks: {unknown_tasks}")

    missing_tasks = sorted(expected - actual)
    if missing_tasks:
        raise ValueError(f"FedForesight manifest is missing tasks: {missing_tasks}")

    if len(ROBOTWIN_FEDFORESIGHT_CLIENT_TASKS) != 8:
        raise ValueError("FedForesight requires exactly 8 client task groups")


def _episode_to_task_map(dataset) -> dict[int, str]:
    """Build a global episode-index to RoboTwin task-name mapping."""
    if not hasattr(dataset, 'datasets'):
        raise NotImplementedError("Only MultiLeRobotDataset is supported")

    episode_to_task: dict[int, str] = {}
    ep_counter = 0
    for ds in dataset.datasets:
        repo_id = getattr(ds, 'repo_id', '')
        parts = repo_id.split('/')
        task_name = parts[1] if len(parts) >= 2 else repo_id

        num_eps = getattr(ds, 'num_episodes', 0)
        for ep_idx in range(num_eps):
            episode_to_task[ep_counter + ep_idx] = task_name

        ep_counter += num_eps

    return episode_to_task


def _subset_by_task_names(dataset, task_names: list[str], label: str) -> Subset:
    """Return a frame-level subset containing all episodes from the requested tasks."""
    task_set = set(task_names)
    episode_to_task = _episode_to_task_map(dataset)

    my_episode_ids = [ep_idx for ep_idx, task in episode_to_task.items() if task in task_set]
    my_frame_indices = []
    from_index = dataset.meta.episodes["dataset_from_index"]
    to_index = dataset.meta.episodes["dataset_to_index"]

    for ep_idx in my_episode_ids:
        start_frame = from_index[ep_idx]
        end_frame = to_index[ep_idx]
        my_frame_indices.extend(range(start_frame, end_frame))

    missing_in_dataset = sorted(task_set - set(episode_to_task.values()))
    if missing_in_dataset:
        logging.warning("[RoboTwinFL] %s tasks not present in dataset: %s", label, missing_in_dataset)

    print(f"[RoboTwinFL] {label}: tasks={len(task_names)}, frames={len(my_frame_indices)}")
    return Subset(dataset, my_frame_indices)


def partition_robottwin_by_fedforesight_category(
    dataset,
    num_clients: int,
    client_id: int,
    seed: int = 42,
) -> Subset:
    """Partition RoboTwin using the FedForesight 8-client category manifest."""
    set_seed(seed)
    validate_fedforesight_manifest()

    if num_clients != 8:
        raise ValueError(f"FedForesight partitioning requires num_clients=8, got {num_clients}")
    if client_id not in ROBOTWIN_FEDFORESIGHT_CLIENT_TASKS:
        raise ValueError(f"FedForesight client_id must be in [0, 7], got {client_id}")

    if client_id == 0:
        print("[RoboTwinFL] ===== FEDFORESIGHT CLIENT TASK ASSIGNMENTS =====")
        for cid in sorted(ROBOTWIN_FEDFORESIGHT_CLIENT_TASKS):
            print(
                f"[RoboTwinFL] Client {cid}: "
                f"tasks={len(ROBOTWIN_FEDFORESIGHT_CLIENT_TASKS[cid])} "
                f"{ROBOTWIN_FEDFORESIGHT_CLIENT_TASKS[cid]}"
            )
        print(f"[RoboTwinFL] Server residual: tasks={len(ROBOTWIN_FEDFORESIGHT_SERVER_TASKS)} {ROBOTWIN_FEDFORESIGHT_SERVER_TASKS}")
        print("[RoboTwinFL] =============================================")

    return _subset_by_task_names(
        dataset,
        ROBOTWIN_FEDFORESIGHT_CLIENT_TASKS[client_id],
        label=f"FedForesight client {client_id}",
    )


def partition_robottwin_fedforesight_server_residual(
    dataset,
    seed: int = 42,
) -> Subset:
    """Return the residual mixed-task server subset for FedForesight MoE training."""
    set_seed(seed)
    validate_fedforesight_manifest()
    return _subset_by_task_names(
        dataset,
        ROBOTWIN_FEDFORESIGHT_SERVER_TASKS,
        label="FedForesight server residual",
    )


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def partition_robottwin_by_category(
    dataset,
    num_clients: int,
    client_id: int,
    seed: int = 42,
) -> Subset:
    """
    Partition RoboTwin dataset by category - each client gets ONE full category.

    This is different from by_task which distributes tasks across clients in a
    balanced manner. Here each client specializes in one category completely.

    Args:
        dataset: The full RoboTwin MultiLeRobotDataset.
        num_clients: Total number of clients.
        client_id: This client's ID.
        seed: Random seed.

    Returns:
        Subset containing all episodes from this client's assigned category.
    """
    set_seed(seed)

    # Step 1: Build episode_to_task mapping
    episode_to_task = {}
    if hasattr(dataset, 'datasets'):
        ep_counter = 0
        for ds in dataset.datasets:
            repo_id = getattr(ds, 'repo_id', '')
            parts = repo_id.split('/')
            task_name = parts[1] if len(parts) >= 2 else repo_id

            num_eps = getattr(ds, 'num_episodes', 0)
            for ep_idx in range(num_eps):
                episode_to_task[ep_counter + ep_idx] = task_name

            ep_counter += num_eps
    else:
        raise NotImplementedError("Only MultiLeRobotDataset is supported")

    # Step 2: Get all categories and their tasks
    task_categories = get_robottwin_tasks_by_category()
    category_names = list(task_categories.keys())

    # Step 3: Assign each client a unique category (round-robin if num_clients > 7)
    assigned_category = category_names[client_id % len(category_names)]

    # Step 4: Get all episodes for this category
    category_tasks = task_categories[assigned_category]
    my_episode_ids = []
    for ep_idx, task in episode_to_task.items():
        if task in category_tasks:
            my_episode_ids.append(ep_idx)

    # Step 5: Convert episode indices to frame indices
    my_frame_indices = []
    from_index = dataset.meta.episodes["dataset_from_index"]
    to_index = dataset.meta.episodes["dataset_to_index"]

    for ep_idx in my_episode_ids:
        start_frame = from_index[ep_idx]
        end_frame = to_index[ep_idx]
        my_frame_indices.extend(range(start_frame, end_frame))

    num_frames = len(my_frame_indices)

    # Print category assignments (only from client 0)
    if client_id == 0:
        print(f"[RoboTwinFL] ===== CATEGORY ASSIGNMENTS (by_category) =====")
        for i in range(num_clients):
            cat = category_names[i % len(category_names)]
            num_tasks = len(task_categories[cat])
            print(f"[RoboTwinFL] Client {i}: category={cat}, tasks={num_tasks}")
        print(f"[RoboTwinFL] ============================")

    print(f"[RoboTwinFL] Client {client_id}: category={assigned_category}, "
          f"tasks={len(category_tasks)}, frames={num_frames}")

    return Subset(dataset, my_frame_indices)


def partition_robottwin_by_task(
    dataset,
    num_clients: int,
    client_id: int,
    tasks_per_client: int | None = None,
    held_out_tasks: int = 0,
    seed: int = 42,
) -> Subset:
    """
    Partition RoboTwin dataset by task.
    Returns Subset with frame indices, so len() returns frame count.

    Args:
        dataset: The full RoboTwin MultiLeRobotDataset.
        num_clients: Total number of clients.
        client_id: This client's ID.
        tasks_per_client: Deprecated, unused.
        held_out_tasks: Deprecated, unused.
        seed: Random seed.

    Returns:
        Subset containing frame indices from assigned tasks.
        len() returns total frames for this client's partition.
    """
    set_seed(seed)

    # Step 1: Build episode_to_task mapping from repo_id
    episode_to_task = {}
    if hasattr(dataset, 'datasets'):
        ep_counter = 0
        for ds in dataset.datasets:
            repo_id = getattr(ds, 'repo_id', '')
            parts = repo_id.split('/')
            task_name = parts[1] if len(parts) >= 2 else repo_id

            num_eps = getattr(ds, 'num_episodes', 0)
            for ep_idx in range(num_eps):
                episode_to_task[ep_counter + ep_idx] = task_name

            ep_counter += num_eps
    else:
        raise NotImplementedError("Only MultiLeRobotDataset is supported")

    # Step 2: Build task_to_episodes mapping
    task_to_episodes = {}
    for ep_idx, task in episode_to_task.items():
        if task not in task_to_episodes:
            task_to_episodes[task] = []
        task_to_episodes[task].append(ep_idx)

    # Step 3: Category-balanced task assignment
    task_categories = get_robottwin_tasks_by_category()
    assignments = {i: [] for i in range(num_clients)}
    client_task_counts = {i: 0 for i in range(num_clients)}

    for category, tasks in task_categories.items():
        random.shuffle(tasks)
        for task in tasks:
            if task not in task_to_episodes:
                continue
            min_client = min(range(num_clients), key=lambda c: client_task_counts[c])
            assignments[min_client].append(task)
            client_task_counts[min_client] += 1

    # Step 4: Get current client's episode indices
    my_tasks = assignments[client_id]
    my_episode_ids = []
    for t in my_tasks:
        my_episode_ids.extend(task_to_episodes[t])

    # Step 5: Convert episode indices to frame indices
    my_frame_indices = []
    from_index = dataset.meta.episodes["dataset_from_index"]
    to_index = dataset.meta.episodes["dataset_to_index"]

    for ep_idx in my_episode_ids:
        start_frame = from_index[ep_idx]
        end_frame = to_index[ep_idx]
        my_frame_indices.extend(range(start_frame, end_frame))

    num_frames = len(my_frame_indices)

    # Print all client assignments and frames (only from client 0)
    if client_id == 0:
        print(f"[RoboTwinFL] ===== TASK ASSIGNMENTS =====")
        for i in range(num_clients):
            print(f"[RoboTwinFL] Client {i}: {len(assignments[i])} tasks -> {assignments[i]}")
        print(f"[RoboTwinFL] ============================")

    # Each client prints its own frames
    print(f"[RoboTwinFL] Client {client_id}: {len(my_tasks)} tasks, {num_frames} frames")

    return Subset(dataset, my_frame_indices)


def partition_robottwin_by_episode(
    dataset,
    num_clients: int,
    client_id: int,
    seed: int = 42,
    shuffle: bool = True,
) -> Subset:
    """
    Partition RoboTwin dataset by episode (round-robin).

    Each client gets episodes from all tasks in round-robin fashion.

    Args:
        dataset: The full RoboTwin LeRobot dataset.
        num_clients: Total number of clients.
        client_id: This client's ID.
        seed: Random seed.
        shuffle: Whether to shuffle episodes.

    Returns:
        Subset containing this client's episodes.
    """
    print(f"[Debug partition_by_episode] num_clients={num_clients}, client_id={client_id}, seed={seed}")

    set_seed(seed)

    num_episodes = len(dataset)
    print(f"[Debug partition_by_episode] total episodes={num_episodes}")

    indices = list(range(num_episodes))

    if shuffle:
        random.shuffle(indices)

    # Round-robin assignment
    client_indices = [idx for idx in indices if idx % num_clients == client_id]

    print(f"[Debug partition_by_episode] client {client_id} gets {len(client_indices)} episodes")
    print(f"[Debug partition_by_episode] sample indices: {client_indices[:10]}")

    # Debug: Check if dataset has task info
    if hasattr(dataset, '_episode_task_names'):
        task_names = [dataset._episode_task_names[idx] for idx in client_indices[:20]]
        print(f"[Debug partition_by_episode] sample task names (first 20): {task_names}")
        # Count clean vs randomized
        clean_count = sum(1 for t in task_names if 'clean' in t.lower())
        randomized_count = sum(1 for t in task_names if 'randomized' in t.lower())
        print(f"[Debug partition_by_episode] first 20: clean={clean_count}, randomized={randomized_count}")

    logging.info(
        f"[RoboTwinFL] Client {client_id}: {len(client_indices)} episodes "
        f"(round-robin from {num_episodes} total)"
    )

    return Subset(dataset, client_indices)


def partition_robottwin_dirichlet(
    dataset,
    num_clients: int,
    client_id: int,
    alpha: float = 0.5,
    seed: int = 42,
    min_episodes_per_client: int = 10,
) -> Subset:
    """
    Partition RoboTwin dataset using Dirichlet distribution.

    Creates non-IID data distribution where each client has biased
    task preferences but still sees all tasks to some degree.

    Args:
        dataset: The full RoboTwin LeRobot dataset.
        num_clients: Total number of clients.
        client_id: This client's ID.
        alpha: Dirichlet concentration parameter.
            - Smaller alpha = more heterogeneous (client bias).
            - Larger alpha = more homogeneous (balanced).
        seed: Random seed.
        min_episodes_per_client: Minimum episodes per client.

    Returns:
        Subset with Dirichlet-weighted episode distribution.
    """
    set_seed(seed)

    num_episodes = len(dataset)

    # Sample Dirichlet for each client
    proportions = np.random.dirichlet([alpha] * num_clients)

    # Assign episodes based on proportions
    client_episode_counts = [max(min_episodes_per_client, int(p * num_episodes))
                            for p in proportions]

    # Normalize to exact total
    total_assigned = sum(client_episode_counts)
    client_episode_counts = [int(c / total_assigned * num_episodes)
                            for c in client_episode_counts]

    # Distribute episodes
    indices = list(range(num_episodes))
    random.shuffle(indices)

    client_indices = []
    start_idx = 0
    for c in range(num_clients):
        count = client_episode_counts[c]
        if c == client_id:
            client_indices = indices[start_idx:start_idx + count]
        start_idx += count

    logging.info(
        f"[RoboTwinFL] Client {client_id}: {len(client_indices)} episodes "
        f"(Dirichlet alpha={alpha})"
    )

    return Subset(dataset, client_indices)


def partition_robottwin_by_source(
    dataset,
    num_clients: int,
    client_id: int,
    data_sources: list[str] | None = None,
    seed: int = 42,
) -> Subset:
    """
    Partition RoboTwin dataset by data source (robot platform).

    Simulates federated learning across heterogeneous robot platforms.

    Args:
        dataset: The full RoboTwin LeRobot dataset.
        num_clients: Total number of clients.
        client_id: This client's ID.
        data_sources: List of data source identifiers.
        seed: Random seed.

    Returns:
        Subset containing this client's source data.
    """
    set_seed(seed)

    if data_sources is None:
        # Infer from dataset metadata or use default
        data_sources = ["aloha_agilex_1", "aloha_agilex_2", "aloha_mobile", "aloha_real"]

    # Assign sources to clients
    source_assignments = {}
    for idx, source in enumerate(data_sources):
        source_assignments[source] = idx % num_clients

    # Filter episodes for this client's sources
    client_sources = [s for s, c in source_assignments.items() if c == client_id]

    # In practice, this would filter by source metadata
    # For now, return full dataset with source tracking
    logging.info(
        f"[RoboTwinFL] Client {client_id}: sources = {client_sources}"
    )

    # Return full dataset - actual filtering happens at data loading
    return dataset


def partition_robottwin_mixed(
    dataset,
    num_clients: int,
    client_id: int,
    seed: int = 42,
    category_weight: float = 0.7,
) -> Subset:
    """
    Mixed partitioning: 70% category-balanced, 30% random.

    Creates realistic heterogeneity where each client has:
    - Primary category expertise (70% of data)
    - Diverse exposure to other categories (30% of data)

    Args:
        dataset: The full RoboTwin LeRobot dataset.
        num_clients: Total number of clients.
        client_id: This client's ID.
        seed: Random seed.
        category_weight: Weight for category-focused data.

    Returns:
        Mixed subset for this client.
    """
    set_seed(seed)

    task_categories = get_robottwin_tasks_by_category()
    category_names = list(task_categories.keys())

    # Assign primary category to this client
    primary_category = category_names[client_id % len(category_names)]
    secondary_categories = [c for c in category_names if c != primary_category]
    random.shuffle(secondary_categories)

    # Collect episodes
    primary_tasks = task_categories[primary_category]
    secondary_tasks = []
    for cat in secondary_categories[:2]:
        secondary_tasks.extend(task_categories[cat][:2])

    # Build episode list
    client_episodes = []

    # Add primary category episodes (70%)
    for task in primary_tasks:
        if hasattr(dataset, 'task_episodes'):
            client_episodes.extend(dataset.task_episodes.get(task, []))
    # Note: actual implementation depends on dataset structure

    logging.info(
        f"[RoboTwinFL] Client {client_id}: primary={primary_category}, "
        f"mixed with {len(secondary_tasks)} other tasks"
    )

    # Return full dataset for now - actual partitioning depends on metadata
    return dataset


def create_robottwin_partitioner(
    strategy: Literal[
        "by_task", "by_category", "by_episode", "dirichlet", "by_source", "mixed",
        "fedforesight_category", "fedforesight_server",
    ] = "by_task",
    **kwargs,
) -> Callable:
    """
    Factory function to create a RoboTwin-specific dataset partitioner.

    Args:
        strategy: Partitioning strategy.
            - "by_task": Category-balanced task distribution (each client has mixed categories)
            - "by_category": Each client gets ONE full category (specialization)
            - "by_episode": Round-robin episode distribution
            - "dirichlet": Dirichlet-based non-IID distribution
            - "by_source": Partition by data source robot
            - "mixed": 70% category-focused, 30% mixed
            - "fedforesight_category": 8 category-siloed client partitions
            - "fedforesight_server": residual mixed-task server partition
        **kwargs: Strategy-specific arguments.

    Returns:
        A partitioner function.
    """
    if strategy == "by_task":
        return lambda ds, nc, cid, s: partition_robottwin_by_task(
            ds, nc, cid,
            tasks_per_client=kwargs.get("tasks_per_client"),
            held_out_tasks=kwargs.get("held_out_tasks", 0),
            seed=s,
        )
    elif strategy == "by_category":
        return lambda ds, nc, cid, s: partition_robottwin_by_category(
            ds, nc, cid, seed=s,
        )
    elif strategy == "by_episode":
        return partition_robottwin_by_episode
    elif strategy == "dirichlet":
        return lambda ds, nc, cid, s: partition_robottwin_dirichlet(
            ds, nc, cid, alpha=kwargs.get("alpha", 0.5), seed=s
        )
    elif strategy == "by_source":
        return lambda ds, nc, cid, s: partition_robottwin_by_source(
            ds, nc, cid, data_sources=kwargs.get("data_sources"), seed=s
        )
    elif strategy == "mixed":
        return lambda ds, nc, cid, s: partition_robottwin_mixed(
            ds, nc, cid, seed=s, category_weight=kwargs.get("category_weight", 0.7)
        )
    elif strategy == "fedforesight_category":
        return lambda ds, nc, cid, s: partition_robottwin_by_fedforesight_category(
            ds, nc, cid, seed=s,
        )
    elif strategy == "fedforesight_server":
        return lambda ds, nc, cid, s: partition_robottwin_fedforesight_server_residual(
            ds, seed=s,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def get_client_task_info(
    dataset,
    client_id: int,
    num_clients: int,
    partition_strategy: str = "by_task",
    seed: int = 42,
) -> dict:
    """
    Get information about tasks available to a client.

    Useful for logging and curriculum learning.

    Returns:
        Dict with task names, episode counts, and categories.
    """
    task_categories = get_robottwin_tasks_by_category()

    if partition_strategy == "by_task":
        # Get assigned tasks
        assignments = RoboTwinTaskAssignment.partition_by_task(
            num_clients, tasks_per_client=None, seed=seed
        )["assignments"]
        client_tasks = assignments.get(client_id, [])

        return {
            "client_id": client_id,
            "num_tasks": len(client_tasks),
            "tasks": client_tasks,
            "categories": list(set(
                cat for cat, tasks in task_categories.items()
                for t in client_tasks if t in tasks
            )),
        }
    else:
        return {
            "client_id": client_id,
            "num_tasks": len(ROBOTWIN_TASKS),
            "tasks": ROBOTWIN_TASKS,
            "categories": list(task_categories.keys()),
        }


class RoboTwinTaskAssignment:
    """Utility class for task assignment (kept for backward compatibility)."""

    task_categories = get_robottwin_tasks_by_category()

    @classmethod
    def get_all_tasks(cls) -> list:
        return ROBOTWIN_TASKS
