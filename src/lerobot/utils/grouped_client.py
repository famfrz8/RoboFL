"""Multi-rank client training without changing the global server process group."""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler


def validate_client_gpus(client_gpus, world_size, num_clients, num_experts, batch_size):
    if client_gpus not in (1, 2, 4):
        raise ValueError("FL_CLIENT_GPUS must be 1, 2 or 4")
    if client_gpus == 1:
        return
    if world_size != client_gpus * num_clients:
        raise ValueError("FL_CLIENT_GPUS>1 requires WORLD_SIZE == FL_CLIENT_GPUS * FL_NUM_CLIENTS")
    if num_experts != num_clients:
        raise ValueError("FL_CLIENT_GPUS>1 requires loramoe_num_experts == FL_NUM_CLIENTS")
    if batch_size <= 0 or batch_size % client_gpus:
        raise ValueError("FL_CLIENT_GPUS>1 requires a positive per-client batch_size divisible by FL_CLIENT_GPUS")


def validate_sequential_clients(world_size, num_clients, num_experts, batch_size):
    """Validate the sequential mode where every client uses all ranks in turn.

    No GPU is dedicated to a client: each client is trained alone by the whole
    process group, then the next client, then the server. Only one server expert
    slot per client is required, matching the existing LoRA-MoE mapping.
    """
    if world_size <= 0:
        raise ValueError("Sequential clients require a positive world_size")
    if num_clients != num_experts:
        raise ValueError("Sequential clients require FL_NUM_CLIENTS == loramoe_num_experts")
    if num_clients < 1:
        raise ValueError("Sequential clients require FL_NUM_CLIENTS >= 1")
    if batch_size <= 0:
        raise ValueError("Sequential clients require a positive per-rank cfg.batch_size")


def global_client_sampler(dataset, rank, world_size, seed):
    """Shard one client partition across every rank for all-GPU sequential training."""
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    return DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, seed=seed, drop_last=False
    )


def create_client_group(num_clients=4, client_gpus=2):
    """Call once on ALL ranks, in the same order, before any client work.

    Interleaving keeps leaders 0..num_clients-1, preserving
    expert_id % world_size as the existing global expert broadcast source.
    The default group is untouched. Defaults preserve the original two-GPU,
    four-client interleaved layout.
    """
    if dist.get_world_size() != num_clients * client_gpus:
        raise ValueError("Client groups require num_clients * client_gpus global ranks")
    rank = dist.get_rank()
    client_id = rank % num_clients
    group_rank = rank // num_clients
    client_group = None
    for leader in range(num_clients):
        group = dist.new_group(ranks=[leader + i * num_clients for i in range(client_gpus)])
        if leader == client_id:
            client_group = group
    return client_group, client_id, group_rank


def wrap_client_policy(policy, group):
    device = next(policy.parameters()).device
    return DistributedDataParallel(
        policy,
        device_ids=[device.index] if device.type == "cuda" else None,
        process_group=group,
        find_unused_parameters=True,
    )


def unwrap_client_policy(policy):
    return policy.module if isinstance(policy, DistributedDataParallel) else policy


def client_sampler(dataset, group_rank, seed, num_replicas=2):
    # Pad an odd-sized partition so all group ranks always see equally sized batches.
    return DistributedSampler(dataset, num_replicas=num_replicas, rank=group_rank, seed=seed, drop_last=False)


def cycle_client_batches(loader):
    """Advance the shared sampler epoch on every pass, including across rounds."""
    if len(loader) == 0:
        raise ValueError("Grouped client dataset must not be empty")
    epoch = 0
    while True:
        loader.sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def average_client_metrics(metrics, group, device):
    keys = sorted(metrics)
    values = torch.tensor([float(metrics[key]) for key in keys], dtype=torch.float64, device=device)
    dist.all_reduce(values, group=group)
    values /= dist.get_world_size(group)
    return dict(zip(keys, values.tolist()))
