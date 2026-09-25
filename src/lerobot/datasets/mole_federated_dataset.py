"""Episode-whole partitions for the approved MoLe ten-task training split."""

import json
from pathlib import Path

from torch.utils.data import Subset


MOLE_TASK_COUNTS = (
    (94, 375), (86, 344), (92, 276), (92, 368), (86, 344),
    (94, 188), (85, 424), (91, 455), (89, 356), (91, 357),
)


def build_mole_partition_manifest(dataset, num_clients=4) -> dict:
    """Validate metadata against actual frame indices, without decoding images.

    The single transformed wrapper shares the base LeRobotDataset's metadata and
    frame table. Filtered datasets must not use absolute metadata ranges as local
    Subset indices, so reject them rather than silently assigning wrong frames.
    """
    if num_clients not in (4, 8):
        raise ValueError("mole_by_task requires four or eight clients")
    if hasattr(dataset, "datasets") or getattr(dataset, "episodes", None) is not None:
        raise ValueError("mole_by_task requires one complete, unfiltered map-style dataset")
    meta = dataset.meta
    if meta.robot_type != "mole_rlbench_single_view":
        raise ValueError("mole_by_task requires robot_type=mole_rlbench_single_view")
    if len(dataset) != 3487 or meta.total_episodes != 900:
        raise ValueError("MoLe training split must contain 900 episodes and 3487 frames")
    tasks = {int(row.task_index): str(language) for language, row in meta.tasks.iterrows()}
    if len(meta.tasks) != 10 or set(tasks) != set(range(10)) or len(set(tasks.values())) != 10:
        raise ValueError("MoLe tasks.parquet must map exactly ten unique instructions to indices 0..9")
    task_indices = {language: tid for tid, language in tasks.items()}
    groups = [dict(task_index=tid, language=tasks[tid], episode_ids=[], frame_indices=[]) for tid in range(10)]
    frames = dataset.hf_dataset.select_columns(["index", "episode_index", "task_index"]).with_format(None)
    frame_ids = frames["index"]
    episode_ids = frames["episode_index"]
    frame_tasks = frames["task_index"]
    if list(frame_ids) != list(range(len(dataset))):
        raise ValueError("MoLe frame table is filtered or reindexed")
    seen_episodes, seen_frames = set(), set()
    for ep in meta.episodes:
        eid = int(ep["episode_index"])
        languages = list(ep["tasks"])
        if len(languages) != 1 or languages[0] not in task_indices:
            raise ValueError(f"Episode {eid} must have one exact tasks.parquet instruction")
        tid = task_indices[languages[0]]
        start, end = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
        if eid in seen_episodes or not 0 <= start < end <= len(dataset) or end - start != int(ep["length"]):
            raise ValueError(f"Invalid or duplicate episode range: {eid}")
        indices = list(range(start, end))
        if seen_frames.intersection(indices):
            raise ValueError(f"Overlapping episode range: {eid}")
        if any(int(episode_ids[i]) != eid or int(frame_tasks[i]) != tid for i in indices):
            raise ValueError(f"Frame table disagrees with episode/task metadata: {eid}")
        seen_episodes.add(eid)
        seen_frames.update(indices)
        groups[tid]["episode_ids"].append(eid)
        groups[tid]["frame_indices"].extend(indices)
    if seen_frames != set(range(len(dataset))) or seen_episodes != set(range(900)):
        raise ValueError("MoLe partitions must cover every training episode and frame exactly once")
    for group, expected in zip(groups, MOLE_TASK_COUNTS, strict=True):
        group["num_episodes"] = len(group["episode_ids"])
        group["num_frames"] = len(group["frame_indices"])
        if (group["num_episodes"], group["num_frames"]) != expected:
            raise ValueError(f"Unexpected counts for task {group['task_index']}: {group['num_episodes']}/{group['num_frames']}")
    # Eight clients split each four-client task pair, keeping the same server data.
    client_assignments = [[2, 9], [1, 5], [3, 6], [4, 8]]
    if num_clients == 8:
        client_assignments = [[tid] for tids in client_assignments for tid in tids]
    server_task_indices = [0, 7]
    server_episodes = sum(groups[t]["num_episodes"] for t in server_task_indices)
    server_frames = sum(groups[t]["num_frames"] for t in server_task_indices)
    return {
        "strategy": "mole_by_task",
        "dataset_root": str(Path(dataset.root).resolve()),
        "repo_id": dataset.repo_id,
        "num_episodes": 900,
        "num_frames": 3487,
        "tasks": groups,
        "clients": {str(cid): tids for cid, tids in enumerate(client_assignments)},
        "server": {"task_indices": server_task_indices, "num_episodes": server_episodes, "num_frames": server_frames},
    }


def partition_mole_client(dataset, num_clients, client_id, seed=42, *, manifest=None):
    if num_clients not in (4, 8):
        raise ValueError("mole_by_task requires four or eight clients")
    if client_id not in range(num_clients):
        raise ValueError(f"mole_by_task requires client IDs 0..{num_clients - 1}")
    manifest = manifest if manifest is not None else build_mole_partition_manifest(dataset, num_clients=num_clients)
    if set(manifest["clients"]) != {str(cid) for cid in range(num_clients)}:
        raise ValueError(f"MoLe manifest client IDs do not match num_clients={num_clients}")
    return Subset(dataset, [i for tid in manifest["clients"][str(client_id)]
                            for i in manifest["tasks"][tid]["frame_indices"]])


def partition_mole_server(dataset, *, manifest=None):
    manifest = manifest if manifest is not None else build_mole_partition_manifest(dataset)
    return Subset(dataset, [i for tid in manifest["server"]["task_indices"]
                            for i in manifest["tasks"][tid]["frame_indices"]])


def create_mole_partitioner(*, manifest=None):
    return lambda ds, nc, cid, seed: partition_mole_client(ds, nc, cid, seed, manifest=manifest)


def persist_mole_partition_manifest(manifest, output_dir):
    """Called only by rank zero; never overwrite an existing run's assignment."""
    path = Path(output_dir) / "mole_partition_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError(f"Existing MoLe partition manifest differs: {path}")
        return
    with path.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
