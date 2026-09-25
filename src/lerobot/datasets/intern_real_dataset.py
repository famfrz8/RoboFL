"""Read-only six-source Intern real-robot training and source-whole FL split."""

import hashlib
import json
import os
from pathlib import Path

import draccus
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Subset

from lerobot.datasets.transformed_dataset import MultiLeRobotDataset, TransformedLeRobotDataset

ROBOT_TYPE = "intern_gello_7dof_robotiq"
# This order defines stable global task IDs, independent of each source's local ID 0.
SOURCES = (("debug", 51, 43257), ("screw", 50, 31960), ("seal", 110, 81892),
           ("shiguan", 51, 26665), ("bottle", 100, 49929), ("pour", 51, 44042))
KEYS = ("observation.state", "action")
CAMERAS = ("observation.images.cam_high", "observation.images.cam_front")

# Three semantic classes for the two-client + one-server layout, keyed by the
# stable global task IDs above. Client 0 handles bottle-body tasks, client 1
# handles precise insertion, and the server owns cup/container pouring.
INTERN_REAL_CLASSES = (
    ("bottle_body", (1, 4)),        # client 0: screw + bottle
    ("precise_insert", (3, 2)),     # client 1: shiguan + seal
    ("cup_container", (0, 5)),      # server: debug + pour
)
INTERN_REAL_CLASS_CLIENTS = {
    str(client_id): list(task_ids)
    for client_id, (_, task_ids) in enumerate(INTERN_REAL_CLASSES[:2])
}
INTERN_REAL_CLASS_SERVER = list(INTERN_REAL_CLASSES[2][1])


def validate_intern_real_config(cfg):
    if (cfg.dataset.action_mode != "abs" or cfg.dataset.streaming or cfg.dataset.dist_loading
            or cfg.dataset.episodes is not None or cfg.dataset.use_external_stats):
        raise ValueError("intern_real_by_task requires abs, complete map-style data, no external stats/dist_loading")
    if cfg.resume:
        raise ValueError("Federated resume is not implemented; use a new output directory")
    if not 1 <= cfg.policy.n_action_steps <= cfg.policy.chunk_size:
        raise ValueError("Require 1 <= n_action_steps <= chunk_size")
    if cfg.policy.n_obs_steps != 1:
        raise ValueError("InternVLA requires n_obs_steps=1")
    if (cfg.policy.max_state_dim, cfg.policy.max_action_dim) != (32, 32):
        raise ValueError("Keep internal state/action padding at 32")
    if (cfg.dataset.height, cfg.dataset.width, cfg.dataset.max_state_dim, cfg.dataset.max_action_dim) != (224, 224, 32, 32):
        raise ValueError("Real-robot baseline requires 224 RGB transforms and internal padding 32")


def source_roots(cfg):
    parent = Path(cfg.dataset.root or os.path.expanduser("~"))
    overrides = json.loads(os.environ.get("INTERN_REAL_ROOTS_JSON", "{}"))
    if set(overrides) - {name for name, _, _ in SOURCES}:
        raise ValueError("Unknown source in INTERN_REAL_ROOTS_JSON")
    roots = {name: Path(overrides.get(name, parent / f"intern-{name}-lerobot")).resolve()
             for name, _, _ in SOURCES}
    if len(set(roots.values())) != 6:
        raise ValueError("Six distinct source roots are required")
    return roots


def inspect_sources(roots):
    """Validate frame/episode tables and compute population moments in float64.

    Only canonical numeric columns are read, never auxiliary wrench/tactile data.
    Source revisions hash metadata and parquet bytes; videos are recorded by stat.
    """
    moments = {key: [] for key in KEYS}
    groups, offset, episode_offset = [], 0, 0
    for tid, (name, expected_eps, expected_frames) in enumerate(SOURCES):
        root = roots[name]
        info = json.loads((root / "meta/info.json").read_text())
        if (info["robot_type"], info["total_episodes"], info["total_frames"]) != (
                ROBOT_TYPE, expected_eps, expected_frames):
            raise ValueError(f"Unexpected source/counts: {name}")
        if any(info["features"][key]["shape"] != [8] for key in KEYS):
            raise ValueError(f"Canonical dimensions must be 8: {name}")
        tasks = pq.read_table(root / "meta/tasks.parquet").to_pandas()
        if len(tasks) != 1 or int(tasks.iloc[0]["task_index"]) != 0:
            raise ValueError(f"Expected one local task 0: {name}")
        language = str(tasks.index[0])
        files = sorted((root / "data").rglob("*.parquet"))
        table = pq.read_table(files, columns=[*KEYS, "index", "episode_index", "task_index"])
        indices = np.asarray(table["index"])
        ep_ids = np.asarray(table["episode_index"])
        if not np.array_equal(indices, np.arange(expected_frames)) or np.any(np.asarray(table["task_index"]) != 0):
            raise ValueError(f"Noncontiguous frame indices or task collision: {name}")
        episodes = pq.read_table(sorted((root / "meta/episodes").rglob("*.parquet"))).to_pylist()
        episodes.sort(key=lambda ep: ep["episode_index"])
        cursor = 0
        ranges = []
        for eid, ep in enumerate(episodes):
            start, end = int(ep["dataset_from_index"]), int(ep["dataset_to_index"])
            if (ep["episode_index"] != eid or start != cursor or end <= start
                    or end - start != ep["length"] or ep["tasks"] != [language]
                    or not np.all(ep_ids[start:end] == eid)):
                raise ValueError(f"Invalid episode/task range: {name}/{eid}")
            ranges.append([start + offset, end + offset])
            cursor = end
        if cursor != expected_frames or len(episodes) != expected_eps:
            raise ValueError(f"Incomplete episode coverage: {name}")
        for key in KEYS:
            values = np.asarray(table[key].to_pylist(), dtype=np.float64)
            if values.shape != (expected_frames, 8) or not np.isfinite(values).all():
                raise ValueError(f"Invalid canonical values: {name}/{key}")
            if key == "action" and ((values[:, 7] < 0).any() or (values[:, 7] > 1).any()):
                raise ValueError(f"Action gripper is not an open fraction: {name}")
            moments[key].append((len(values), values.mean(0), values.var(0), values.min(0), values.max(0)))
        digest = hashlib.sha256()
        for path in sorted([*files, *(root / "meta").rglob("*.json"), *(root / "meta").rglob("*.parquet")]):
            digest.update(str(path.relative_to(root)).encode())
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        for camera in CAMERAS:
            if not list((root / "videos" / camera).rglob("*.mp4")):
                raise ValueError(f"Missing RGB videos: {name}/{camera}")
        videos = [{"path": str(p.relative_to(root)), "size": p.stat().st_size,
                   "mtime_ns": p.stat().st_mtime_ns}
                  for camera in CAMERAS for p in sorted((root / "videos" / camera).rglob("*.mp4"))]
        groups.append(dict(source=name, repo_id=f"intern-{name}-lerobot", root=str(root),
                           revision_sha256=digest.hexdigest(), videos=videos, task_index=tid,
                           local_task_index=0, language=language, num_episodes=expected_eps,
                           num_frames=expected_frames, frame_range=[offset, offset + expected_frames],
                           episode_offset=episode_offset, episode_ranges=ranges))
        offset += expected_frames
        episode_offset += expected_eps
    stats = {}
    for key, parts in moments.items():
        count = sum(p[0] for p in parts)
        mean = sum(n * m for n, m, _, _, _ in parts) / count
        variance = sum(n * (v + (m - mean) ** 2) for n, m, v, _, _ in parts) / count
        stats[key] = dict(mean=mean, std=np.sqrt(variance), count=np.array([count]),
                          min=np.minimum.reduce([p[3] for p in parts]),
                          max=np.maximum.reduce([p[4] for p in parts]))
    manifest = dict(strategy="intern_real_by_task", robot_type=ROBOT_TYPE, num_frames=offset,
                    num_episodes=episode_offset, validation=[], tasks=groups,
                    clients={str(i): [i] for i in range(4)},
                    server=dict(task_indices=[4, 5], num_episodes=151, num_frames=93971),
                    classes={name: list(task_ids) for name, task_ids in INTERN_REAL_CLASSES},
                    class_clients=INTERN_REAL_CLASS_CLIENTS,
                    class_server=INTERN_REAL_CLASS_SERVER)
    return manifest, stats


class InternRealDataset(MultiLeRobotDataset):
    def __getitem__(self, idx):
        tid, local_idx = self._locate_dataset(idx)
        sample = self.datasets[tid][local_idx]
        sample["source_id"] = self.manifest["tasks"][tid]["source"]
        sample["task_index"] = torch.tensor(tid)
        sample["task"] = self.manifest["tasks"][tid]["language"]
        return sample


def build_intern_real_dataset(cfg):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.transforms import ImageTransforms
    from lerobot.transforms.core import DeltaActionTransformFn, NormalizeTransformFn, PadStateAndActionTransformFn

    validate_intern_real_config(cfg)
    transforms = cfg.dataset.data_transforms.inputs
    norm = [i for i, t in enumerate(transforms) if isinstance(t, NormalizeTransformFn)]
    pad = [i for i, t in enumerate(transforms) if isinstance(t, PadStateAndActionTransformFn)]
    if not norm or not pad or max(norm) >= min(pad) or any(isinstance(t, DeltaActionTransformFn) for t in transforms):
        raise ValueError("Require normalization before padding and no delta transform")
    manifest, stats = inspect_sources(source_roots(cfg))
    cfg.policy.input_features["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(8,))
    cfg.policy.output_features["action"] = PolicyFeature(type=FeatureType.ACTION, shape=(8,))
    manifest["configuration"] = dict(action_mode="abs", chunk_size=cfg.policy.chunk_size,
        n_action_steps=cfg.policy.n_action_steps,
        image_history_stride=cfg.policy.image_history_stride,
        cameras=list(CAMERAS),
        pretrained_path=str(cfg.policy.pretrained_path), policy=draccus.encode(cfg.policy),
        dataset=draccus.encode(cfg.dataset), seed=cfg.seed, batch_size=cfg.batch_size,
        steps=cfg.steps, num_workers=cfg.num_workers,
        training_env={k: v for k, v in sorted(os.environ.items())
                      if k.startswith(("FL_", "MOE_", "ENABLE_", "LAMBDA_", "FARD_"))})
    datasets = []
    for group in manifest["tasks"]:
        info = json.loads((Path(group["root"]) / "meta/info.json").read_text())
        fps = info["fps"]
        deltas = {"action": [i / fps for i in cfg.policy.action_delta_indices],
                  **{key: [i / fps for i in cfg.policy.image_delta_indices] for key in CAMERAS}}
        base = LeRobotDataset(group["repo_id"], root=group["root"], delta_timestamps=deltas,
                             revision=cfg.dataset.revision, video_backend=cfg.dataset.video_backend,
                             image_transforms=ImageTransforms(cfg.dataset.image_transforms)
                             if cfg.dataset.image_transforms.enable else None)
        base.meta.stats = stats
        # Remove auxiliary numeric fields before fetching samples, and non-RGB video
        # metadata before the decoder enumerates cameras. Source files stay untouched.
        allowed = {*KEYS, *CAMERAS, "index", "episode_index", "task_index", "frame_index", "timestamp"}
        base.meta.info["features"] = {k: v for k, v in base.meta.features.items() if k in allowed}
        base.hf_dataset = base.hf_dataset.select_columns([k for k in base.hf_dataset.column_names if k in allowed])
        datasets.append(TransformedLeRobotDataset.from_base(base, transforms))
    dataset = InternRealDataset(datasets)
    dataset.manifest = manifest
    dataset.meta.stats = stats
    dataset.meta.robot_type = ROBOT_TYPE
    return dataset, {ROBOT_TYPE: stats}


def partition_intern_real(dataset, num_clients=4, client_id=None, seed=42):
    """Source-whole partition for either the four-client or three-class layout.

    ``num_clients=4`` keeps the original one-source-per-client split with the
    bottle+pour server. ``num_clients=2`` returns the semantic classes: client 0
    (screw+bottle), client 1 (shiguan+seal), and the debug+pour server.
    """
    if num_clients == 4:
        if client_id is not None and client_id not in range(4):
            raise ValueError("intern_real_by_task requires four clients, IDs 0..3")
        tids = [4, 5] if client_id is None else [client_id]
    elif num_clients == 2:
        if client_id is not None and client_id not in range(2):
            raise ValueError("intern_real class partition requires two clients, IDs 0..1")
        tids = INTERN_REAL_CLASS_SERVER if client_id is None else INTERN_REAL_CLASS_CLIENTS[str(client_id)]
    else:
        raise ValueError("intern_real partitioning supports num_clients in {2, 4}")
    return Subset(dataset, [i for tid in tids for i in range(*dataset.manifest["tasks"][tid]["frame_range"])])


def persist_intern_real_artifacts(dataset, output_dir):
    """Rank-zero only. Compare both artifacts before writing either one."""
    root = Path(output_dir)
    stats = {ROBOT_TYPE: {k: {s: v.tolist() for s, v in values.items()}
                          for k, values in dataset.meta.stats.items()}}
    artifacts = {"intern_real_manifest.json": dataset.manifest, "stats.json": stats}
    artifacts = {name: json.loads(json.dumps(value)) for name, value in artifacts.items()}
    for name, value in artifacts.items():
        path = root / name
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f"Existing real-robot run artifact differs: {path}")
    root.mkdir(parents=True, exist_ok=True)
    for name, value in artifacts.items():
        path = root / name
        if not path.exists():
            with path.open("x") as stream:
                json.dump(value, stream, indent=2)
                stream.write("\n")
