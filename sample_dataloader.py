#!/usr/bin/env python3
"""PyTorch dataloader for exported Open-H surgical LeRobot v2.1 datasets.

This script builds a frame-level training dataset from one or many exported
LeRobot roots (e.g. under ./KnotTying and ./Suturing), then demonstrates
loading batches via torch.utils.data.DataLoader.

Sample usage:
  python sample_dataloader.py \
    --scan-roots ./KnotTying ./Suturing \
    --batch-size 4 \
    --num-workers 2 \
    --max-batches 2

  python sample_dataloader.py \
    --dataset-root ./KnotTying/1/34a9fb79cd \
    --batch-size 8 \
    --shuffle
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError as e:
    raise RuntimeError("PyTorch is required. Install with: pip install torch") from e


VIDEO_KEYS = [
    "observation.images.left",
    "observation.images.right",
    "observation.images.side",
    "observation.images.gaze",
]

REQUIRED_COLUMNS = [
    "index",
    "episode_index",
    "frame_index",
    "timestamp",
    "task_index",
    "next.done",
    "action",
    "observation.state",
]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_import_pandas():
    try:
        import pandas as pd  # type: ignore

        return pd
    except ImportError as e:
        raise RuntimeError(
            "pandas and pyarrow are required to read episode parquet files. "
            "Install with: pip install pandas pyarrow"
        ) from e


def read_episode_parquet(parquet_path: Path):
    pd = safe_import_pandas()
    try:
        return pd.read_parquet(parquet_path, engine="pyarrow")
    except ImportError as e:
        raise RuntimeError("pyarrow is required to read parquet files. Install with: pip install pyarrow") from e


def discover_dataset_roots(paths: Iterable[Path]) -> List[Path]:
    found: List[Path] = []
    seen: set[Path] = set()

    for p in paths:
        p = p.resolve()
        if not p.exists():
            continue

        if (p / "meta" / "info.json").exists() and (p / "data").exists() and (p / "videos").exists():
            if p not in seen:
                found.append(p)
                seen.add(p)
            continue

        for info_path in p.glob("**/meta/info.json"):
            root = info_path.parent.parent.resolve()
            if (root / "data").exists() and (root / "videos").exists() and root not in seen:
                found.append(root)
                seen.add(root)

    return sorted(found)


@dataclass
class FrameRecord:
    dataset_root: Path
    task_index_to_name: Dict[int, str]
    episode_index: int
    row_index: int
    frame_index: int
    timestamp: float
    done: bool
    task_index: int
    action: np.ndarray
    observation_state: np.ndarray
    video_paths: Dict[str, Path]


class SurgicalFrameDataset(Dataset):
    """Frame-level PyTorch dataset across one or multiple exported dataset roots."""

    def __init__(self, dataset_roots: List[Path], image_dtype: torch.dtype = torch.float32):
        if not dataset_roots:
            raise ValueError("dataset_roots is empty")

        self.dataset_roots = [p.resolve() for p in dataset_roots]
        self.image_dtype = image_dtype
        self.records: List[FrameRecord] = []

        # OpenCV capture cache (lazy per-process/per-worker).
        self._cap_cache: Dict[str, cv2.VideoCapture] = {}

        self._build_index()
        if not self.records:
            raise RuntimeError("No frame records indexed from dataset roots")

    def _validate_columns(self, df, parquet_path: Path) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{parquet_path}: missing required columns {missing}")

        present_video_cols = [k for k in VIDEO_KEYS if k in df.columns]
        if not present_video_cols:
            raise ValueError(f"{parquet_path}: no video columns found")

    def _resolve_episode_video_paths(self, dataset_root: Path, row0, parquet_path: Path) -> Dict[str, Path]:
        paths: Dict[str, Path] = {}
        for key in VIDEO_KEYS:
            if key not in row0.index:
                continue
            cell = row0[key]
            if not isinstance(cell, dict) or "path" not in cell:
                raise ValueError(f"{parquet_path}: unexpected VideoFrame in {key}: {type(cell)}")

            full = (dataset_root / cell["path"]).resolve()
            if not full.exists():
                raise FileNotFoundError(f"{parquet_path}: video for {key} not found: {full}")
            paths[key] = full

        return paths

    def _build_index(self) -> None:
        for dataset_root in self.dataset_roots:
            tasks_path = dataset_root / "meta" / "tasks.jsonl"
            episodes_path = dataset_root / "meta" / "episodes.jsonl"
            if not tasks_path.exists() or not episodes_path.exists():
                raise FileNotFoundError(f"Missing metadata files under {dataset_root / 'meta'}")

            tasks = read_jsonl(tasks_path)
            task_index_to_name = {int(r["task_index"]): str(r["task"]) for r in tasks}
            episodes = read_jsonl(episodes_path)

            for ep in episodes:
                episode_index = int(ep["episode_index"])
                parquet_path = dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
                if not parquet_path.exists():
                    raise FileNotFoundError(f"Episode parquet missing: {parquet_path}")

                df = read_episode_parquet(parquet_path)
                self._validate_columns(df, parquet_path)
                if len(df) == 0:
                    continue

                row0 = df.iloc[0]
                video_paths = self._resolve_episode_video_paths(dataset_root, row0, parquet_path)

                for i, row in df.iterrows():
                    record = FrameRecord(
                        dataset_root=dataset_root,
                        task_index_to_name=task_index_to_name,
                        episode_index=int(row["episode_index"]),
                        row_index=int(i),
                        frame_index=int(row["frame_index"]),
                        timestamp=float(row["timestamp"]),
                        done=bool(row["next.done"]),
                        task_index=int(row["task_index"]),
                        action=np.asarray(row["action"], dtype=np.float32),
                        observation_state=np.asarray(row["observation.state"], dtype=np.float32),
                        video_paths=video_paths,
                    )
                    self.records.append(record)

    def __len__(self) -> int:
        return len(self.records)

    def _get_cap(self, video_path: Path) -> cv2.VideoCapture:
        key = str(video_path)
        cap = self._cap_cache.get(key)
        if cap is None:
            cap = cv2.VideoCapture(key)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")
            self._cap_cache[key] = cap
        return cap

    def _read_frame(self, cap: cv2.VideoCapture, frame_index: int, video_path: Path) -> np.ndarray:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Failed reading frame {frame_index} from {video_path}")
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return frame_rgb

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]

        images: Dict[str, torch.Tensor] = {}
        for key, video_path in record.video_paths.items():
            cap = self._get_cap(video_path)
            img = self._read_frame(cap, record.frame_index, video_path)

            # HWC uint8 -> CHW tensor for model input.
            tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()
            if self.image_dtype.is_floating_point:
                tensor = tensor.to(self.image_dtype) / 255.0
            images[key] = tensor

        label_name = record.task_index_to_name.get(record.task_index, f"task_{record.task_index}")

        return {
            "dataset_root": str(record.dataset_root),
            "episode_index": record.episode_index,
            "row_index": record.row_index,
            "frame_index": record.frame_index,
            "timestamp": record.timestamp,
            "done": record.done,
            "task_index": record.task_index,
            "task_name": label_name,
            "action": torch.from_numpy(record.action),
            "observation_state": torch.from_numpy(record.observation_state),
            "images": images,
        }

    def close(self) -> None:
        for cap in self._cap_cache.values():
            cap.release()
        self._cap_cache.clear()



def surgical_collate_fn(batch: List[dict]) -> dict:
    if not batch:
        raise ValueError("Empty batch")

    # Stack available camera tensors per camera key.
    camera_keys = sorted(set().union(*[set(sample["images"].keys()) for sample in batch]))
    images: Dict[str, torch.Tensor] = {}
    for cam in camera_keys:
        cam_tensors = [sample["images"][cam] for sample in batch if cam in sample["images"]]
        if len(cam_tensors) != len(batch):
            raise ValueError(f"Inconsistent camera availability in batch for key: {cam}")
        images[cam] = torch.stack(cam_tensors, dim=0)

    return {
        "dataset_root": [s["dataset_root"] for s in batch],
        "episode_index": torch.tensor([s["episode_index"] for s in batch], dtype=torch.long),
        "row_index": torch.tensor([s["row_index"] for s in batch], dtype=torch.long),
        "frame_index": torch.tensor([s["frame_index"] for s in batch], dtype=torch.long),
        "timestamp": torch.tensor([s["timestamp"] for s in batch], dtype=torch.float32),
        "done": torch.tensor([s["done"] for s in batch], dtype=torch.bool),
        "task_index": torch.tensor([s["task_index"] for s in batch], dtype=torch.long),
        "task_name": [s["task_name"] for s in batch],
        "action": torch.stack([s["action"] for s in batch], dim=0),
        "observation_state": torch.stack([s["observation_state"] for s in batch], dim=0),
        "images": images,
    }


def create_dataloader(
    dataset_roots: List[Path],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
) -> Tuple[SurgicalFrameDataset, DataLoader]:
    dataset = SurgicalFrameDataset(dataset_roots=dataset_roots)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=surgical_collate_fn,
        drop_last=False,
    )
    return dataset, loader


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch DataLoader for surgical LeRobot exports")
    parser.add_argument("--dataset-root", type=Path, default=None, help="One dataset root with meta/, data/, videos/")
    parser.add_argument(
        "--scan-roots",
        type=Path,
        nargs="*",
        default=[Path("./KnotTying"), Path("./Suturing")],
        help="Discover dataset roots recursively from these paths",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--max-batches", type=int, default=2, help="How many batches to iterate for demo")
    args = parser.parse_args()

    if args.dataset_root is not None:
        dataset_roots = [args.dataset_root.resolve()]
    else:
        dataset_roots = discover_dataset_roots(args.scan_roots)

    if not dataset_roots:
        raise RuntimeError("No dataset roots found. Use --dataset-root or valid --scan-roots")

    print(f"Discovered {len(dataset_roots)} dataset roots")

    dataset, loader = create_dataloader(
        dataset_roots=dataset_roots,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        pin_memory=args.pin_memory,
    )

    print(f"Indexed {len(dataset)} frame samples")

    for batch_idx, batch in enumerate(loader):
        camera_shapes = {k: tuple(v.shape) for k, v in batch["images"].items()}
        print(
            f"batch={batch_idx} "
            f"B={batch['action'].shape[0]} "
            f"action={tuple(batch['action'].shape)} "
            f"state={tuple(batch['observation_state'].shape)} "
            f"cameras={camera_shapes} "
            f"labels={batch['task_name'][:3]}"
        )
        if batch_idx + 1 >= args.max_batches:
            break

    # Explicit release for the main process when num_workers=0.
    dataset.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        raise SystemExit(1)
