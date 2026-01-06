#!/usr/bin/env python3
"""
export_mm_to_lerobot_v21_fixed_stats.py

Rewrite of exporter to satisfy Open-H validate_formatting.py and LeRobotDataset loading.

Key fixes:
- Writes meta/info.json with REQUIRED features: "action" and "observation.state"
- Writes meta/info.json with top-level "stats" (mean/std) for required features
- Writes meta/episodes_stats.jsonl where EACH LINE includes a top-level "stats" field
  (fixes LeRobotDataset KeyError: 'stats')
- Writes Parquet using pyarrow with stable FixedSizeList<float32> columns for action/state
- Adds recommended split keys 'val' and 'test' (empty ranges) to remove warnings
- Adds recommended image feature prefix 'observation.images.left' (metadata only)

Public-vs-internal:
- Default exports dummy zeros for action/state.
- Map internal columns using --action_cols/--state_cols.

Example:
  python export_mm_to_lerobot_v21_fixed_stats.py \
    --sync_labeled /path/sync_table_left.csv \
    --sync_mm /path/sync_table_mm.csv \
    --dataset_root /path/out_dataset \
    --ts_col timestamp \
    --fps 30
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def find_first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def guess_time_unit_scale_to_seconds(ts: np.ndarray) -> float:
    ts = ts.astype(np.float64)
    ts = ts[np.isfinite(ts)]
    if ts.size == 0:
        return 1.0
    m = np.nanmedian(np.abs(ts))
    if m > 1e14:   # ns
        return 1e-9
    if m > 1e11:   # us
        return 1e-6
    if m > 1e8:    # ms-ish
        return 1e-3
    return 1.0


def estimate_fps_from_ts_seconds(ts_s: np.ndarray) -> float:
    ts_s = ts_s.astype(np.float64)
    ts_s = ts_s[np.isfinite(ts_s)]
    if ts_s.size < 3:
        return 0.0
    d = np.diff(ts_s)
    d = d[np.isfinite(d)]
    d = d[d > 0]
    if d.size == 0:
        return 0.0
    dt = np.median(d)
    return float(1.0 / dt) if dt > 0 else 0.0


def contiguous_episode_spans(df_lab: pd.DataFrame) -> List[Dict[str, Any]]:
    if "episode_id" not in df_lab.columns or "episode_label" not in df_lab.columns:
        raise ValueError("sync_labeled must contain episode_id and episode_label (run GUI commit_labels first).")

    ep_id = df_lab["episode_id"].astype(str).fillna("-1").to_numpy()
    ep_lab = pd.to_numeric(df_lab["episode_label"], errors="coerce").fillna(-1).astype(int).to_numpy()

    spans: List[Dict[str, Any]] = []
    n = len(df_lab)
    i = 0
    while i < n:
        cur_id = str(ep_id[i])
        cur_lab = int(ep_lab[i])

        j = i
        while j + 1 < n and str(ep_id[j + 1]) == cur_id and int(ep_lab[j + 1]) == cur_lab:
            j += 1

        if cur_id not in ("-1", "nan", "None") and cur_lab != -1:
            spans.append({"episode_id": cur_id, "episode_label": cur_lab, "start_row": int(i), "end_row": int(j)})

        i = j + 1
    return spans


def safe_insert(df: pd.DataFrame, loc: int, col: str, values) -> None:
    if col in df.columns:
        df[col] = values
    else:
        df.insert(loc, col, values)


def coerce_feature_matrix(df: pd.DataFrame, cols: List[str], out_dim: int, dtype=np.float32) -> np.ndarray:
    n = len(df)
    if out_dim <= 0:
        return np.zeros((n, 0), dtype=dtype)
    m = np.zeros((n, out_dim), dtype=dtype)
    if not cols:
        return m
    use = cols[:out_dim]
    for k, c in enumerate(use):
        if c not in df.columns:
            continue
        m[:, k] = pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(dtype=dtype)
    return m


def to_fixed_list_array(mat: np.ndarray) -> pa.Array:
    """Convert (N, D) float32 matrix to FixedSizeListArray(list_size=D)."""
    if mat.ndim != 2:
        raise ValueError("mat must be 2D")
    mat = np.asarray(mat, dtype=np.float32, order="C")
    n, d = mat.shape
    flat = pa.array(mat.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, list_size=d)


def compute_stats(mat: np.ndarray, eps: float = 1e-8) -> Dict[str, Any]:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.size == 0:
        return {"mean": [], "std": []}
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    std = np.maximum(std, eps)
    return {"mean": mean.astype(float).tolist(), "std": std.astype(float).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync_labeled", type=str, required=True)
    ap.add_argument("--sync_mm", type=str, required=True)
    ap.add_argument("--dataset_root", type=str, required=True)

    ap.add_argument("--ts_col", type=str, default="")
    ap.add_argument("--fps", type=float, default=0.0)
    ap.add_argument("--task_prefix", type=str, default="surgical_gesture")
    ap.add_argument("--chunk_id", type=str, default="chunk-000")

    ap.add_argument("--action_dim", type=int, default=1)
    ap.add_argument("--state_dim", type=int, default=1)
    ap.add_argument("--action_cols", type=str, nargs="*", default=[])
    ap.add_argument("--state_cols", type=str, nargs="*", default=[])

    ap.add_argument("--copy_videos", type=str, nargs="*", default=[])
    ap.add_argument("--image_h", type=int, default=480)
    ap.add_argument("--image_w", type=int, default=640)
    args = ap.parse_args()

    sync_labeled = Path(args.sync_labeled)
    sync_mm = Path(args.sync_mm)
    root = Path(args.dataset_root)

    data_dir = root / "data" / args.chunk_id
    videos_dir = root / "videos"
    meta_dir = root / "meta"
    ensure_dir(data_dir)
    ensure_dir(videos_dir)
    ensure_dir(meta_dir)

    df_lab = pd.read_csv(sync_labeled)
    df_mm = pd.read_csv(sync_mm)

    spans = contiguous_episode_spans(df_lab)
    if not spans:
        raise ValueError("No labeled episodes found (episode_label != -1).")

    # timestamps
    ts_col = args.ts_col.strip()
    if not ts_col:
        ts_col = find_first_existing_col(df_lab, ["timestamp", "left_unix_us", "left_ts", "ts", "time"]) or ""
        if not ts_col:
            num_cols = [c for c in df_lab.columns if pd.api.types.is_numeric_dtype(df_lab[c])]
            ts_col = num_cols[0] if num_cols else ""

    if ts_col and ts_col in df_lab.columns:
        ts_raw = pd.to_numeric(df_lab[ts_col], errors="coerce").to_numpy(dtype=np.float64)
        scale = guess_time_unit_scale_to_seconds(ts_raw)
        ts_s = ts_raw * scale
        fps_est = estimate_fps_from_ts_seconds(ts_s)
    else:
        ts_s = np.full((len(df_lab),), np.nan, dtype=np.float64)
        fps_est = 0.0

    fps = float(args.fps) if args.fps > 0 else float(fps_est if fps_est > 0 else 30.0)

    # copy videos
    for vp in args.copy_videos:
        src = Path(vp)
        if not src.exists():
            raise FileNotFoundError(f"Video not found: {src}")
        dst = videos_dir / src.name
        if dst.resolve() != src.resolve():
            shutil.copy2(src, dst)

    # tasks
    labels = sorted({int(s["episode_label"]) for s in spans})
    label_to_task_index = {lab: i for i, lab in enumerate(labels)}
    tasks_rows = [{"task_index": int(label_to_task_index[lab]), "task": f"{args.task_prefix}:{lab}"} for lab in labels]

    episodes_rows: List[Dict[str, Any]] = []
    episodes_stats_rows: List[Dict[str, Any]] = []

    total_frames = 0
    all_action = []
    all_state = []

    for ep_i, sp in enumerate(spans):
        srow, erow = int(sp["start_row"]), int(sp["end_row"])
        eid = str(sp["episode_id"])
        lab = int(sp["episode_label"])

        df_ep = df_mm.iloc[srow:erow + 1].copy().reset_index(drop=True)
        n = len(df_ep)
        if n <= 0:
            continue

        safe_insert(df_ep, 0, "index", np.arange(total_frames, total_frames + n, dtype=np.int64))
        safe_insert(df_ep, 1, "episode_index", np.full((n,), ep_i, dtype=np.int64))
        safe_insert(df_ep, 2, "frame_index", np.arange(n, dtype=np.int64))

        # episode-relative timestamp
        if ts_col and ts_col in df_lab.columns:
            ts_seg = ts_s[srow:erow + 1].astype(np.float64)
            ok = np.all(np.isfinite(ts_seg)) and np.all(np.diff(ts_seg) > 0)
            ts_rel = (ts_seg - float(ts_seg[0])) if ok else (np.arange(n, dtype=np.float64) / fps)
        else:
            ts_rel = (np.arange(n, dtype=np.float64) / fps)

        if "timestamp" in df_ep.columns:
            df_ep.rename(columns={"timestamp": "timestamp_raw"}, inplace=True)
        safe_insert(df_ep, 3, "timestamp", ts_rel.astype(np.float64))
        safe_insert(df_ep, 4, "task_index", np.full((n,), label_to_task_index[lab], dtype=np.int64))

        df_ep["next.done"] = False
        df_ep.loc[n - 1, "next.done"] = True

        action_mat = coerce_feature_matrix(df_ep, args.action_cols, args.action_dim, dtype=np.float32)
        state_mat = coerce_feature_matrix(df_ep, args.state_cols, args.state_dim, dtype=np.float32)

        all_action.append(action_mat)
        all_state.append(state_mat)

        # Arrow table: base columns + FixedSizeList action/state
        base_tbl = pa.Table.from_pandas(df_ep, preserve_index=False)
        for drop in ["action", "observation.state"]:
            if drop in base_tbl.column_names:
                base_tbl = base_tbl.drop([drop])

        tbl = base_tbl.append_column("action", to_fixed_list_array(action_mat))
        tbl = tbl.append_column("observation.state", to_fixed_list_array(state_mat))

        out_ep = data_dir / f"episode_{ep_i:06d}.parquet"
        pq.write_table(tbl, out_ep, compression="zstd")

        episodes_rows.append({
            "episode_index": int(ep_i),
            "tasks": [f"{args.task_prefix}:{lab}"],
            "length": int(n),
            "source_episode_id": eid,
        })

        # IMPORTANT: include top-level "stats" (LeRobotDataset expects this)
        ep_stats = {
            "action": compute_stats(action_mat),
            "observation.state": compute_stats(state_mat),
        }
        episodes_stats_rows.append({
            "episode_index": int(ep_i),
            "length": int(n),
            "fps": float(fps),
            "duration_s": float(ts_rel[-1] if n > 1 else 0.0),
            "stats": ep_stats,
        })

        total_frames += n

    if total_frames == 0:
        raise RuntimeError("No episode frames exported (check labels and alignment).")

    all_action_mat = np.concatenate(all_action, axis=0) if all_action else np.zeros((0, args.action_dim), np.float32)
    all_state_mat = np.concatenate(all_state, axis=0) if all_state else np.zeros((0, args.state_dim), np.float32)

    dataset_stats = {
        "action": compute_stats(all_action_mat),
        "observation.state": compute_stats(all_state_mat),
    }

    info = {
        "codebase_version": "v2.1",
        "robot_type": "surgical",
        "fps": float(fps),
        "total_episodes": int(len(episodes_rows)),
        "total_frames": int(total_frames),
        "total_tasks": int(len(tasks_rows)),
        "splits": {"train": f"0:{len(episodes_rows)}", "val": "0:0", "test": "0:0"},
        "notes": {"sync_labeled": str(sync_labeled), "sync_mm": str(sync_mm), "ts_col": ts_col},
        "features": {
            "action": {"dtype": "float32", "shape": [int(args.action_dim)]},
            "observation.state": {"dtype": "float32", "shape": [int(args.state_dim)]},
            "observation.images.left": {"dtype": "uint8", "shape": [int(args.image_h), int(args.image_w), 3]},
        },
        "stats": dataset_stats,
        "features_minimal": [
            "index", "episode_index", "frame_index", "timestamp", "task_index", "next.done",
            "action", "observation.state",
        ],
    }

    write_json(meta_dir / "info.json", info)
    write_jsonl(meta_dir / "tasks.jsonl", tasks_rows)
    write_jsonl(meta_dir / "episodes.jsonl", episodes_rows)
    write_jsonl(meta_dir / "episodes_stats.jsonl", episodes_stats_rows)

    (meta_dir / "README.md").write_text(
        f"""# Surgical Multimodal Dataset (LeRobot v2.1)

This dataset contains synchronized multimodal surgical recordings.

## Synchronization
All modalities are aligned using timestamp-based synchronization.
Episodes are segmented and labeled based on the left endoscopic video table (`sync_labeled`).

## Episodes & Tasks
Each episode is a contiguous labeled clip (episode_label != -1). Tasks are encoded as `{args.task_prefix}:<label>`.

## Privacy / Release Notes
For public release, `action` and `observation.state` are exported as dummy zeros unless you map internal columns via
`--action_cols` / `--state_cols`.
""",
        encoding="utf-8",
    )

    print("✅ Export complete")
    print(f"Dataset root: {root}")
    print(f"Next: python validate_formatting.py {root} --verbose")


if __name__ == "__main__":
    main()
