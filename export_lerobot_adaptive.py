#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a LeRobot v2.1 dataset from:
1) sync_table_all_*.csv : per-row synchronized indices across modalities
2) sync_table.csv       : GUI-exported episode_id + episode_label per row

What this script does
---------------------
- For each labeled episode_id (episode_id != -1), it:
  1) selects the rows belonging to the episode
  2) for each modality video, extracts needed frames (by index) from the ORIGINAL video
     (using ffmpeg; supports a one-time "extract all frames" cache per modality)
  3) re-encodes per-episode MP4 videos (one per modality) with a unified episode FPS
  4) writes v2.1 directory structure:
       data/chunk-000/episode_000000.parquet
       videos/chunk-000/observation.images.<cam>/episode_000000.mp4
       meta/{info.json,tasks.jsonl,episodes.jsonl,episodes_stats.jsonl,README.md}
- Writes Parquet rows with:
    index, episode_index, frame_index, timestamp, task_index, next.done,
    action, observation.state,
    observation.images.left/right/side/gaze  (as VideoFrame dicts: {"path":..., "timestamp":...})

Notes
-----
- LeRobot expects video features to be stored as MP4s on disk, and referenced from Parquet via
  a dict {"path": "...", "timestamp": ...} for each row.
- validate_formatting.py used in Open-H checks requires action + observation.state to exist.
  If you cannot release them, you can still fill them with zeros for "public" export; this
  is a *format* requirement, not a semantic one.

Requirements
------------
- ffmpeg + ffprobe on PATH
- python: pandas, numpy, pyarrow, opencv-python

Example
-------
python export_lerobot_from_sync_csv.py \
  --sync-all /path/sync_table_all_xxx.csv \
  --sync-ep  /path/sync_table.csv \
  --left-video  /path/left.mp4 \
  --right-video /path/right.mp4 \
  --side-video  /path/side.mp4 \
  --gaze-video  /path/gaze.mp4 \
  --out /path/out_dataset \
  --dataset-name surgical_openh_demo \
  --profile public
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

# -----------------------------
# FFmpeg helpers
# -----------------------------

def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"{' '.join(cmd)}\n\n"
            f"STDOUT:\n{p.stdout}\n\nSTDERR:\n{p.stderr}\n"
        )

def ffmpeg_has_encoder(name: str) -> bool:
    """Return True if `ffmpeg -encoders` lists the given encoder name."""
    try:
        p = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return name in p.stdout
    except Exception:
        return False

def ffprobe_fps(video_path: Path) -> float:
    """
    Return FPS as float using ffprobe.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(video_path),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0 or not p.stdout.strip():
        raise RuntimeError(f"ffprobe failed for {video_path}:\n{p.stderr}")
    rate = p.stdout.strip()  # e.g. "30000/1001" or "30/1"
    if "/" in rate:
        num, den = rate.split("/")
        return float(num) / float(den)
    return float(rate)

def extract_all_frames(video_path: Path, frames_dir: Path) -> None:
    """
    Extract all frames once, cached on disk. Output naming starts at 0.

    frames_dir/frame_000000.png
    frames_dir/frame_000001.png
    ...
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    # If already extracted, skip (simple heuristic)
    sample = frames_dir / "frame_000000.png"
    if sample.exists():
        return

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-start_number", "0",
        "-vsync", "0",
        str(frames_dir / "frame_%06d.png"),
    ]
    _run(cmd)

def make_episode_video_from_frames(frames_in_dir: Path, frame_indices: List[int], out_mp4: Path, fps: float, image_size: Tuple[int, int]=(256, 256)) -> Tuple[int, int]:
    """
    Copy needed frames into a temp sequence (0..N-1) then encode to mp4.

    Returns (width, height) from the first frame, if available (otherwise (0,0)).
    """
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    H, W = image_size  # (H, W)
    tmp = out_mp4.parent / (out_mp4.stem + "_tmp_frames")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    # Copy frames
    for j, idx in enumerate(frame_indices):
        src = frames_in_dir / f"frame_{idx:06d}.png"
        if not src.exists():
            raise FileNotFoundError(f"Missing frame {src} (idx={idx})")
        dst = tmp / f"frame_{j:06d}.png"
        shutil.copyfile(src, dst)

    # Encode to MP4 (H.264 / libx264) with constant frame rate for best LeRobot compatibility.
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-framerate", f"{fps}",
        "-i", str(tmp / "frame_%06d.png"),
        "-vf", f"scale={W}:{H},setsar=1,setdar=0",
        # "-vf", f"scale={W}:{H}",
        "-r", f"{fps}",
        "-vsync", "cfr",
        "-pix_fmt", "yuv420p",
        "-c:v", "libx264",
        "-movflags", "+faststart",
        "-y",
        str(out_mp4),
    ]
    _run(cmd)

    # Read size with ffprobe
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0:s=x",
        str(out_mp4),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    wh = p.stdout.strip()
    w, h = (0, 0)
    if "x" in wh:
        w, h = map(int, wh.split("x"))
    shutil.rmtree(tmp)
    return w, h


# -----------------------------
# LeRobot metadata helpers
# -----------------------------

def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def stats_for_array(x: np.ndarray) -> dict:
    # x: (T, D)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 0:
        # scalar -> treat as (T=1, D=1)
        x = x.reshape(1, 1)
    elif x.ndim == 1:
        # (T,) -> (T,1)
        x = x[:, None]
    T = int(x.shape[0])
    mn = np.atleast_1d(np.min(x, axis=0))
    mx = np.atleast_1d(np.max(x, axis=0))
    mu = np.atleast_1d(np.mean(x, axis=0))
    sd = np.atleast_1d(np.std(x, axis=0) + 1e-12)
    return {
        "count": [T],
        "min": mn.tolist(),
        "max": mx.tolist(),
        "mean": mu.tolist(),
        "std": sd.tolist(),
    }

def write_episode_parquet_pyarrow(out_parquet: Path, rows: List[dict], action_dim: int, state_dim: int) -> None:
    """Write episode parquet with an explicit schema to avoid scalar/0-d issues in LeRobot.

    - action and observation.state are stored as fixed_size_list<float32>[D]
    - video features are stored as struct<path: string, timestamp: float32>
    """
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    # Primitive columns
    index_arr = pa.array([r["index"] for r in rows], type=pa.int64())
    episode_index_arr = pa.array([r["episode_index"] for r in rows], type=pa.int64())
    frame_index_arr = pa.array([r["frame_index"] for r in rows], type=pa.int64())
    timestamp_arr = pa.array([r["timestamp"] for r in rows], type=pa.float32())
    task_index_arr = pa.array([r["task_index"] for r in rows], type=pa.int64())
    done_arr = pa.array([r["next.done"] for r in rows], type=pa.bool_())

    # Fixed-size list helper
    def _fixed_list(name: str, dim: int):
        vals = []
        for r in rows:
            v = r[name]
            if isinstance(v, (list, tuple, np.ndarray)):
                v = list(v)
            else:
                # scalar -> wrap
                v = [float(v)]
            if len(v) != dim:
                raise ValueError(f"{name} length mismatch: expected {dim}, got {len(v)}")
            vals.append([float(x) for x in v])
        # Build FixedSizeListArray from flattened values
        flat = pa.array([x for row in vals for x in row], type=pa.float32())
        return pa.FixedSizeListArray.from_arrays(flat, dim)

    action_arr = _fixed_list("action", action_dim)
    state_arr = _fixed_list("observation.state", state_dim)

    # VideoFrame struct helper
    vf_type = pa.struct([("path", pa.string()), ("timestamp", pa.float32())])

    def _vf(colname: str):
        paths = [r[colname]["path"] for r in rows]
        tss = [r[colname]["timestamp"] for r in rows]
        return pa.StructArray.from_arrays(
            [pa.array(paths, type=pa.string()), pa.array(tss, type=pa.float32())],
            fields=list(vf_type),
        )

    left_arr = _vf("observation.images.left")
    right_arr = _vf("observation.images.right")
    side_arr = _vf("observation.images.side")
    gaze_arr = _vf("observation.images.gaze")

    schema = pa.schema([
        ("index", pa.int64()),
        ("episode_index", pa.int64()),
        ("frame_index", pa.int64()),
        ("timestamp", pa.float32()),
        ("task_index", pa.int64()),
        ("next.done", pa.bool_()),
        ("action", pa.list_(pa.float32(), list_size=action_dim)),
        ("observation.state", pa.list_(pa.float32(), list_size=state_dim)),
        ("observation.images.left", vf_type),
        ("observation.images.right", vf_type),
        ("observation.images.side", vf_type),
        ("observation.images.gaze", vf_type),
    ])

    table = pa.Table.from_arrays(
        [
            index_arr, episode_index_arr, frame_index_arr, timestamp_arr,
            task_index_arr, done_arr, action_arr, state_arr,
            left_arr, right_arr, side_arr, gaze_arr
        ],
        schema=schema,
    )
    pq.write_table(table, out_parquet, compression="zstd")

def safe_float_timestamp_series(ts: np.ndarray) -> np.ndarray:
    # Ensure float64 seconds
    ts = np.asarray(ts, dtype=np.float64)
    return ts


@dataclass
class EpisodeSpec:
    episode_id: str
    label: int
    rows: pd.DataFrame


def build_episodes(df_ep: pd.DataFrame) -> List[EpisodeSpec]:
    """
    df_ep: GUI sync_table.csv
      columns: row_idx,left_frame,proxy_frame,left_idx,timestamp,episode_id,episode_label
    """
    df = df_ep.copy()
    # normalize types
    df["episode_id"] = df["episode_id"].astype(str)
    df["episode_label"] = pd.to_numeric(df["episode_label"], errors="coerce").fillna(-1).astype(int)

    # keep labeled episodes only
    df = df[df["episode_id"] != "-1"].copy()
    if df.empty:
        return []

    eps: List[EpisodeSpec] = []
    for eid, g in df.groupby("episode_id", sort=False):
        # assume one label per episode id; if multiple, take mode
        lbl = int(g["episode_label"].mode().iloc[0])
        g = g.sort_values("row_idx").reset_index(drop=True)
        eps.append(EpisodeSpec(episode_id=eid, label=lbl, rows=g))
    return eps


# -----------------------------
# Main export
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sync-all", type=Path, required=True, help="sync_table_all_*.csv")
    ap.add_argument("--sync-ep", type=Path, required=True, help="GUI exported sync_table.csv")
    ap.add_argument("--left-video", type=Path, required=True)
    ap.add_argument("--right-video", type=Path, required=False)
    ap.add_argument("--side-video", type=Path, required=False)
    ap.add_argument("--gaze-video", type=Path, required=False)

    ap.add_argument("--out", type=Path, required=True, help="output dataset root folder")
    ap.add_argument("--dataset-name", type=str, default="dataset")
    ap.add_argument("--profile", choices=["public", "internal"], default="public")
    ap.add_argument("--cache-dir", type=Path, default=None, help="where to store extracted frames cache (default: <out>/.frame_cache)")
    ap.add_argument("--fps-mode", choices=["from_timestamps", "fixed"], default="from_timestamps")
    ap.add_argument("--fixed-fps", type=float, default=30.0)

    ap.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[480, 640],
        metavar=("H", "W"),
        help="Resize images before encoding episode videos (default: 480 640)",
    )

    # action/state dims (for placeholder)
    ap.add_argument("--action-dim", type=int, default=1)
    ap.add_argument("--state-dim", type=int, default=1)

    args = ap.parse_args()

    # LeRobot (video_backend="pytorch") is most reliable with H.264 (libx264) MP4.
    if not ffmpeg_has_encoder("libx264"):
        raise RuntimeError(
            "ffmpeg encoder 'libx264' not found. Please install an ffmpeg build with libx264 "
            "(e.g., conda install -c conda-forge ffmpeg) and re-run."
        )

    out_root = args.out / args.dataset_name
    data_dir = out_root / "data" / "chunk-000"
    videos_root = out_root / "videos" / "chunk-000"
    meta_dir = out_root / "meta"

    out_root.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir or (out_root / ".frame_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Load CSVs
    df_all = pd.read_csv(args.sync_all)
    df_ep  = pd.read_csv(args.sync_ep)

    # Merge episode rows with sync_all by row index (preferred) else by left_idx
    if "row_idx" in df_ep.columns and "row_idx" in df_all.columns:
        merged = df_ep.merge(df_all, on="row_idx", how="left", suffixes=("", "_all"))
    else:
        merged = df_ep.merge(df_all, on="left_idx", how="left", suffixes=("", "_all"))

    episodes = build_episodes(merged)
    if not episodes:
        raise RuntimeError("No labeled episodes found (episode_id != -1).")

    # Extract all frames once per modality (cache)
    left_frames_dir  = cache_dir / "left_frames"
    right_frames_dir = cache_dir / "right_frames"
    side_frames_dir  = cache_dir / "side_frames"
    gaze_frames_dir  = cache_dir / "gaze_frames"

    extract_all_frames(args.left_video, left_frames_dir)
    args.right_video and extract_all_frames(args.right_video, right_frames_dir)
    args.side_video and extract_all_frames(args.side_video, side_frames_dir)
    args.gaze_video and extract_all_frames(args.gaze_video, gaze_frames_dir)

    # Determine output FPS (unified for all episode videos)
    # We default to using GUI timestamps (absolute unix seconds) -> infer median dt.
    # Then you can override with --fixed-fps.
    if args.fps_mode == "fixed":
        fps_ref = float(args.fixed_fps)
    else:
        ts = pd.to_numeric(merged["timestamp"], errors="coerce").dropna().to_numpy(dtype=np.float64)
        # If timestamp is unix seconds and consecutive, dt median works.
        dts = np.diff(ts)
        dts = dts[np.isfinite(dts) & (dts > 0)]
        if len(dts) == 0:
            fps_ref = float(args.fixed_fps)
        else:
            fps_ref = float(1.0 / np.median(dts))
            # guardrails
            if fps_ref < 1 or fps_ref > 240:
                fps_ref = float(args.fixed_fps)

    # Tasks: map label -> task_index
    unique_labels = sorted({ep.label for ep in episodes})
    label_to_task_index = {lbl: i for i, lbl in enumerate(unique_labels)}
    tasks_rows = [{"task_index": label_to_task_index[lbl], "task": f"gesture_{lbl}"} for lbl in unique_labels]
    write_jsonl(meta_dir / "tasks.jsonl", tasks_rows)

    episodes_rows = []
    episodes_stats_rows = []

    # We'll fill these after we create first episode videos (to know shapes)
    video_shapes: Dict[str, Tuple[int, int]] = {}

    global_index = 0
    total_frames = 0
    total_videos = 0

    # Create per-episode parquet + videos
    for episode_index, ep in enumerate(episodes):
        rows = ep.rows.copy()

        # Indices for each modality, 0-based frame indices into ORIGINAL videos
        # (Your sync_table_all_* uses columns left_idx/right_idx/side_idx/gaze_idx.)
        left_idxs  = rows["left_idx"].astype(int).tolist()
        right_idxs = rows["right_idx"].astype(int).tolist()
        side_idxs  = rows["side_idx"].astype(int).tolist()
        gaze_idxs  = rows["gaze_idx"].astype(int).tolist()

        # Encode videos (unified fps_ref)
        def _video_out(cam_key: str) -> Path:
            return videos_root / cam_key / f"episode_{episode_index:06d}.mp4"

        def _maybe_encode(video_path, frames_dir, idxs, cam_key):
            if video_path is None:
                return (0,0)
            return make_episode_video_from_frames(frames_dir, idxs, _video_out(cam_key), fps_ref, tuple(args.image_size))

        cam_left = "observation.images.left"
        cam_right = "observation.images.right"
        cam_side = "observation.images.side"
        cam_gaze = "observation.images.gaze"

        w, h = _maybe_encode(args.left_video, left_frames_dir, left_idxs, cam_left)
        print(w, h)
        video_shapes.setdefault(cam_left, (h, w))
        w, h = _maybe_encode(args.right_video, right_frames_dir, right_idxs, cam_right)
        print(w, h)
        video_shapes.setdefault(cam_right, (h, w))
        w, h = _maybe_encode(args.side_video, side_frames_dir, side_idxs, cam_side)
        print(w, h)
        video_shapes.setdefault(cam_side, (h, w))
        w, h = _maybe_encode(args.gaze_video, gaze_frames_dir, gaze_idxs, cam_gaze)
        print(w, h)
        video_shapes.setdefault(cam_gaze, (h, w))

        total_videos += sum([
            args.left_video is not None,
            args.right_video is not None,
            args.side_video is not None,
            args.gaze_video is not None,
        ])

        # Episode-relative timestamps (seconds from start)
        T = len(rows)
        total_frames += T

        dt = 1.0 / float(fps_ref)

        # Build parquet rows
        parquet_rows = []
        task_index = label_to_task_index[ep.label]

        # Placeholder action/state (format-required)
        action = np.zeros((T, args.action_dim), dtype=np.float32)
        state  = np.zeros((T, args.state_dim), dtype=np.float32)

        for i in range(T):
            done = (i == T - 1)
            t = float(i * dt)   

            def vf(cam_key: str, present: bool) -> dict:
                if not present:
                    return {"path": "", "timestamp": float("nan")}
                rel = Path("videos") / "chunk-000" / cam_key / f"episode_{episode_index:06d}.mp4"
                return {"path": str(rel.as_posix()), "timestamp": t}

            parquet_rows.append({
                "index": int(global_index),
                "episode_index": int(episode_index),
                "frame_index": int(i),
                "timestamp": t,                 # ✅
                "task_index": int(task_index),
                "next.done": bool(done),
                "action": action[i].tolist(),
                "observation.state": state[i].tolist(),
                cam_left: vf(cam_left, args.left_video is not None),
                cam_right: vf(cam_right, args.right_video is not None),
                cam_side: vf(cam_side, args.side_video is not None),
                cam_gaze: vf(cam_gaze, args.gaze_video is not None),
            })

            global_index += 1

        ts = np.array([r["timestamp"] for r in parquet_rows])
        assert np.allclose(np.diff(ts), 1.0 / fps_ref, atol=1e-4)

        # Write parquet
        out_parquet = data_dir / f"episode_{episode_index:06d}.parquet"
        write_episode_parquet_pyarrow(out_parquet, parquet_rows, args.action_dim, args.state_dim)
        # df_parquet = pd.DataFrame(parquet_rows)
        # df_parquet.to_parquet(out_parquet, index=False, engine="pyarrow")

        episodes_rows.append({
            "episode_index": int(episode_index),
            "tasks": [f"gesture_{ep.label}"],
            "length": int(T),
        })

        episodes_stats_rows.append({
            "episode_index": int(episode_index),
            "stats": {
                "action": stats_for_array(action),
                "observation.state": stats_for_array(state),
            }
        })

    write_jsonl(meta_dir / "episodes.jsonl", episodes_rows)
    write_jsonl(meta_dir / "episodes_stats.jsonl", episodes_stats_rows)

    # Write meta/README.md (Open-H requirement per validator)
    (meta_dir / "README.md").write_text(
        "# Dataset\n\n"
        "This dataset was exported from synchronized multi-modal recordings.\n\n"
        "## Synchronization\n"
        "Rows are aligned by a reference timestamp; each frame stores VideoFrame pointers into per-episode videos.\n\n"
        "## Missing modalities\n"
        "If a modality is missing for an episode, the corresponding VideoFrame entry is present in the parquet\n"
        "schema but has an empty path (\"\") and NaN timestamp.\n",
        encoding="utf-8"
    )

    # Build info.json
    features = {
        "action": {"dtype": "float32", "shape": [int(args.action_dim)], "names": [f"a{i}" for i in range(args.action_dim)]},
        "observation.state": {"dtype": "float32", "shape": [int(args.state_dim)], "names": [f"s{i}" for i in range(args.state_dim)]},
    }

    for cam_key, (h, w) in video_shapes.items():
        if h <= 0 or w <= 0:
            # fallback
            h, w = 480, 640
        features[cam_key] = {
            "dtype": "video",
            "shape": [int(h), int(w), 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.fps": float(fps_ref),
                "video.codec": "mp4",
            },
        }

    info = {
        "codebase_version": "v2.1",
        "robot_type": "surgical",
        "fps": float(fps_ref),
        "total_episodes": int(len(episodes)),
        "total_frames": int(total_frames),
        "total_tasks": int(len(tasks_rows)),
        "total_videos": int(total_videos),
        "chunks": ["chunk-000"],
        "chunks_size": int(len(episodes)),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "meta_path": "meta",
        "splits": {
            "train": f"0:{len(episodes)}",
            "val": "0:0",
            "test": "0:0",
        },
        "features": features,
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")

    # Top-level README (optional but helpful)
    (out_root / "README.md").write_text(
        f"# {args.dataset_name}\n\n"
        "LeRobot v2.1 dataset exported from Open-H GUI episode labels.\n",
        encoding="utf-8"
    )

    print("\n✅ Export complete.")
    print(f"Dataset written to: {out_root}")
    print("\nNow validate:")
    print(f"  python /path/to/validate_formatting.py {out_root}\n")


if __name__ == "__main__":
    main()
