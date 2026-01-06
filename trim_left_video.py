#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import pandas as pd
import cv2
import subprocess


def read_left_indices_and_ts(
    sync_csv: str,
    idx_col: str = "left_idx",
    ts_col: str = "t_ref_s",   # or "left_t_s"
):
    """
    Read (left_idx, timestamp) pairs from sync table.
    Returns:
      left_indices: np.ndarray[int]
      left_ts_s:    np.ndarray[float] (UTC epoch seconds)
    """
    df = pd.read_csv(sync_csv)

    if idx_col not in df.columns:
        raise ValueError(f"Column '{idx_col}' not found in {sync_csv}. Columns={list(df.columns)}")
    if ts_col not in df.columns:
        raise ValueError(f"Column '{ts_col}' not found in {sync_csv}. Columns={list(df.columns)}")

    df = df[[idx_col, ts_col]].dropna()
    df[idx_col] = df[idx_col].astype(int)
    df[ts_col] = df[ts_col].astype(float)

    # Ensure chronological order (and consistent with proxy writing)
    df = df.sort_values(ts_col).reset_index(drop=True)

    left_indices = df[idx_col].to_numpy(dtype=int)
    left_ts_s = df[ts_col].to_numpy(dtype=float)

    if left_indices.size == 0:
        raise ValueError(f"No valid rows in {sync_csv} after reading {idx_col},{ts_col}")

    return left_indices, left_ts_s


def make_proxy_video(
    video_path: str,
    left_indices,
    left_ts_s,
    out_video: str,
    out_map_csv: str,
    out_frames_dir: str | None = None,
    fps_override: float | None = None,
    resize_wh: tuple[int, int] | None = (480, 270),
    jpg_quality: int = 95,
    verbose: bool = True,
):
    """
    Extract frames at left_indices from the original video, write a proxy mp4, and
    write an index map csv containing (proxy_frame, left_idx, timestamp).

    NOTE:
      - timestamp is taken from sync csv (left_ts_s), not inferred from video fps.
      - left_indices and left_ts_s must be aligned and same length.
    """
    if len(left_indices) != len(left_ts_s):
        raise ValueError(f"left_indices and left_ts_s must have same length, got {len(left_indices)} vs {len(left_ts_s)}")

    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = float(fps_override) if fps_override is not None else float(src_fps if src_fps and src_fps > 1e-3 else 30.0)

    if resize_wh is None:
        out_w, out_h = src_w, src_h
    else:
        out_w, out_h = resize_wh

    if verbose:
        print("=== Source video ===")
        print(f"path: {video_path}")
        print(f"fps: {src_fps:.3f}  size: {src_w}x{src_h}  frames: {src_n}")
        print("=== Proxy settings ===")
        print(f"out: {out_video}")
        print(f"fps: {fps:.3f}  size: {out_w}x{out_h}")
        print(f"indices: {len(left_indices)} (min={int(left_indices[0])}, max={int(left_indices[-1])})")

    # Validate indices range
    if int(left_indices[0]) < 0 or int(left_indices[-1]) >= src_n:
        raise ValueError(f"left_idx out of range for video frame_count={src_n}: min={int(left_indices[0])}, max={int(left_indices[-1])}")

    # Prepare writer
    Path(os.path.dirname(out_video) or ".").mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # fourcc = cv2.VideoWriter_fourcc(*"avc1")     # video codec h264
    writer = cv2.VideoWriter(out_video, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {out_video}")

    # Optional frames dir
    frames_dir = None
    if out_frames_dir is not None:
        frames_dir = Path(out_frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    last_pos = -1
    proxy_frame_id = 0

    for original_idx, ts_s in zip(left_indices, left_ts_s):
        original_idx = int(original_idx)

        # Seek if not sequential
        if original_idx != last_pos + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, original_idx)

        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame at index {original_idx} from {video_path}")

        last_pos = original_idx

        if resize_wh is not None and (frame.shape[1] != out_w or frame.shape[0] != out_h):
            frame_out = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        else:
            frame_out = frame

        writer.write(frame_out)

        if frames_dir is not None:
            img_path = frames_dir / f"{proxy_frame_id:06d}.jpg"
            cv2.imwrite(str(img_path), frame_out, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpg_quality)])

        rows.append({
            "proxy_frame": proxy_frame_id,
            "left_idx": original_idx,
            "timestamp": float(ts_s),  # UTC epoch seconds
        })

        proxy_frame_id += 1
        if verbose and proxy_frame_id % 500 == 0:
            print(f"written {proxy_frame_id}/{len(left_indices)} frames...")

    writer.release()
    cap.release()

    # mp4v to h264 using ffmpeg for better compatibility
    h264_out = out_video.replace(".mp4", "_h264.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", out_video,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            h264_out
        ],
        check=True
    )

    if verbose:
        print(f"H.264 proxy video: {h264_out}")


    pd.DataFrame(rows).to_csv(out_map_csv, index=False)

    if verbose:
        print("=== Done ===")
        print(f"proxy video: {out_video}")
        print(f"index map : {out_map_csv}")
        if frames_dir is not None:
            print(f"frames dir: {str(frames_dir)}")


def main():
    ap = argparse.ArgumentParser(description="Extract left frames listed in sync_all.csv and build a proxy video for GUI.")
    ap.add_argument("--sync_csv", required=True, help="sync table csv containing left_idx and timestamp column")
    ap.add_argument("--video", required=True, help="original left video (mp4)")
    ap.add_argument("--out_video", default="proxy_left.mp4")
    ap.add_argument("--out_map", default="proxy_left_index_map.csv")
    ap.add_argument("--idx_col", default="left_idx", help="column name for left indices")
    ap.add_argument("--ts_col", default="t_ref_s", help="timestamp column name (e.g., t_ref_s or left_t_s)")
    ap.add_argument("--fps", type=float, default=None, help="override FPS for proxy video (default: source fps)")
    ap.add_argument("--no_resize", action="store_true", help="do not resize; keep original resolution")
    ap.add_argument("--resize", default="480x270", help="resize WxH, default 480x270 (ignored if --no_resize)")
    ap.add_argument("--frames_dir", default=None, help="optional: also dump proxy frames as jpgs into this folder")
    ap.add_argument("--jpg_quality", type=int, default=95)
    args = ap.parse_args()

    left_indices, left_ts_s = read_left_indices_and_ts(
        args.sync_csv,
        idx_col=args.idx_col,
        ts_col=args.ts_col,
    )

    resize_wh = None
    if not args.no_resize:
        if "x" not in args.resize.lower():
            raise ValueError("--resize must look like 480x270")
        w, h = args.resize.lower().split("x")
        resize_wh = (int(w), int(h))

    make_proxy_video(
        video_path=args.video,
        left_indices=left_indices,
        left_ts_s=left_ts_s,
        out_video=args.out_video,
        out_map_csv=args.out_map,
        out_frames_dir=args.frames_dir,
        fps_override=args.fps,
        resize_wh=resize_wh,
        jpg_quality=args.jpg_quality,
        verbose=True,
    )


if __name__ == "__main__":
    main()
