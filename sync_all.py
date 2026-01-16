#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo


# -------------------------
# Data structures
# -------------------------

@dataclass
class Modality:
    name: str
    ts_s: np.ndarray      # sorted epoch seconds (UTC)
    dt_s: float
    fps: float
    unit_desc: str


# -------------------------
# Utilities
# -------------------------

def _to_1d(x) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def robust_median_dt(ts_s: np.ndarray) -> float:
    ts = np.sort(_to_1d(ts_s).astype(np.float64))
    ts = ts[np.isfinite(ts)]
    if ts.size < 3:
        raise ValueError("Need >= 3 timestamps to estimate dt.")
    dt = np.diff(ts)
    dt = dt[(dt > 0) & np.isfinite(dt)]
    if dt.size == 0:
        raise ValueError("No positive dt found; timestamps might be constant/invalid.")
    return float(np.median(dt))


def nearest_map(ref_t: np.ndarray, src_t_sorted: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    src_t_sorted must be sorted ascending.
    For each ref_t, return nearest index in src_t_sorted and error (src - ref).
    """
    ref_t = _to_1d(ref_t).astype(np.float64)
    src = _to_1d(src_t_sorted).astype(np.float64)

    pos = np.searchsorted(src, ref_t, side="left")
    pos0 = np.clip(pos - 1, 0, src.size - 1)
    pos1 = np.clip(pos, 0, src.size - 1)

    d0 = np.abs(src[pos0] - ref_t)
    d1 = np.abs(src[pos1] - ref_t)
    idx = np.where(d1 < d0, pos1, pos0).astype(np.int64)

    err = (src[idx] - ref_t).astype(np.float64)
    return idx, err


def compute_overlap(mods: Dict[str, Modality]) -> Tuple[float, float]:
    t0 = max(m.ts_s[0] for m in mods.values())
    t1 = min(m.ts_s[-1] for m in mods.values())
    if not (t1 > t0):
        raise ValueError(f"No overlap. overlap_start={t0}, overlap_end={t1}")
    return float(t0), float(t1)


def abs_err_stats(name: str, err_s: np.ndarray) -> str:
    e = np.abs(err_s[np.isfinite(err_s)])
    if e.size == 0:
        return f"{name}: empty"
    return (f"{name}: |err| p50={np.percentile(e,50):.6f}s "
            f"p95={np.percentile(e,95):.6f}s max={np.max(e):.6f}s")


# -------------------------
# Read left/right/side ISO timestamps
# -------------------------

def read_text_lines(path: str) -> np.ndarray:
    vals = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # allow "idx, value"
            if "," in s:
                s = s.split(",")[-1].strip()
            vals.append(s)
    return np.array(vals, dtype=object)


def read_video_iso_timestamps(path: str, csv_col: Optional[str] = None) -> np.ndarray:
    """
    ISO timestamps from:
      - .txt/.log: one per line
      - .csv: use csv_col if provided, else first column
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt", ".log"]:
        return read_text_lines(path)

    df = pd.read_csv(path, dtype=str)
    if df.shape[1] == 0:
        raise ValueError(f"{path} has no columns.")
    col = df.columns[0] if csv_col is None else csv_col
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in {path}. Available: {list(df.columns)}")
    return df[col].astype(str).to_numpy()


def parse_iso_naive_to_epoch_seconds(iso_strings: np.ndarray, tz_name: str) -> np.ndarray:
    """
    ISO like '2025-12-11T13:49:42.220008' (no timezone):
    treat as local time tz_name -> convert to UTC epoch seconds.
    """
    s = pd.Series(_to_1d(iso_strings).astype(str))
    dt = pd.to_datetime(s, errors="coerce", format="ISO8601")
    if dt.isna().any():
        bad = int(dt.isna().sum())
        example = s[dt.isna()].iloc[0]
        raise ValueError(f"Failed to parse {bad} ISO timestamps. Example bad: {example}")

    tz = ZoneInfo(tz_name)
    dt_local = dt.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    dt_utc = dt_local.dt.tz_convert("UTC")
    return (dt_utc.astype("int64").to_numpy() / 1e9).astype(np.float64)


def build_video_mod(name: str, iso_ts: np.ndarray, tz_name: str) -> Modality:
    ts_s = np.sort(parse_iso_naive_to_epoch_seconds(iso_ts, tz_name))
    dt_s = robust_median_dt(ts_s)
    fps = 1.0 / dt_s if dt_s > 0 else 0.0
    return Modality(name=name, ts_s=ts_s, dt_s=dt_s, fps=fps, unit_desc="iso_local->utc_s")


# -------------------------
# Read gaze TXT: extract epoch-us only
# -------------------------

def read_gaze_epoch_us_from_txt(gaze_path: str) -> np.ndarray:
    """
    Gaze log is a .txt (not guaranteed CSV). Extract ONLY epoch-us timestamps (>=1e13).
    Supports values like 1765489754602388.5
    Returns: epoch seconds (float64), sorted.
    """
    pat = re.compile(r"(?<!\d)(\d{13,}(?:\.\d+)?)(?!\d)")
    ts_us = []

    with open(gaze_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # skip header-like lines
            if "timestamp" in line and "timetick" in line:
                continue

            m = pat.search(line)
            if not m:
                continue
            val = float(m.group(1))
            if val >= 1e13:
                ts_us.append(val)

    if len(ts_us) < 10:
        raise ValueError(f"Too few epoch-us timestamps extracted from gaze txt: {gaze_path}. Extracted={len(ts_us)}")

    ts_us = np.array(ts_us, dtype=np.float64)
    if float(np.median(ts_us)) < 1e13:
        raise ValueError(f"Gaze extracted median too small: {np.median(ts_us)}")

    return np.sort(ts_us * 1e-6)


def build_gaze_mod_from_txt(name: str, gaze_path: str) -> Modality:
    ts_s = read_gaze_epoch_us_from_txt(gaze_path)
    dt_s = robust_median_dt(ts_s)
    fps = 1.0 / dt_s if dt_s > 0 else 0.0
    return Modality(name=name, ts_s=ts_s, dt_s=dt_s, fps=fps, unit_desc="gaze_txt_epoch_us")


# -------------------------
# Read DVAPI CSV (epoch-us) and dedup by api_cnt
# -------------------------

def read_dvapi_csv_dedup_by_api_cnt(dvapi_csv_path: str) -> Tuple[pd.DataFrame, np.ndarray, float, float]:
    """
    Reads dvapi csv, expects:
      - Time_stamp: epoch microseconds (int-like)
      - api_cnt: counter (repeats). For sync, keep one row per api_cnt.
    Returns:
      dvapi_dedup_df: dedup table with a new column 'dvapi_row' = 0..N-1
      dvapi_ts_s: sorted seconds array aligned with dvapi_dedup_df order (same order as dvapi_dedup_df)
      fps_raw: estimated FPS from raw (non-dedup) timestamps (median dt)
      fps_dedup: estimated FPS from dedup (median dt)
    """
    df = pd.read_csv(dvapi_csv_path, engine="python", on_bad_lines="skip")

    if "Time_stamp" not in df.columns or "api_cnt" not in df.columns:
        raise ValueError(f"DVAPI csv must contain 'Time_stamp' and 'api_cnt'. Columns: {list(df.columns)}")

    # raw timestamps (for fps report)
    ts_us_raw = pd.to_numeric(df["Time_stamp"], errors="coerce").dropna().astype(np.int64).to_numpy()
    ts_s_raw = np.sort(ts_us_raw * 1e-6)
    dt_raw = np.diff(ts_s_raw)
    dt_raw = dt_raw[dt_raw > 0]
    fps_raw = float(1.0 / np.median(dt_raw)) if dt_raw.size > 0 else 0.0

    # drop rows without api_cnt
    d = df.dropna(subset=["api_cnt"]).copy()
    d["api_cnt"] = pd.to_numeric(d["api_cnt"], errors="coerce").astype("Int64")
    d = d.dropna(subset=["api_cnt"])
    d["api_cnt"] = d["api_cnt"].astype(np.int64)

    # parse timestamp as int64 (epoch-us). keep numeric for ordering
    d["Time_stamp"] = pd.to_numeric(d["Time_stamp"], errors="coerce").astype("Int64")
    d = d.dropna(subset=["Time_stamp"])
    d["Time_stamp"] = d["Time_stamp"].astype(np.int64)

    # Dedup: one row per api_cnt.
    # choose the FIRST timestamp for each api_cnt (and all other fields from that row)
    d_sorted = d.sort_values(["api_cnt", "Time_stamp"], kind="mergesort")
    dedup = d_sorted.groupby("api_cnt", as_index=False).first()

    # Compute dedup fps
    ts_s = np.sort(dedup["Time_stamp"].to_numpy(dtype=np.int64) * 1e-6)
    dt = np.diff(ts_s)
    dt = dt[dt > 0]
    fps_dedup = float(1.0 / np.median(dt)) if dt.size > 0 else 0.0

    # For sync we want dvapi_dedup_df order aligned with its timestamp column.
    # We'll sort by Time_stamp increasing to make nearest_map happy.
    dedup = dedup.sort_values("Time_stamp", kind="mergesort").reset_index(drop=True)
    dedup.insert(0, "dvapi_row", np.arange(len(dedup), dtype=np.int64))

    dvapi_ts_s = dedup["Time_stamp"].to_numpy(dtype=np.int64) * 1e-6
    return dedup, dvapi_ts_s.astype(np.float64), fps_raw, fps_dedup


def build_dvapi_mod(name: str, dvapi_ts_s: np.ndarray) -> Modality:
    ts_s = np.sort(dvapi_ts_s.astype(np.float64))
    dt_s = robust_median_dt(ts_s)
    fps = 1.0 / dt_s if dt_s > 0 else 0.0
    return Modality(name=name, ts_s=ts_s, dt_s=dt_s, fps=fps, unit_desc="dvapi_epoch_us_dedup")


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Sync left/right/side ISO + gaze TXT + dvapi CSV (dedup by api_cnt).")
    ap.add_argument("--config", required=True, help="Path to config.json")
    ap.add_argument("--out_csv", default="sync_table.csv", help="Filtered sync table (kept left frames)")
    ap.add_argument("--out_full_debug", default="sync_table_full_debug.csv", help="Full tref table with err/valid flags (for QC)")
    ap.add_argument("--dvapi_dedup_csv", default="dvapi_dedup.csv", help="Output dedup dvapi table (one row per api_cnt)")
    ap.add_argument("--dvapi_aligned_csv", default="dvapi_aligned.csv", help="Output dvapi table aligned to kept frames (expanded per frame)")
    ap.add_argument("--max_err_ms", type=float, default=33.33,
                    help="Drop tref if ANY modality nearest error exceeds this threshold (ms). Default 33.33ms.")
    ap.add_argument("--write_full_debug", action="store_true", help="If set, write full debug table.")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    tz_name = cfg.get("timezone", "America/Vancouver")

    left_path = cfg["left_ts"]
    right_path = cfg["right_ts"]
    side_path = cfg["side_ts"]
    gaze_path = cfg["gaze_log"]

    # NEW: dvapi
    dvapi_csv_path = cfg.get("dvapi_csv")
    if not dvapi_csv_path:
        raise ValueError("config.json missing 'dvapi_csv' path for dvapi data.")

    left_col = cfg.get("left_col")
    right_col = cfg.get("right_col")
    side_col = cfg.get("side_col")

    # Read timestamps (videos + gaze)
    left_iso = read_video_iso_timestamps(left_path, csv_col=left_col)
    right_iso = read_video_iso_timestamps(right_path, csv_col=right_col)
    side_iso = read_video_iso_timestamps(side_path, csv_col=side_col)

    left = build_video_mod("left", left_iso, tz_name)
    right = build_video_mod("right", right_iso, tz_name)
    side = build_video_mod("side", side_iso, tz_name)
    gaze = build_gaze_mod_from_txt("gaze", gaze_path)

    # Read dvapi and dedup
    dvapi_dedup_df, dvapi_ts_s, dvapi_fps_raw, dvapi_fps_dedup = read_dvapi_csv_dedup_by_api_cnt(dvapi_csv_path)
    dvapi = build_dvapi_mod("dvapi", dvapi_ts_s)

    print("=== DVAPI FPS estimate ===")
    print(f"dvapi raw fps≈{dvapi_fps_raw:.3f}")
    print(f"dvapi dedup-by-api_cnt fps≈{dvapi_fps_dedup:.3f} (this is used for sync)")
    print(f"dvapi dedup rows: {len(dvapi_dedup_df)}")
    dvapi_dedup_df.to_csv(args.dvapi_dedup_csv, index=False)
    print(f"Saved dvapi dedup table: {args.dvapi_dedup_csv}")

    # Compute overlap among all modalities INCLUDING dvapi
    mods = {"left": left, "right": right, "side": side, "gaze": gaze, "dvapi": dvapi}

    print("\n=== Modality stats (UTC epoch seconds) ===")
    for k, m in mods.items():
        print(f"{k:>5s}: {m.unit_desc:>20s}, dt={m.dt_s:.6f}s, fps≈{m.fps:.3f}, "
              f"start={m.ts_s[0]:.3f}, end={m.ts_s[-1]:.3f}, n={m.ts_s.size}")

    t0, t1 = compute_overlap(mods)
    print(f"\n=== Overlap window ===\nstart={t0:.3f}, end={t1:.3f}, duration={t1-t0:.3f}s")

    # tref = left timestamps within overlap
    tref = left.ts_s[(left.ts_s >= t0) & (left.ts_s <= t1)]
    if tref.size < 2:
        raise ValueError("Left timestamps within overlap are too few (<2).")

    tref_dt = robust_median_dt(tref)
    print(f"\n=== tref ===\nUsing left timestamps in overlap: {tref.size} samples, approx fps≈{1.0/tref_dt:.3f}")

    # left indices for tref
    left_idx = np.searchsorted(left.ts_s, tref, side="left").astype(np.int64)

    # Nearest mapping
    right_idx, right_err = nearest_map(tref, right.ts_s)
    side_idx, side_err = nearest_map(tref, side.ts_s)
    gaze_idx, gaze_err = nearest_map(tref, gaze.ts_s)

    # dvapi mapping: note dvapi.ts_s is sorted, but we also need the dvapi_row/api_cnt
    dvapi_idx, dvapi_err = nearest_map(tref, dvapi.ts_s)

    thr_s = args.max_err_ms * 1e-3
    right_ok = np.abs(right_err) <= thr_s
    side_ok = np.abs(side_err) <= thr_s
    gaze_ok = np.abs(gaze_err) <= thr_s
    dvapi_ok = np.abs(dvapi_err) <= thr_s

    valid = right_ok & side_ok & gaze_ok & dvapi_ok

    print(f"\n=== Nearest-match error QC ===")
    print(abs_err_stats("right", right_err))
    print(abs_err_stats(" side", side_err))
    print(abs_err_stats(" gaze", gaze_err))
    print(abs_err_stats("dvapi", dvapi_err))
    print(f"threshold = {thr_s:.6f}s ({args.max_err_ms:.2f} ms)")
    print(f"valid rows (all modalities within threshold): {int(valid.sum())}/{len(valid)}")

    # Build full debug table (optional)
    # dvapi fields we put in sync table: dvapi_row, api_cnt, err
    # dvapi_idx refers to index in dvapi.ts_s sorted array, but dvapi_dedup_df is also sorted by Time_stamp so it matches.
    dvapi_row = dvapi_dedup_df.iloc[dvapi_idx]["dvapi_row"].to_numpy(dtype=np.int64)
    dvapi_api_cnt = dvapi_dedup_df.iloc[dvapi_idx]["api_cnt"].to_numpy(dtype=np.int64)

    df_full = pd.DataFrame({
        "t_ref_s": tref,
        "left_idx": left_idx,

        "right_idx": right_idx,
        "side_idx": side_idx,
        "gaze_idx": gaze_idx,
        "dvapi_idx": dvapi_idx,
        "dvapi_row": dvapi_row,
        "dvapi_api_cnt": dvapi_api_cnt,

        "right_t_s": right.ts_s[right_idx],
        "side_t_s": side.ts_s[side_idx],
        "gaze_t_s": gaze.ts_s[gaze_idx],
        "dvapi_t_s": dvapi.ts_s[dvapi_idx],

        "right_err_s": right_err,
        "side_err_s": side_err,
        "gaze_err_s": gaze_err,
        "dvapi_err_s": dvapi_err,

        "right_ok": right_ok,
        "side_ok": side_ok,
        "gaze_ok": gaze_ok,
        "dvapi_ok": dvapi_ok,
        "valid": valid,
    })

    if args.write_full_debug:
        df_full.to_csv(args.out_full_debug, index=False)
        print(f"Saved full debug table: {args.out_full_debug}")

    # Filtered sync table
    df_sync = df_full[df_full["valid"]].reset_index(drop=True)

    df_sync = df_sync[[
        # reference
        "t_ref_s",        # = left overlap timestamp (UTC epoch seconds)
        "left_idx",

        # right
        "right_idx",
        "right_t_s",

        # side
        "side_idx",
        "side_t_s",

        # gaze
        "gaze_idx",
        "gaze_t_s",

        # dvapi
        "dvapi_idx",
        "dvapi_row",
        "dvapi_api_cnt",
        "dvapi_t_s",
    ]]

    df_sync.to_csv(args.out_csv, index=False)
    print(f"\nSaved filtered sync table: {args.out_csv}")
    print(f"Final kept rows: {len(df_sync)}")

    # Synced FPS (based on kept tref)
    if len(df_sync) >= 2:
        synced_tref = df_sync["t_ref_s"].to_numpy()
        synced_dt = np.diff(synced_tref)
        synced_dt = synced_dt[synced_dt > 0]
        if synced_dt.size > 0:
            fps_sync = 1.0 / np.median(synced_dt)
            print(f"\n=== Synced FPS ===")
            print(f"Synced left FPS ≈ {fps_sync:.3f}")
            print(f"Kept frames: {len(df_sync)}")
            print(f"Duration: {synced_tref[-1] - synced_tref[0]:.3f} s")
    keep_ratio = len(df_sync) / len(tref)
    print(f"Keep ratio: {keep_ratio*100:.2f}%")

    # Build dvapi aligned table (expanded per kept frame): join by dvapi_row
    # This gives you per-frame dvapi pose/joint columns aligned to the kept frames.
    dvapi_cols = list(dvapi_dedup_df.columns)  # includes dvapi_row, api_cnt, Time_stamp, ...
    dvapi_aligned = df_sync[["t_ref_s", "left_idx", "dvapi_row", "dvapi_api_cnt"]].merge(
        dvapi_dedup_df,
        on="dvapi_row",
        how="left",
        suffixes=("", "_dvapi"),
    )
    dvapi_aligned.to_csv(args.dvapi_aligned_csv, index=False)
    print(f"Saved dvapi aligned table: {args.dvapi_aligned_csv}")


if __name__ == "__main__":
    main()
