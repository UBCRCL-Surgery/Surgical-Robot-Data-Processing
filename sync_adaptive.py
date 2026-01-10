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
# Read video timestamps
# -------------------------

def read_text_lines(path: str) -> np.ndarray:
    vals = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if "," in s:
                s = s.split(",")[-1].strip()
            vals.append(s)
    return np.array(vals, dtype=object)


def read_video_iso_timestamps(path: str, csv_col: Optional[str] = None) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".txt", ".log"]:
        return read_text_lines(path)

    df = pd.read_csv(path, dtype=str)
    if df.shape[1] == 0:
        raise ValueError(f"{path} has no columns.")
    col = df.columns[0] if csv_col is None else csv_col
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in {path}.")
    return df[col].astype(str).to_numpy()


def parse_iso_naive_to_epoch_seconds(iso_strings: np.ndarray, tz_name: str) -> np.ndarray:
    s = pd.Series(_to_1d(iso_strings).astype(str))
    dt = pd.to_datetime(s, errors="coerce", format="ISO8601")
    if dt.isna().any():
        raise ValueError("Failed to parse ISO timestamps.")
    tz = ZoneInfo(tz_name)
    dt_local = dt.dt.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    dt_utc = dt_local.dt.tz_convert("UTC")
    return (dt_utc.astype("int64").to_numpy() / 1e9).astype(np.float64)


def build_video_mod(name: str, iso_ts: np.ndarray, tz_name: str) -> Modality:
    ts_s = np.sort(parse_iso_naive_to_epoch_seconds(iso_ts, tz_name))
    dt_s = robust_median_dt(ts_s)
    fps = 1.0 / dt_s if dt_s > 0 else 0.0
    return Modality(name, ts_s, dt_s, fps, "iso_local->utc_s")


# -------------------------
# Gaze
# -------------------------

def read_gaze_epoch_us_from_txt(gaze_path: str) -> np.ndarray:
    pat = re.compile(r"(?<!\d)(\d{13,}(?:\.\d+)?)(?!\d)")
    ts_us = []
    with open(gaze_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pat.search(line)
            if m:
                ts_us.append(float(m.group(1)))
    if len(ts_us) < 10:
        raise ValueError("Too few gaze timestamps.")
    return np.sort(np.array(ts_us) * 1e-6)


def build_gaze_mod_from_txt(name: str, gaze_path: str) -> Modality:
    ts_s = read_gaze_epoch_us_from_txt(gaze_path)
    dt_s = robust_median_dt(ts_s)
    fps = 1.0 / dt_s
    return Modality(name, ts_s, dt_s, fps, "gaze_epoch_us")


# -------------------------
# DVAPI
# -------------------------

def read_dvapi_csv_dedup_by_api_cnt(dvapi_csv_path: str):
    df = pd.read_csv(dvapi_csv_path)
    df = df.dropna(subset=["Time_stamp", "api_cnt"])
    df["Time_stamp"] = df["Time_stamp"].astype(np.int64)
    df["api_cnt"] = df["api_cnt"].astype(np.int64)

    df = df.sort_values(["api_cnt", "Time_stamp"])
    dedup = df.groupby("api_cnt", as_index=False).first()
    dedup = dedup.sort_values("Time_stamp").reset_index(drop=True)
    dedup.insert(0, "dvapi_row", np.arange(len(dedup)))

    ts_s = dedup["Time_stamp"].to_numpy() * 1e-6
    return dedup, ts_s.astype(np.float64)


def build_dvapi_mod(name: str, dvapi_ts_s: np.ndarray) -> Modality:
    ts_s = np.sort(dvapi_ts_s)
    dt_s = robust_median_dt(ts_s)
    fps = 1.0 / dt_s
    return Modality(name, ts_s, dt_s, fps, "dvapi_epoch_us")


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_csv", default="sync_table.csv")
    ap.add_argument("--max_err_ms", type=float, default=33.33)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    tz_name = cfg.get("timezone", "America/Vancouver")

    # ---- Paths (None allowed) ----
    left_path  = cfg["left_ts"]
    right_path = cfg.get("right_ts")
    side_path  = cfg.get("side_ts")
    gaze_path  = cfg.get("gaze_log")
    dvapi_path = cfg.get("dvapi_csv")

    # ---- Build modalities ----
    left = build_video_mod("left",
        read_video_iso_timestamps(left_path), tz_name)

    right = build_video_mod("right",
        read_video_iso_timestamps(right_path), tz_name) if right_path else None

    side = build_video_mod("side",
        read_video_iso_timestamps(side_path), tz_name) if side_path else None

    gaze = build_gaze_mod_from_txt("gaze", gaze_path) if gaze_path else None

    if dvapi_path:
        dvapi_df, dvapi_ts_s = read_dvapi_csv_dedup_by_api_cnt(dvapi_path)
        dvapi = build_dvapi_mod("dvapi", dvapi_ts_s)
    else:
        dvapi_df = None
        dvapi = None

    # ---- Active modalities ----
    mods = {"left": left}
    if right: mods["right"] = right
    if side:  mods["side"] = side
    if gaze:  mods["gaze"] = gaze
    if dvapi: mods["dvapi"] = dvapi

    print("Active modalities:", list(mods.keys()))

    # ---- Overlap ----
    t0, t1 = compute_overlap(mods)
    tref = left.ts_s[(left.ts_s >= t0) & (left.ts_s <= t1)]
    left_idx = np.searchsorted(left.ts_s, tref)

    N = len(tref)
    thr_s = args.max_err_ms * 1e-3

    # ---- Init defaults ----
    def init():
        return np.full(N, -1), np.full(N, np.nan), np.ones(N, dtype=bool)

    right_idx, right_err, right_ok = init()
    side_idx,  side_err,  side_ok  = init()
    gaze_idx,  gaze_err,  gaze_ok  = init()
    dvapi_idx, dvapi_err, dvapi_ok = init()

    # ---- Conditional sync ----
    if right:
        right_idx, right_err = nearest_map(tref, right.ts_s)
        right_ok = np.abs(right_err) <= thr_s

    if side:
        side_idx, side_err = nearest_map(tref, side.ts_s)
        side_ok = np.abs(side_err) <= thr_s

    if gaze:
        gaze_idx, gaze_err = nearest_map(tref, gaze.ts_s)
        gaze_ok = np.abs(gaze_err) <= thr_s

    if dvapi:
        dvapi_idx, dvapi_err = nearest_map(tref, dvapi.ts_s)
        dvapi_ok = np.abs(dvapi_err) <= thr_s

    valid = right_ok & side_ok & gaze_ok & dvapi_ok

    # ---- Build output ----
    df = pd.DataFrame({
        "t_ref_s": tref,
        "left_idx": left_idx,

        "right_idx": right_idx if right else np.nan,
        "side_idx":  side_idx  if side  else np.nan,
        "gaze_idx":  gaze_idx  if gaze  else np.nan,
        "dvapi_idx": dvapi_idx if dvapi else np.nan,

        "valid": valid
    })

    df = df[df["valid"]].reset_index(drop=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
