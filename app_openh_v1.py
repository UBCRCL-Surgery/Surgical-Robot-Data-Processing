from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel


# ============================================================
# Sync already done -> Episode labeling only
#
# Ingest mode:
#   B) Server-path ingest (provide absolute paths on the server; no upload/copy of video)
#
# Improvements in this version:
#   - Robust video serving with HTTP Range support (fixes "video not showing" in some browsers)
#   - FPS estimation auto-detects timestamp units (seconds / ms / us / ns)
# ============================================================

APP_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("SURG_GUI_DATA_ROOT", str(APP_ROOT / "data" / "projects")))
DATA_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Surgical Labeling GUI (Sync Done) - Path Ingest Edition (Range Streaming)")



@app.get("/debug/paths")
def debug_paths():
    return {
        "app_root": str(APP_ROOT),
        "data_root": str(DATA_ROOT),
        "cwd": os.getcwd(),
        "exists_data_root": DATA_ROOT.exists(),
        "projects": sorted([p.name for p in DATA_ROOT.glob("*") if p.is_dir()])[:50],
    }


# -------------------------
# FS helpers
# -------------------------
def ensure_project_dir(project_id: str) -> Path:
    pdir = DATA_ROOT / project_id
    (pdir / "index").mkdir(parents=True, exist_ok=True)
    (pdir / "episodes").mkdir(parents=True, exist_ok=True)
    (pdir / "media").mkdir(parents=True, exist_ok=True)
    (pdir / "exports").mkdir(parents=True, exist_ok=True)
    return pdir


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_filename(name: str) -> str:
    name = name.replace("\\", "/").split("/")[-1].strip()
    return name or "file"


# -------------------------
# Data schema
# -------------------------
def default_label_schema() -> Dict[str, Any]:
    return {
        "labels": [
            {"id": 0, "name": "Gesture0", "color": "#4C78A8"},
            {"id": 1, "name": "Gesture1", "color": "#F58518"},
            {"id": 2, "name": "Gesture2", "color": "#54A24B"},
        ],
        "unlabeled": {"id": -1, "name": "Unlabeled", "color": "#B0B0B0"},
    }


def normalize_sync_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.reset_index(drop=True, inplace=True)
    if "row_idx" not in df.columns:
        df.insert(0, "row_idx", np.arange(len(df), dtype=np.int64))
    else:
        df["row_idx"] = pd.to_numeric(df["row_idx"], errors="coerce").fillna(-1).astype(np.int64)

    if "left_frame" not in df.columns:
        df.insert(1, "left_frame", df["row_idx"].astype(np.int64))
    else:
        df["left_frame"] = pd.to_numeric(df["left_frame"], errors="coerce").fillna(-1).astype(np.int64)
    return df


# -------------------------
# Video probe
# -------------------------
def ffprobe_video_info(video_path: Path) -> Dict[str, Any]:
    if not video_path.exists():
        raise FileNotFoundError(str(video_path))

    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate,nb_frames,codec_name,codec_type,profile,pix_fmt",
        "-show_entries", "format=duration,format_name",
        "-of", "json",
        str(video_path),
    ]
    out = subprocess.check_output(cmd).decode("utf-8", errors="ignore")
    j = json.loads(out)

    duration_s = None
    fps = None
    nb_frames = None
    codec = None
    fmt = None

    if "format" in j:
        fmt = j["format"].get("format_name")
        if j["format"].get("duration") is not None:
            try:
                duration_s = float(j["format"]["duration"])
            except Exception:
                duration_s = None

    streams = j.get("streams", [])
    if streams:
        s0 = streams[0]
        codec = s0.get("codec_name")
        afr = s0.get("avg_frame_rate")
        if afr and isinstance(afr, str) and "/" in afr:
            num, den = afr.split("/", 1)
            try:
                numf = float(num)
                denf = float(den)
                if denf != 0:
                    fps = numf / denf
            except Exception:
                pass
        nf = s0.get("nb_frames")
        if nf and nf != "N/A":
            try:
                nb_frames = int(nf)
            except Exception:
                nb_frames = None

    if nb_frames is None:
        cmd2 = [
            "ffprobe", "-v", "error",
            "-count_frames",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames",
            "-of", "default=nokey=1:noprint_wrappers=1",
            str(video_path),
        ]
        try:
            out2 = subprocess.check_output(cmd2).decode("utf-8", errors="ignore").strip()
            if out2:
                nb_frames = int(out2)
        except Exception:
            nb_frames = None

    return {"duration_s": duration_s, "fps": fps, "nb_frames": nb_frames, "codec": codec, "format": fmt}


# -------------------------
# FPS estimation (auto units)
# -------------------------
def estimate_fps_from_ts(ts: np.ndarray) -> float:
    """
    Accept timestamps in seconds / ms / us / ns (int or float).
    Heuristic:
      - If median(ts) ~ 1e9 and diffs ~ 1e-2..1e-1 => seconds
      - If median(ts) ~ 1e12 => ms
      - If median(ts) ~ 1e15 => us
      - If median(ts) ~ 1e18 => ns
    """
    ts = np.asarray(ts)
    ts = ts[np.isfinite(ts)]
    if ts.size < 2:
        return 0.0
    ts = ts.astype(np.float64)
    ts.sort()

    med = float(np.median(ts))
    # unit scale to seconds
    if med >= 1e17:
        scale = 1e-9   # ns -> s
    elif med >= 1e14:
        scale = 1e-6   # us -> s
    elif med >= 1e11:
        scale = 1e-3   # ms -> s
    else:
        scale = 1.0    # s

    dur_s = float((ts[-1] - ts[0]) * scale)
    if dur_s <= 0:
        return 0.0
    return float((ts.size - 1) / dur_s)


# -------------------------
# Range streaming for video
# -------------------------
_RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def _parse_range(range_header: str, file_size: int) -> Optional[Tuple[int, int]]:
    """
    Return (start, end) inclusive, or None if invalid.
    Supports:
      bytes=start-end
      bytes=start-
      bytes=-suffix
    """
    m = _RANGE_RE.match(range_header.strip())
    if not m:
        return None
    s0, s1 = m.group(1), m.group(2)

    if s0 == "" and s1 == "":
        return None

    if s0 != "":
        start = int(s0)
        end = int(s1) if s1 != "" else file_size - 1
    else:
        # suffix
        suffix = int(s1)
        if suffix <= 0:
            return None
        start = max(0, file_size - suffix)
        end = file_size - 1

    start = max(0, min(start, file_size - 1))
    end = max(0, min(end, file_size - 1))
    if end < start:
        return None
    return start, end


def _iter_file_range(path: Path, start: int, end: int, chunk_size: int = 1024 * 1024) -> Iterable[bytes]:
    with path.open("rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            n = min(chunk_size, remaining)
            buf = f.read(n)
            if not buf:
                break
            yield buf
            remaining -= len(buf)


def guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".mp4":
        return "video/mp4"
    if ext == ".webm":
        return "video/webm"
    if ext == ".ogg" or ext == ".ogv":
        return "video/ogg"
    return "application/octet-stream"


# -------------------------
# File IO helpers
# -------------------------
def load_validate_write_sync(sync_src: Path, sync_dst: Path) -> int:
    try:
        df = pd.read_csv(sync_src)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read sync_table.csv: {e}")
    if len(df) == 0:
        raise HTTPException(status_code=400, detail="sync_table.csv is empty")
    df = normalize_sync_table(df)
    df.to_csv(sync_dst, index=False)
    return int(len(df))


def load_validate_write_label_schema(schema_path: Optional[Path], out_path: Path) -> None:
    if schema_path is None:
        if not out_path.exists():
            write_json(out_path, default_label_schema())
        return
    if not schema_path.exists():
        raise HTTPException(status_code=400, detail=f"label_schema.json not found: {schema_path}")
    try:
        obj = json.loads(schema_path.read_text(encoding="utf-8"))
        if "labels" not in obj:
            raise ValueError("label schema must contain 'labels'")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid label schema json: {e}")
    write_json(out_path, obj)


def episode_path(pdir: Path, episode_id: str) -> Path:
    return pdir / "episodes" / f"{episode_id}.json"


def list_episodes(pdir: Path) -> List[Dict[str, Any]]:
    out = []
    for p in sorted((pdir / "episodes").glob("*.json")):
        try:
            out.append(read_json(p))
        except Exception:
            continue
    out.sort(key=lambda x: (x.get("start_idx", 10**18), x.get("end_idx", 10**18)))
    return out


# =========================
# Models
# =========================
class EpisodeCreateReq(BaseModel):
    start_idx: int
    end_idx: int
    label: int = -1
    name: str = "episode"


class EpisodeUpdateReq(BaseModel):
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None
    label: Optional[int] = None
    name: Optional[str] = None


class IngestPathReq(BaseModel):
    left_video_path: Optional[str] = None
    sync_table_csv: Optional[str] = None
    label_schema_json: Optional[str] = None
    config_json_path: Optional[str] = None
# =========================
# Project
# =========================
@app.post("/projects")
def create_project():
    project_id = uuid.uuid4().hex[:10]
    ensure_project_dir(project_id)
    return {"project_id": project_id}


@app.get("/sample/label_schema.json")
def sample_label_schema():
    return JSONResponse(default_label_schema())


# =========================
# Ingest (Server-path ONLY)
# =========================
@app.post("/projects/{project_id}/ingest_path")
def ingest_path(project_id: str, req: IngestPathReq):
    pdir = ensure_project_dir(project_id)

    config_json = read_json(Path(req.config_json_path))
    base_path = config_json.get("base_path")
    left_video_path = base_path + config_json.get("out_video")
    sync_table_csv = base_path + config_json.get("sync_csv")
    label_schema_json = config_json.get("label_schema")

    vsrc = Path(left_video_path)
    if not vsrc.exists():
        raise HTTPException(status_code=400, detail=f"left_video_path not found on server: {vsrc}")

    ssrc = Path(sync_table_csv)
    if not ssrc.exists():
        raise HTTPException(status_code=400, detail=f"sync_table_csv not found on server: {ssrc}")

    # Copy sync_table into project (small & reproducible)
    spath = pdir / "index" / "sync_table.csv"
    n = load_validate_write_sync(ssrc, spath)

    # Copy label schema into project (optional)
    schema_out = pdir / "index" / "label_schema.json"
    if label_schema_json:
        load_validate_write_label_schema(Path(label_schema_json), schema_out)
    else:
        load_validate_write_label_schema(None, schema_out)

    # probe
    try:
        vinf = ffprobe_video_info(vsrc)
    except subprocess.CalledProcessError:
        raise HTTPException(status_code=500, detail="ffprobe failed. Please install ffmpeg/ffprobe on the server.")
    v_frames = vinf.get("nb_frames")
    warnings: List[str] = []
    if v_frames is not None and int(v_frames) != n:
        raise HTTPException(status_code=400, detail=f"Frame mismatch: video frames={v_frames}, sync rows={n}.")
    if v_frames is None:
        warnings.append("ffprobe could not determine nb_frames; skipped strict frame-count check.")

    # fps
    fps_est = 0.0
    # Prefer ffprobe FPS (timestamp column is optional and not required for the labeling UI)
    if vinf.get("fps") is not None:
        fps_est = float(vinf["fps"])
    if fps_est <= 0:
        fps_est = 30.0
        warnings.append("Could not estimate FPS; defaulted to 30.0.")
    meta = {
        "project_id": project_id,
        "created": datetime.now().isoformat(),
        "ingest_mode": "path",
        "paths": {
            "left_video_path": str(vsrc),          # reference original location
            "sync_table_csv": str(spath),          # normalized copy inside project
            "label_schema_json": str(schema_out),  # copy inside project
        },
        "n_rows": n,
        "fps_est": float(fps_est),
        "video_info": vinf,
    }
    write_json(pdir / "metadata.json", meta)
    return {"ok": True, "project_id": project_id, "n_rows": n, "fps_est": float(fps_est), "video_info": vinf, "warnings": warnings}


# =========================
# Media serving (Range)
# =========================
@app.get("/projects/{project_id}/media/left")
def get_left_media(project_id: str, request: Request):
    """
    Serve the video with HTTP Range support.
    This fixes cases where <video> tag fails to load with plain FileResponse.
    """
    pdir = ensure_project_dir(project_id)
    meta_path = pdir / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Project not ingested yet.")
    meta = read_json(meta_path)
    vp = Path(meta["paths"]["left_video_path"])
    if not vp.exists():
        raise HTTPException(status_code=404, detail=f"Left video missing: {vp}")

    file_size = vp.stat().st_size
    range_header = request.headers.get("range")
    mime = guess_mime(vp)

    if not range_header:
        # Fallback: whole file (still works for many cases)
        return FileResponse(str(vp), media_type=mime)

    parsed = _parse_range(range_header, file_size)
    if parsed is None:
        # Invalid Range
        return StreamingResponse(iter([b""]), status_code=416, headers={"Content-Range": f"bytes */{file_size}"})

    start, end = parsed
    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
    }
    return StreamingResponse(
        _iter_file_range(vp, start, end),
        status_code=206,
        media_type=mime,
        headers=headers,
    )


# =========================
# Data access
# =========================
@app.get("/projects/{project_id}/sync_table")
def get_sync_table_info(project_id: str):
    pdir = ensure_project_dir(project_id)
    meta_path = pdir / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Project not ingested yet.")
    meta = read_json(meta_path)
    sp = Path(meta["paths"]["sync_table_csv"])
    df = pd.read_csv(sp)
    return {
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "fps_est": meta.get("fps_est", 30.0),
        "video_info": meta.get("video_info", {}),
        "ingest_mode": meta.get("ingest_mode", "unknown"),
        "left_video_path": meta.get("paths", {}).get("left_video_path"),
    }


@app.get("/projects/{project_id}/label_schema")
def get_label_schema(project_id: str):
    pdir = ensure_project_dir(project_id)
    schema_path = pdir / "index" / "label_schema.json"
    if not schema_path.exists():
        write_json(schema_path, default_label_schema())
    return read_json(schema_path)


@app.post("/projects/{project_id}/label_schema")
def set_label_schema(project_id: str, schema: Dict[str, Any]):
    pdir = ensure_project_dir(project_id)
    schema_path = pdir / "index" / "label_schema.json"
    if "labels" not in schema:
        raise HTTPException(status_code=400, detail="schema must contain 'labels'")
    write_json(schema_path, schema)
    return {"ok": True}


# =========================
# Episodes
# =========================
@app.get("/projects/{project_id}/episodes")
def api_list_episodes(project_id: str):
    pdir = ensure_project_dir(project_id)
    return {"project_id": project_id, "episodes": list_episodes(pdir)}


@app.post("/projects/{project_id}/episodes")
def api_create_episode(project_id: str, req: EpisodeCreateReq):
    pdir = ensure_project_dir(project_id)
    meta_path = pdir / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=400, detail="Project not ingested yet.")
    meta = read_json(meta_path)
    n = int(meta.get("n_rows", 0))

    s, e = int(req.start_idx), int(req.end_idx)
    if s < 0 or e < 0 or s >= n or e >= n or e < s:
        raise HTTPException(status_code=400, detail=f"Invalid episode range: [{s}, {e}] with n_rows={n}")

    episode_id = uuid.uuid4().hex[:10]
    ep = {
        "episode_id": episode_id,
        "name": req.name,
        "start_idx": s,
        "end_idx": e,
        "label": int(req.label),
        "created": datetime.now().isoformat(),
    }
    write_json(episode_path(pdir, episode_id), ep)
    return {"ok": True, "episode": ep}


@app.delete("/projects/{project_id}/episodes/{episode_id}")
def api_delete_episode(project_id: str, episode_id: str):
    pdir = ensure_project_dir(project_id)
    p = episode_path(pdir, episode_id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="Episode not found.")
    p.unlink()
    return {"ok": True}

@app.patch("/projects/{project_id}/episodes/{episode_id}")
def api_update_episode(project_id: str, episode_id: str, req: EpisodeUpdateReq):
    pdir = ensure_project_dir(project_id)
    ep_path = episode_path(pdir, episode_id)

    if not ep_path.exists():
        raise HTTPException(status_code=404, detail="Episode not found.")

    ep = read_json(ep_path)

    meta = read_json(pdir / "metadata.json")
    n = int(meta.get("n_rows", 0))

    if req.start_idx is not None:
        if req.start_idx < 0 or req.start_idx >= n:
            raise HTTPException(status_code=400, detail="Invalid start_idx")
        ep["start_idx"] = int(req.start_idx)

    if req.end_idx is not None:
        if req.end_idx < 0 or req.end_idx >= n:
            raise HTTPException(status_code=400, detail="Invalid end_idx")
        ep["end_idx"] = int(req.end_idx)

    if ep["end_idx"] < ep["start_idx"]:
        raise HTTPException(status_code=400, detail="end_idx < start_idx")

    if req.label is not None:
        ep["label"] = int(req.label)

    if req.name is not None:
        ep["name"] = str(req.name)

    ep["modified"] = datetime.now().isoformat()
    write_json(ep_path, ep)

    return {"ok": True, "episode": ep}


# =========================
# Commit episode labels back to sync_table.csv
# =========================
def _parse_iso_dt(s: str) -> float:
    try:
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return 0.0


def apply_episodes_to_sync_table(pdir: Path) -> Dict[str, Any]:
    """Write two columns back into index/sync_table.csv:
    - episode_id: string, default "-1"
    - episode_label: int, default -1

    Episodes overwrite on overlap by 'created' time (later created wins).
    """
    meta_path = pdir / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="Project not ingested yet.")
    meta = read_json(meta_path)
    sp = Path(meta["paths"]["sync_table_csv"])
    if not sp.exists():
        raise HTTPException(status_code=404, detail="sync_table.csv not found.")

    df = pd.read_csv(sp)
    df = normalize_sync_table(df)
    n = len(df)

    eps = list_episodes(pdir)
    # sort by created time so latest wins
    eps_sorted = sorted(eps, key=lambda e: _parse_iso_dt(str(e.get("created", ""))))

    ep_id_col = np.full((n,), "-1", dtype=object)
    ep_label_col = np.full((n,), -1, dtype=np.int64)

    applied = 0
    for e in eps_sorted:
        s = int(e.get("start_idx", -1))
        t = int(e.get("end_idx", -1))
        if s < 0 or t < 0 or s >= n or t >= n or t < s:
            continue
        eid = str(e.get("episode_id", "-1"))
        lab = int(e.get("label", -1))
        ep_id_col[s:t+1] = eid
        ep_label_col[s:t+1] = lab
        applied += 1

    df["episode_id"] = ep_id_col
    df["episode_label"] = ep_label_col
    df.to_csv(sp, index=False)

    labeled_frames = int(np.sum(ep_label_col != -1))
    return {
        "ok": True,
        "n_rows": int(n),
        "episodes_applied": int(applied),
        "labeled_frames": labeled_frames,
        "labeled_ratio": float(labeled_frames / max(1, n)),
        "sync_table_csv": str(sp),
    }


@app.post("/projects/{project_id}/commit_labels")
def commit_labels(project_id: str):
    pdir = ensure_project_dir(project_id)
    return apply_episodes_to_sync_table(pdir)


# =========================
# Export
# =========================
@app.get("/projects/{project_id}/export")
def export_project(project_id: str):
    pdir = ensure_project_dir(project_id)
    meta_path = pdir / "metadata.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="metadata.json not found. Ingest first.")

    # Ensure sync_table.csv contains episode_id/episode_label columns before exporting
    try:
        apply_episodes_to_sync_table(pdir)
    except Exception:
        # Export should still work even if commit fails; episodes are still included separately.
        pass

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = pdir / "exports" / f"{project_id}_export_{ts}.zip"

    def add_if_exists(zf: zipfile.ZipFile, fpath: Path):
        if fpath.exists() and fpath.is_file():
            zf.write(fpath, arcname=str(fpath.relative_to(pdir)))

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        add_if_exists(zf, pdir / "metadata.json")
        add_if_exists(zf, pdir / "index" / "sync_table.csv")
        add_if_exists(zf, pdir / "index" / "label_schema.json")
        for f in sorted((pdir / "episodes").glob("*.json")):
            add_if_exists(zf, f)

    return FileResponse(path=str(zip_path), media_type="application/zip", filename=zip_path.name)


# =========================
# UI
# =========================
@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Labeling GUI - Home</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 18px; }
    .card { border: 1px solid #ddd; padding: 12px; border-radius: 10px; background: #fff; max-width: 980px; }
    button { padding: 8px 12px; cursor: pointer; }
    .hint { color: #666; font-size: 12px; line-height: 1.4; margin-top: 8px; }
    code { background: #f6f6f6; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <h2>Sync-Done Episode Labeling GUI</h2>
  <div class="card">
    <button id="btnNew">Create New Project</button>
    <div class="hint">
      Each project uses server-path ingest:
      <ul>
        <li><b>Server-path ingest</b>: directly enter the absolute server path (no video copying)</li>
      </ul>
      Example label json: <a href="/sample/label_schema.json" target="_blank">sample/label_schema.json</a>
    </div>
    <div id="out" style="margin-top:10px;"></div>
  </div>

<script>
document.getElementById('btnNew').onclick = async () => {
  const res = await fetch('/projects', {method:'POST'});
  const j = await res.json();
  if (!res.ok) { alert(JSON.stringify(j)); return; }
  const pid = j.project_id;
  document.getElementById('out').innerHTML =
    'Created: <code>' + pid + '</code><br/>' +
    '<a href="/projects/' + pid + '/ui">Open UI</a>';
};
</script>
</body>
</html>
""")


@app.get("/projects/{project_id}/ui", response_class=HTMLResponse)
def ui(project_id: str):
    ensure_project_dir(project_id)
    tpl = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Project __PID__ - Episodes & Labels</title>
  <style>
    body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 16px; }
    .row { display: flex; gap: 12px; align-items: flex-start; flex-wrap: wrap; }
    .card { border: 1px solid #ddd; padding: 12px; border-radius: 10px; background: #fff; }
    video { width: min(960px, 100%); border: 1px solid #ccc; border-radius: 8px; background:#000; }
    .panel { display:flex; gap: 8px; align-items: center; flex-wrap: wrap; margin-top: 10px; }
    button { padding: 6px 10px; cursor: pointer; }
    input { padding: 4px; }
    #episodesDump { white-space: pre; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono"; font-size: 12px; max-height: 280px; overflow: auto; }
    #bar { width: 100%; height: 86px; border: 1px solid #ccc; border-radius: 8px; }
    textarea { width: 100%; height: 200px; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono"; }
    .hint { color: #666; font-size: 12px; margin-top: 8px; line-height: 1.4; }
    .stats { margin-top: 8px; color: #222; font-size: 13px; }
    .pill { display:inline-block; padding: 2px 8px; border-radius: 999px; border: 1px solid #ddd; background:#fafafa; font-size: 12px; }
    .ok { color: #0a7a0a; }
    .ingestBox { display:flex; gap:10px; align-items:flex-end; flex-wrap:wrap; }
    .ingestBox label { font-size: 12px; color:#333; display:block; margin-bottom:4px; }
    .pathRow { display:flex; gap:10px; flex-wrap:wrap; }
    .pathRow input { width: 420px; }
    .kv { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas; font-size: 12px; }
  </style>
</head>
<body>
  <h2>Project __PID__</h2>

  <div class="card" style="margin-bottom:12px;">
    <b>Ingest</b>
    <div class="hint">
      Example label json：<a href="/sample/label_schema.json" target="_blank">sample/label_schema.json</a>
    </div>

    <div class="kv" id="debugInfo" style="margin-top:8px;"></div>

    <hr/>

    <div class="pathRow" style="margin-top:8px;">
      <div>
        <label>config.json</label>
        <input id="pConfigJson" placeholder="/data/.../config.json"/>
      </div>
<button id="btnIngestPath">Ingest from Paths</button>
      <span id="ingestStatus" class="pill">not ingested</span>
    </div>

    <div class="hint">
      <b> if page shows /sync_table 404 (but you are sure you have ingested)</b>: Check in browser DevTools → Network whether <code>/projects/__PID__/media/left</code> returns 206/200 or 404/403.<br/>
      Also make sure the video codec is supported by browser (H.264/AVC in mp4 is recommended).
    </div>
  </div>

  <div class="row">
    <div class="card" style="flex: 1 1 640px;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <b>Left video</b>
        <span class="pill" id="fpsPill">FPS: -</span>
      </div>
      <video id="vL" controls></video>

      <div class="panel">
        <button id="btnSetIn">Set IN</button>
        <button id="btnSetOut">Set OUT</button>
        <span>IN idx <b id="inIdx">-</b></span>
        <span>OUT idx <b id="outIdx">-</b></span>
      </div>

      <div class="panel">
        <input id="epName" value="episode" />
        <span>Label:</span>
        <input id="labelId" type="number" value="0" style="width:80px"/>
        <button id="btnCreateEp">Create Episode</button>
        <button id="btnRefresh">Refresh</button>
        <button id="btnCommit">Commit labels</button>
        <button id="btnExport">Export (.zip)</button>
      </div>
    </div>

    <div class="card" style="flex: 1 1 520px; min-width: 420px;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <b>Label Timeline</b>
        <div>
          <button id="btnReloadSchema">Reload schema</button>
          <button id="btnSaveSchema">Save schema</button>
        </div>
      </div>

      <canvas id="bar"></canvas>
      <div class="stats" id="stats">waiting ingest...</div>

      <div style="margin-top:10px;">
        <b>Label schema (id → name + color)</b>
        <textarea id="schemaBox"></textarea>
      </div>
    </div>
  </div>

  <div class="card" style="margin-top:12px;">
    <b>Episodes (JSON)</b>
    <div id="episodesDump">waiting ingest...</div>
  </div>
  <div class="card" style="margin-top:12px;">
    <b>Edit Episode</b>

    <div class="panel">
        <span>ID</span>
        <input id="editEpId" disabled style="width:160px"/>
    </div>

    <div class="panel">
        <span>Name</span>
        <input id="editEpName" style="width:200px"/>
    </div>

    <div class="panel">
        <span>Label</span>
        <input id="editEpLabel" type="number" style="width:80px"/>
    </div>

    <div class="panel">
        <span>Start</span>
        <input id="editEpStart" type="number" style="width:100px"/>
        <button id="btnUseIn">Use IN</button>
    </div>

    <div class="panel">
        <span>End</span>
        <input id="editEpEnd" type="number" style="width:100px"/>
        <button id="btnUseOut">Use OUT</button>
    </div>

    <div class="panel">
        <button id="btnSaveEp">Save</button>
        <button id="btnDeleteEp">Delete</button>
    </div>
  </div>

  
<script>
let selectedEpisode = null;

const PID = "__PID__";
const vL = document.getElementById('vL');
const fpsPill = document.getElementById('fpsPill');
const ingestStatus = document.getElementById('ingestStatus');
const debugInfo = document.getElementById('debugInfo');

const inIdxEl = document.getElementById('inIdx');
const outIdxEl = document.getElementById('outIdx');

const episodesDump = document.getElementById('episodesDump');
const schemaBox = document.getElementById('schemaBox');

const canvas = document.getElementById('bar');
const ctx = canvas.getContext('2d');
const statsDiv = document.getElementById('stats');

let fps = 30.0;
let nRows = 0;
let idxIn = null, idxOut = null;

function clamp(x,a,b){ return Math.max(a, Math.min(b, x)); }

function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.floor(canvas.clientWidth * dpr);
  canvas.height = Math.floor(canvas.clientHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function timeToIdx(t) {
  if (!isFinite(t)) return null;
  const dur = vL.duration;
  if (isFinite(dur) && dur > 0 && nRows > 1) {
    return clamp(Math.round((t / dur) * (nRows - 1)), 0, Math.max(0, nRows - 1));
  }
  // fallback: use fps if duration is not available
  return clamp(Math.round(t * fps), 0, Math.max(0, nRows-1));
}
function idxToTime(i) {
  const dur = vL.duration;
  if (isFinite(dur) && dur > 0 && nRows > 1) {
    return (i / (nRows - 1)) * dur;
  }
  return i / fps;
}

async function tryLoadSyncInfo() {
  const res = await fetch(`/projects/${PID}/sync_table`);
  if (!res.ok) return null;
  return await res.json();
}

async function loadSyncInfo() {
  const j = await (await fetch(`/projects/${PID}/sync_table`)).json();
  nRows = j.n_rows || 0;
  fps = (j.fps_est || 30.0);
  fpsPill.textContent = "FPS: " + fps.toFixed(3) + " | N: " + nRows;
  ingestStatus.textContent = "ingested";
  ingestStatus.classList.add("ok");

  const vi = j.video_info || {};
  debugInfo.textContent =
    "ingest_mode=" + (j.ingest_mode || "?") + " | " +
    "left_video_path=" + (j.left_video_path || "?") + "\n" +
    "video codec=" + (vi.codec || "?") + " | format=" + (vi.format || "?") + " | nb_frames=" + (vi.nb_frames ?? "?") + " | fps=" + (vi.fps ?? "?");
}

async function loadSchema() {
  const schema = await (await fetch(`/projects/${PID}/label_schema`)).json();
  schemaBox.value = JSON.stringify(schema, null, 2);
  return schema;
}

async function saveSchema() {
  let obj = null;
  try { obj = JSON.parse(schemaBox.value); }
  catch (e) { alert("Invalid JSON: " + e); return null; }

  const res = await fetch(`/projects/${PID}/label_schema`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(obj),
  });
  const out = await res.json();
  if (!res.ok) { alert(JSON.stringify(out)); return null; }
  return obj;
}

document.getElementById('btnReloadSchema').onclick = async () => { await refreshAll(); };
document.getElementById('btnSaveSchema').onclick = async () => { const s = await saveSchema(); if (s) await refreshAll(); };

async function refreshEpisodes() {
  const j = await (await fetch(`/projects/${PID}/episodes`)).json();
  const eps = j.episodes || [];

  episodesDump.innerHTML = "";
  eps.forEach(e => {
    const btn = document.createElement("button");
    btn.textContent = `${e.episode_id} [${e.start_idx}, ${e.end_idx}] label=${e.label}`;
    btn.onclick = () => selectEpisode(e);
    episodesDump.appendChild(btn);
    episodesDump.appendChild(document.createElement("br"));
  });

  return eps;
}

function selectEpisode(e) {
  selectedEpisode = e;
  document.getElementById('editEpId').value = e.episode_id;
  document.getElementById('editEpName').value = e.name;
  document.getElementById('editEpLabel').value = e.label;
  document.getElementById('editEpStart').value = e.start_idx;
  document.getElementById('editEpEnd').value = e.end_idx;

  // jump video to episode start (optional but useful)
  vL.currentTime = idxToTime(e.start_idx);
}

document.getElementById('btnUseIn').onclick = () => {
  if (idxIn !== null) document.getElementById('editEpStart').value = idxIn;
};

document.getElementById('btnUseOut').onclick = () => {
  if (idxOut !== null) document.getElementById('editEpEnd').value = idxOut;
};

document.getElementById('btnSaveEp').onclick = async () => {
  if (!selectedEpisode) return alert("No episode selected");

  const body = {
    name: document.getElementById('editEpName').value,
    label: parseInt(document.getElementById('editEpLabel').value),
    start_idx: parseInt(document.getElementById('editEpStart').value),
    end_idx: parseInt(document.getElementById('editEpEnd').value),
  };

  const res = await fetch(
    `/projects/${PID}/episodes/${selectedEpisode.episode_id}`,
    {
      method: "PATCH",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(body),
    }
  );

  const out = await res.json();
  if (!res.ok) { alert(JSON.stringify(out)); return; }

  await refreshTimelineOnly();
};

document.getElementById('btnDeleteEp').onclick = async () => {
  if (!selectedEpisode) return alert("No episode selected");
  if (!confirm("Delete this episode?")) return;

  const res = await fetch(
    `/projects/${PID}/episodes/${selectedEpisode.episode_id}`,
    { method: "DELETE" }
  );

  if (!res.ok) {
    alert("Delete failed");
    return;
  }

  selectedEpisode = null;
  await refreshTimelineOnly();
};


function drawTimeline(eps, schema) {
  resizeCanvas();
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;

  ctx.clearRect(0,0,W,H);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0,0,W,H);

  if (nRows <= 0) return;

  const span = Math.max(1, nRows-1);

  const colorMap = new Map();
  for (const it of (schema.labels || [])) colorMap.set(it.id, it.color || "#4C78A8");
  const unlabeledColor = (schema.unlabeled && schema.unlabeled.color) ? schema.unlabeled.color : "#B0B0B0";

  ctx.strokeStyle = "#999";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, H/2);
  ctx.lineTo(W, H/2);
  ctx.stroke();

  const sorted = [...eps].sort((a,b) => (a.start_idx ?? 0) - (b.start_idx ?? 0));
  for (const e of sorted) {
    const s = clamp(e.start_idx ?? 0, 0, nRows-1);
    const t = clamp(e.end_idx ?? 0, 0, nRows-1);
    const x0 = (s / span) * W;
    const x1 = (t / span) * W;
    const lab = (typeof e.label === "number") ? e.label : -1;
    const col = (lab === -1) ? unlabeledColor : (colorMap.get(lab) || "#222");
    ctx.fillStyle = col;
    ctx.fillRect(x0, 10, Math.max(1, x1-x0), H-20);
  }
}

async function refreshAll() {
  // Preserve playhead across refreshes (important: creating episodes should not jump to 0)
  const prevTime = isFinite(vL.currentTime) ? vL.currentTime : 0;
  const hadSrc = !!vL.getAttribute("src");

  const info = await tryLoadSyncInfo();
  if (!info) {
    fpsPill.textContent = "FPS: -";
    ingestStatus.textContent = "not ingested";
    ingestStatus.classList.remove("ok");
    statsDiv.textContent = "waiting ingest...";
    episodesDump.textContent = "waiting ingest...";
    schemaBox.value = JSON.stringify(await (await fetch('/sample/label_schema.json')).json(), null, 2);
    debugInfo.textContent = "";
    return;
  }
  await loadSyncInfo();
  const schema = await loadSchema();
  const eps = await refreshEpisodes();
  drawTimeline(eps, schema);
  statsDiv.textContent = `Ready. Total frames=${nRows}`;

  const desiredSrc = `/projects/${PID}/media/left`;
  // Only set src when needed; resetting src forces the browser to reload the video and seek to 0.
  if (!hadSrc || vL.getAttribute("src") !== desiredSrc) {
    vL.src = desiredSrc;
    // best effort restore after metadata loads
    vL.addEventListener("loadedmetadata", () => {
      vL.currentTime = clamp(prevTime, 0, (isFinite(vL.duration) && vL.duration>0) ? vL.duration : Math.max(0, (nRows-1)/fps));
    }, { once: true });
  } else {
    // Keep current src; just restore time (in case refresh changed fps/nRows)
    vL.currentTime = clamp(prevTime, 0, (isFinite(vL.duration) && vL.duration>0) ? vL.duration : Math.max(0, (nRows-1)/fps));
  }
}

async function refreshTimelineOnly() {
  // Lightweight refresh: do NOT touch video src/currentTime
  const schema = await loadSchema();
  const eps = await refreshEpisodes();
  drawTimeline(eps, schema);
}


document.getElementById('btnSetIn').onclick = () => { idxIn = timeToIdx(vL.currentTime); inIdxEl.textContent = String(idxIn); };
document.getElementById('btnSetOut').onclick = () => { idxOut = timeToIdx(vL.currentTime); outIdxEl.textContent = String(idxOut); };

document.getElementById('btnCreateEp').onclick = async () => {
  if (idxIn === null || idxOut === null) { alert("Set IN and OUT first."); return; }
  let s = idxIn, e = idxOut;
  if (e < s) { const tmp = s; s = e; e = tmp; }

  const body = {
    name: document.getElementById('epName').value || "episode",
    start_idx: s,
    end_idx: e,
    label: parseInt(document.getElementById('labelId').value || "0"),
  };

  const res = await fetch(`/projects/${PID}/episodes`, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(body),
  });
  const out = await res.json();
  if (!res.ok) { alert(JSON.stringify(out)); return; }
  await refreshTimelineOnly();
};

document.getElementById('btnRefresh').onclick = refreshAll;
document.getElementById('btnCommit').onclick = async () => {
  const res = await fetch(`/projects/${PID}/commit_labels`, {method:"POST"});
  const out = await res.json();
  if (!res.ok) { alert(JSON.stringify(out)); return; }
  alert(`Committed. labeled_frames=${out.labeled_frames}/${out.n_rows}`);
  await refreshAll();
};

document.getElementById('btnExport').onclick = () => { window.location = `/projects/${PID}/export`; };

// Path ingest
document.getElementById('btnIngestPath').onclick = async () => {
  const config_json_path = document.getElementById('pConfigJson').value.trim();

  ingestStatus.textContent = "ingesting...";
  ingestStatus.classList.remove("ok");

  const body = {
    config_json_path: config_json_path
  };

  const res = await fetch(`/projects/${PID}/ingest_path`, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(body),
  });
  const out = await res.json();
  if (!res.ok) { ingestStatus.textContent = "failed"; alert(JSON.stringify(out)); return; }
  ingestStatus.textContent = "ingested";
  ingestStatus.classList.add("ok");
  await refreshAll();
};

canvas.addEventListener('click', (ev) => {
  if (nRows <= 0) return;
  const rect = canvas.getBoundingClientRect();
  const x = ev.clientX - rect.left;
  const span = Math.max(1, nRows-1);
  const idx = clamp(Math.round((x / Math.max(1, rect.width)) * span), 0, nRows-1);
  vL.currentTime = idxToTime(idx);
});

window.addEventListener('resize', refreshAll);
refreshAll();
</script>
</body>
</html>
"""
    return HTMLResponse(tpl.replace("__PID__", project_id))
