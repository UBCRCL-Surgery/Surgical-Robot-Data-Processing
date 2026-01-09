"""
Simple Gradio video loader / preview app.

Run:
  python app_vid.py

Then open the local URL printed in the terminal and upload a video to preview it.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Optional, Tuple


def _extract_video_path(video_value: Any) -> Optional[str]:
    """
    Gradio `Video` components may pass either:
      - a string filepath, or
      - a dict like {"path": "...", ...} depending on Gradio version/mode.
    """
    if video_value is None:
        return None
    if isinstance(video_value, str):
        return video_value
    if isinstance(video_value, dict):
        p = video_value.get("path")
        return str(p) if p else None
    return None


_CACHE_DIR = Path(__file__).resolve().parent / ".gradio_video_cache"


def _describe_file(path: str) -> str:
    try:
        p = Path(path)
        size = p.stat().st_size if p.exists() else None
        size_mb = (size / (1024 * 1024)) if size is not None else None
        msg = f"Loaded: {p.name}"
        if size_mb is not None:
            msg += f" ({size_mb:.2f} MB)"
        return msg
    except Exception as e:
        return f"Loaded video, but failed to read file info: {e}"


def _transcode_to_h264_mp4(input_path: str) -> Tuple[Optional[str], str]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None, "ffmpeg not found on PATH. Install ffmpeg or disable transcoding."

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _CACHE_DIR / f"{uuid.uuid4().hex}.mp4"

    cmd = [
        "ffmpeg.exe",
        "-y",
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "128k",
        "-movflags",
        "+faststart",
        str(out_path),
    ]

    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        if p.returncode != 0:
            err = (p.stderr or "").strip()
            tail = err[-1500:] if err else "(no stderr)"
            return None, f"ffmpeg failed (code {p.returncode}). stderr tail:\n{tail}"
        return str(out_path), f"Transcoded to H.264 MP4: {out_path.name}"
    except Exception as e:
        return None, f"Failed to run ffmpeg: {e}"


def prepare_preview(
    video_value: Any, transcode_h264: bool
) -> Tuple[Optional[str], str]:
    """
    Returns:
      - a value accepted by gr.Video (best-effort across Gradio versions)
      - info text
    """
    path = _extract_video_path(video_value)
    if not path:
        return None, "No video selected."

    info = _describe_file(path)

    # Pass-through (may be unplayable in browser if codec is HEVC/H.265, etc.)
    if not transcode_h264:
        return None, info

    out_path, t_info = _transcode_to_h264_mp4(path)
    if not out_path:
        return (
            path,
            info + "\n\n" + t_info + "\nShowing original upload (may be unplayable).",
        )

    return (
        out_path,
        info + "\n\n" + t_info + "\nShowing original upload (may be unplayable).",
    )


def main() -> None:
    # Windows: Gradio/Uvicorn/AnyIO can occasionally surface noisy
    # ProactorEventLoop connection-reset tracebacks when the browser disconnects.
    # Using the selector event loop policy is a common mitigation.
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())  # type: ignore[attr-defined]
        except Exception:
            pass

    try:
        import gradio as gr
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Gradio is not installed. Install it with:\n"
            "  pip install gradio\n"
            f"\nOriginal import error: {e}"
        )

    with gr.Blocks(title="VideoLoader") as demo:
        gr.Markdown("### Video Loader\nUpload a video and preview it.")

        with gr.Row():
            inp = gr.File(label="Upload video", file_types=["video"])
            with gr.Column():
                transcode_h264 = gr.Checkbox(
                    value=True,
                    label="Transcode to browser-friendly H.264 MP4 (requires ffmpeg)",
                )
                info = gr.Textbox(label="Info", interactive=False)
        preview = gr.Video(label="Preview", interactive=False)

        inp.change(
            fn=prepare_preview, inputs=[inp, transcode_h264], outputs=[preview, info]
        )
        transcode_h264.change(
            fn=prepare_preview, inputs=[inp, transcode_h264], outputs=[preview, info]
        )

    # Let Gradio pick a free port; bind to localhost by default.
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))


if __name__ == "__main__":
    main()
