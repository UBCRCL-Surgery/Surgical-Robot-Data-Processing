#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import subprocess
from pathlib import Path
import argparse


def extract_clip_index(p: Path) -> int:
    """
    eyeVideo_xxx.avi      -> 1
    eyeVideo_xxx.av_2.avi -> 2
    eyeVideo_xxx.av_3.avi -> 3
    """
    m = re.search(r"_([0-9]+)\.avi$", p.name)
    if m:
        return int(m.group(1))
    return 1


def merge_gaze_videos(
    video_dir: Path,
    prefix: str,
    out_video: Path,
):
    videos = sorted(
        video_dir.glob(f"{prefix}*.avi"),
        key=extract_clip_index,
    )

    if not videos:
        raise RuntimeError("No gaze videos found.")

    print("Found gaze clips:")
    for v in videos:
        print(f"  {v.name}")

    # ffmpeg concat list
    concat_list = video_dir / "gaze_concat_list.txt"
    with open(concat_list, "w") as f:
        for v in videos:
            f.write(f"file '{v.resolve()}'\n")

    cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(out_video),
    ]

    print("\nRunning ffmpeg:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    print(f"\nMerged gaze video saved to:\n  {out_video}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="Directory containing gaze videos")
    ap.add_argument("--prefix", required=True, help="eyeVideo prefix (without index)")
    ap.add_argument("--out", default="gaze_merged.avi", help="Output merged video")
    args = ap.parse_args()

    video_dir = Path(args.dir)
    out_video = video_dir / args.out

    merge_gaze_videos(
        video_dir=video_dir,
        prefix=args.prefix,
        out_video=out_video,
    )


if __name__ == "__main__":
    main()
