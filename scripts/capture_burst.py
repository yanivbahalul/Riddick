"""Capture a burst of real frames from a connected camera, for merge testing.

    python scripts/capture_burst.py burst/ --seconds 5 --device 0

--device is an index on macOS/Linux and the camera's name on Windows
(see riddick/common.py camera_input_args).

Records to a video file first and extracts frames afterwards. Writing PNGs
directly from the camera makes ffmpeg duplicate a single frame to fill the
count (look for "dup=" in its output) - the burst comes out as N identical
copies, which merges to exactly nothing. Learned that the hard way.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import hashlib
import shutil
import subprocess
import tempfile

from riddick.common import camera_input_args


def main():
    p = argparse.ArgumentParser(description="Record a burst of frames from a camera")
    p.add_argument("outdir")
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--device", default="0", help="camera index (macOS/Linux) or name (Windows)")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    out = Path(args.outdir)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)

    with tempfile.TemporaryDirectory() as tmp:
        video = Path(tmp) / "burst.mkv"
        subprocess.run(["ffmpeg", "-loglevel", "error",
                        *camera_input_args(args.device, args.width, args.height, args.fps),
                        "-t", str(args.seconds), "-c:v", "copy", str(video)], check=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(video), str(out / "f%04d.png")],
                       check=True)

    frames = sorted(out.glob("*.png"))
    unique = len({hashlib.md5(f.read_bytes()).hexdigest() for f in frames})
    print(f"{out}: {len(frames)} frames, {unique} unique "
          f"({args.width}x{args.height} @ {args.fps}fps, {args.seconds}s)")
    if unique < len(frames):
        print("WARNING: duplicate frames - merging them buys nothing")


if __name__ == "__main__":
    main()
