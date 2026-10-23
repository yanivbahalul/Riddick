"""Live night enhancement: camera in, brightened video out, toggled by a key.

Runs the same three stages as pipeline.py, but on a rolling window of the
live feed rather than a captured burst:

    rolling merge  -> brightness match -> enhance

Two things make it fast enough for live, both measured in docs/RESEARCH.md:

  * The merge is incremental. A rolling mean updates in O(1) per frame
    (add the new frame, subtract the one leaving the window) instead of
    re-averaging the whole window, which was 17 ms per frame at 960x540.
  * Curve coefficients are predicted at reduced resolution and upsampled,
    while the curve itself is still applied at full resolution. Measured
    at 960x540: 28 ms -> 8.8 ms, for PSNR 79.8 dB against the full-res
    result and SSIM 1.00000 - no visible cost (section 1, section 9).

Keys:  SPACE toggle enhancement   [ / ]  fewer / more merged frames   q quit

    python scripts/live_enhance.py --checkpoint models_x/model_epoch150.pth
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import subprocess
import time
from collections import deque

import numpy as np
import torch

from riddick.common import TRAIN_MEAN, add_checkpoint_args, build_model, pick_device


class RollingMerge:
    """Mean of the last N frames, updated in O(1) rather than O(N)."""

    def __init__(self, n):
        self.buf = deque(maxlen=n)
        self.total = None

    def push(self, frame):
        if self.total is None:
            self.total = np.zeros_like(frame)
        if len(self.buf) == self.buf.maxlen:
            self.total -= self.buf[0]          # drop the frame leaving the window
        self.buf.append(frame)
        self.total += frame
        return self.total / len(self.buf)

    def resize(self, n):
        keep = list(self.buf)[-n:]
        self.buf = deque(keep, maxlen=n)
        self.total = sum(keep) if keep else None


def open_camera(device, width, height, fps):
    """Raw frames straight off the camera, no intermediate file."""
    cmd = ["ffmpeg", "-loglevel", "error", "-f", "avfoundation",
           "-pixel_format", "uyvy422", "-framerate", str(fps),
           "-video_size", f"{width}x{height}", "-i", str(device),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=width * height * 3 * 4)


def main():
    """Read frames, rolling-merge, enhance, show raw and output side by side."""
    p = argparse.ArgumentParser(description="Live low-light enhancement from a camera")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="0", help="avfoundation device index")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frames", type=int, default=8, help="rolling merge window")
    p.add_argument("--coeff-scale", type=int, default=4,
                   help="predict curve coefficients at 1/N resolution (1 = full)")
    add_checkpoint_args(p)
    p.set_defaults(curvenet=True)   # the live path is Phase 2's CurveNet by default
    args = p.parse_args()

    try:
        import cv2
    except ImportError:
        sys.exit("needs opencv for the preview window: pip install opencv-python")

    dev = pick_device()
    model = build_model(args, dev)

    merger = RollingMerge(args.frames)
    proc = open_camera(args.device, args.width, args.height, args.fps)
    nbytes = args.width * args.height * 3
    enhancing, times = True, deque(maxlen=30)

    print(f"{dev} | SPACE toggle, [ ] window size, q quit")
    try:
        while True:
            raw = proc.stdout.read(nbytes)
            if len(raw) < nbytes:
                break
            frame = np.frombuffer(raw, np.uint8).reshape(args.height, args.width, 3).astype(np.float32) / 255

            t0 = time.time()
            if enhancing:
                merged = merger.push(frame)
                x = torch.from_numpy(merged).permute(2, 0, 1).unsqueeze(0).float().to(dev)
                x = (x * (TRAIN_MEAN / x.mean())).clamp(0, 1)
                with torch.no_grad():
                    out = model(x, coeff_scale=args.coeff_scale).clamp(0, 1)
                shown = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
            else:
                shown = frame
            times.append((time.time() - t0) * 1000)

            side = np.hstack([frame, shown])
            label = (f"{'ON' if enhancing else 'OFF'}  merge={len(merger.buf)}  "
                     f"{np.mean(times):.0f}ms ({1000 / max(np.mean(times), 1e-6):.0f} fps max)")
            bgr = (side[:, :, ::-1] * 255).astype(np.uint8).copy()
            cv2.putText(bgr, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("raw  |  enhanced", bgr)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" "):
                enhancing = not enhancing
            if k == ord("[") and len(merger.buf) > 1:
                merger.resize(max(1, merger.buf.maxlen // 2))
            if k == ord("]"):
                merger.resize(min(64, merger.buf.maxlen * 2))
    finally:
        proc.terminate()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
