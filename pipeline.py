"""The full night pipeline: capture -> merge -> normalize -> enhance.

Each stage exists because a measurement said it was needed (docs/RESEARCH.md):

  merge      multi-frame averaging is the only stage that adds real signal
             rather than inferring it. Measured ~2.5x noise reduction on a
             real 22-frame burst (section 9.8).
  normalize  the model is calibrated to the brightness it trained on
             (~0.05); feed it a real night frame (~0.21) and it overshoots
             by ~50% and blows out highlights (section 9.4).
  enhance    the learned model, which only now sees input in the regime it
             was actually trained for.

Usage:
    python pipeline.py --frames burst/ --checkpoint models_x/model_epoch150.pth --curvenet -o out.png
    python pipeline.py --video clip.mov --num-frames 16 --checkpoint ... --curvenet -o out.png
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import TRAIN_MEAN, add_checkpoint_args, build_model, pick_device
from merge import frames_from_video, load, merge_frames


def run(args):
    """Merge the burst, rescale into the trained range, enhance, save."""
    device = pick_device()

    paths = (frames_from_video(args.video, args.num_frames) if args.video
             else sorted(Path(args.frames).iterdir())[:args.num_frames])
    frames = load(paths)

    merged, shifts = merge_frames(frames, align=not args.no_align)
    moved = sum(1 for s in shifts if s != (0, 0))
    print(f"merge     {len(frames)} frames, {moved} realigned, brightness {merged.mean():.4f}")

    x = torch.from_numpy(merged).permute(2, 0, 1).unsqueeze(0).float().to(device)

    if not args.no_normalize:
        x = (x * (args.train_mean / x.mean())).clamp(0, 1)
        print(f"normalize scaled to {args.train_mean} (the trained regime)")

    with torch.no_grad():
        out = build_model(args, device)(x).clamp(0, 1)
    print(f"enhance   output brightness {out.mean():.4f}, "
          f"blown-out {(out > 0.98).float().mean() * 100:.2f}%")

    arr = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(args.output)
    print(f"saved     {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Merge a burst, then enhance it")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--frames", type=str, help="Directory of frames")
    src.add_argument("--video", type=str, help="Video file")
    p.add_argument("--num-frames", type=int, default=16)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("-o", "--output", type=str, required=True)
    p.add_argument("--train-mean", type=float, default=TRAIN_MEAN)
    p.add_argument("--no-align", action="store_true", help="Skip alignment (locked-off camera)")
    p.add_argument("--no-normalize", action="store_true", help="Skip brightness matching (to see what it was fixing)")
    add_checkpoint_args(p)
    run(p.parse_args())
