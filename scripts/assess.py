"""Score a checkpoint on every criterion that matters, not just the benchmark.

Step 28 came out worse on `eval15` and better on real footage, which made
it clear that one number cannot pick a checkpoint any more (RESEARCH 9.15).
This reports all three axes side by side so the trade is visible instead
of being rediscovered each time.

  benchmark   PSNR/SSIM on eval15, at native resolution. Guards against
              regression - but eval15 is clean, in-distribution, and has
              had ~20 tuning decisions selected against it, so a high
              score here is necessary and nowhere near sufficient.

  robustness  How much the output drifts when the same scenes are fed at
              0.5x to 4x input brightness. A calibrated enhancer should
              land near the same place; one that just applies gain tracks
              its input. Lower is better.

  real        Behaviour on real night photos fed raw, with no
              normalisation patch: output brightness and the share of
              blown-out pixels. This is closest to the actual goal.

    python scripts/assess.py models_x [models_y ...] [--curvenet] [--hidden-channels 64]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from riddick.common import add_checkpoint_args, build_model  # noqa: E402
from riddick.metrics import psnr, ssim  # noqa: E402

# eval15 must be loaded at native resolution: every benchmark number in
# docs/ comes from evaluate.py, which does not resize, and mixing the two
# protocols invents improvements that are not there - a 256px resize once
# showed 0.8367 SSIM where the truth was 0.7847 (RESEARCH 9.14).
TO_TENSOR = transforms.ToTensor()
# real_night photos are large and varied in size; they are only used for
# brightness and blown-out share, both scale-invariant, so resizing is safe
# and keeps the run fast.
RESIZE = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])


def load_dir(d, resize=False):
    """Load every image in a directory as a 1xCxHxW tensor."""
    t = RESIZE if resize else TO_TENSOR
    return [t(Image.open(f"{d}/{n}").convert("RGB")).unsqueeze(0)
            for n in sorted(os.listdir(d))]


def assess(model, low, high, night):
    """Return (psnr, ssim, drift, brightness, blown%) - the three axes."""
    with torch.no_grad():
        # benchmark: eval15 is evaluated at its own exposure, untouched
        p = [psnr(model(x).clamp(0, 1), y) for x, y in zip(low, high)]
        s = [ssim(model(x).clamp(0, 1), y) for x, y in zip(low, high)]

        # robustness: same scenes, deliberately wrong exposures
        drift = [np.mean([model((x * k).clamp(0, 1)).clamp(0, 1).mean().item() for x in low])
                 for k in (0.5, 1.0, 2.0, 4.0)]

        # real: no normalisation patch, so the model is on its own
        outs = [model(x).clamp(0, 1) for x in night]
        bright = np.mean([o.mean().item() for o in outs])
        blown = np.mean([(o > 0.98).float().mean().item() * 100 for o in outs])

    return np.mean(p), np.mean(s), max(drift) - min(drift), bright, blown


def main():
    """Score each given checkpoint and print one row per checkpoint."""
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+", help="checkpoint dirs or .pth files")
    ap.add_argument("--epoch", type=int, default=150, help="which epoch to read from a dir")
    add_checkpoint_args(ap)
    ap.set_defaults(hidden_channels=64)
    args = ap.parse_args()

    low, high = load_dir("data/eval15_low"), load_dir("data/eval15_high")
    night = load_dir("data/real_night", resize=True)

    print(f"{'checkpoint':>26} | {'PSNR':>6} {'SSIM':>7} | {'drift':>7} | {'bright':>7} {'blown':>7}")
    print(f"{'':>26} | {'benchmark':>14} | {'robust':>7} | {'real footage':>15}")
    print("-" * 80)
    for ck in args.checkpoints:
        path = ck if ck.endswith(".pth") else f"{ck}/model_epoch{args.epoch}.pth"
        if not os.path.exists(path):
            print(f"{os.path.basename(ck):>26} | missing: {path}")
            continue
        model = build_model(args, torch.device("cpu"), checkpoint=path)
        p, s, d, b, bl = assess(model, low, high, night)
        print(f"{os.path.basename(ck):>26} | {p:6.2f} {s:7.4f} | {d:7.4f} | {b:7.4f} {bl:6.2f}%")

    print("\nbenchmark: higher better, but clean + in-distribution + heavily tuned against")
    print("robust:    LOWER better - output spread across an 8x input-exposure range")
    print("real:      blown-out share, LOWER better - raw input, no normalisation patch")


if __name__ == "__main__":
    main()
