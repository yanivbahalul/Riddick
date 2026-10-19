"""End-to-end test of the night pipeline in near-total darkness.

Simulated rather than captured, and that is the stronger test here: a real
burst has no ground truth, so "did it damage the image?" can only be
eyeballed. Here the clean scene is known exactly, so brightness, noise,
detail and colour can all be scored against it.

The scene itself is real - the 22-frame merge of an actual camera burst,
which measured ~12 effective bits (docs/RESEARCH.md 9.9), so it is a
genuine photograph rather than synthetic content. It is then taken down to
near-darkness in *linear* light and re-photographed through a
Poisson-Gaussian sensor model at that much lower photon count, which is
what actually makes a dark frame noisy.

    python scripts/dark_scene_test.py --frames ~/Desktop/burst
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from merge import merge_frames  # noqa: E402
from metrics import psnr, ssim  # noqa: E402
from train_unsupervised import CurveNet  # noqa: E402

GAMMA = 2.2
FULL_WELL = 2000.0
READ_NOISE = 2.0
TRAIN_MEAN = 0.0496


def capture_in_the_dark(clean_srgb, darkness, rng):
    """Re-photograph a scene at `darkness` of its original light level."""
    linear = clean_srgb ** GAMMA * darkness
    electrons = linear * FULL_WELL
    noisy = rng.poisson(electrons) + rng.normal(0, READ_NOISE, electrons.shape)
    return np.clip(np.clip(noisy / FULL_WELL, 0, 1) ** (1 / GAMMA), 0, 1)


def to_t(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).unsqueeze(0).float()


def enhance(model, a, normalize=True):
    x = to_t(a)
    if normalize:
        x = (x * (TRAIN_MEAN / x.mean())).clamp(0, 1)
    with torch.no_grad():
        return model(x).clamp(0, 1)


def flat_noise(a):
    """std on the flattest 8x8 block - dominated by noise, not structure."""
    g = a.mean(2) if a.ndim == 3 else a
    h, w = g.shape
    bs = 64
    blocks = [(g[y:y + bs, x:x + bs].std(), y, x)
              for y in range(0, h - bs, bs) for x in range(0, w - bs, bs)]
    return float(min(blocks)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=os.path.expanduser("~/Desktop/burst"))
    ap.add_argument("--checkpoint", default="models_seedcheck_s1/model_epoch150.pth")
    ap.add_argument("--darkness", type=float, default=0.06,
                    help="fraction of original light kept; 0.06 is roughly 4 stops down")
    ap.add_argument("--num-frames", type=int, default=16)
    ap.add_argument("--out", default="docs/comparisons/dark_scene_test.png")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    paths = sorted(os.listdir(args.frames))[:22]
    src = [np.asarray(Image.open(f"{args.frames}/{p}").convert("RGB"), np.float64) / 255 for p in paths]
    truth, _ = merge_frames(src, align=False)          # clean reference scene
    truth = truth[300:1000, 600:1300]                  # crop: no faces, no window

    dark = [capture_in_the_dark(truth, args.darkness, rng) for _ in range(args.num_frames)]
    model = CurveNet(hidden_channels=32, n_iterations=8)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    merged, _ = merge_frames(dark, align=False)
    single_out = enhance(model, dark[0])
    merged_out = enhance(model, merged)

    t = to_t(truth)
    rows = [
        ("original scene (reference)", to_t(truth), None),
        (f"in the dark ({args.darkness:.0%} light)", to_t(dark[0]), None),
        ("1 frame -> enhanced", single_out, flat_noise(single_out[0].permute(1, 2, 0).numpy())),
        (f"{args.num_frames} frames -> enhanced", merged_out, flat_noise(merged_out[0].permute(1, 2, 0).numpy())),
    ]

    print(f"Scene taken down to {args.darkness:.0%} of its light "
          f"(mean {truth.mean():.4f} -> {dark[0].mean():.4f}), then re-photographed "
          f"{args.num_frames}x through a sensor model.\n")
    print("PSNR/SSIM are computed after scaling the output to the reference's")
    print("brightness. The enhancer is *meant* to end up brighter than the dim")
    print("original, so scoring it against the reference directly would measure")
    print("that intended exposure difference instead of how much of the scene")
    print("actually survived.\n")
    print(f"{'':34s} {'bright':>8} {'PSNR':>7} {'SSIM':>7} {'flat noise':>11}")
    print("-" * 72)
    for label, img, noise in rows:
        if noise is None:
            print(f"{label:34s} {img.mean():8.4f} {'-':>7} {'-':>7} {'-':>11}")
            continue
        matched = (img * (t.mean() / img.mean())).clamp(0, 1)
        print(f"{label:34s} {img.mean():8.4f} {psnr(matched, t):7.2f} "
              f"{ssim(matched, t):7.4f} {noise:11.5f}")

    W = H = 330
    gap, lbl = 8, 18
    canvas = Image.new("RGB", (len(rows) * W + (len(rows) - 1) * gap, H + lbl), "white")
    d = ImageDraw.Draw(canvas)
    for i, (label, img, _) in enumerate(rows):
        arr = (img[0].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        canvas.paste(Image.fromarray(arr).resize((W, H)), (i * (W + gap), lbl))
        d.text((i * (W + gap) + 3, 4), label, fill="black")
    canvas.save(args.out)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
