"""Score a checkpoint against a folder of low/high ground-truth pairs.

    python scripts/evaluate.py --checkpoint models_x/model_epoch150.pth
    python scripts/evaluate.py --checkpoint models_p2/...pth --curvenet --low-dir data/lolv2_test_low --high-dir data/lolv2_test_high

Always run at native resolution. Resizing to 256px once invented a 0.05
SSIM gain that was not real (docs/RESEARCH.md 9.14), and every number in
the log assumes this protocol.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from riddick.common import add_checkpoint_args, build_model, pick_device
from riddick.metrics import psnr, ssim


def evaluate(args):
    """Run the model on every low/high pair, print per-image and mean scores."""
    device = pick_device()
    model = build_model(args, device)
    to_tensor = transforms.ToTensor()

    low_dir, high_dir = Path(args.low_dir), Path(args.high_dir)
    filenames = sorted(f.name for f in low_dir.iterdir())
    scores = []

    with torch.no_grad():
        for name in filenames:
            low = to_tensor(Image.open(low_dir / name).convert("RGB")).unsqueeze(0).to(device)
            high = to_tensor(Image.open(high_dir / name).convert("RGB")).unsqueeze(0).to(device)

            out = model(low).clamp(0, 1)
            p, s = psnr(out, high), ssim(out, high)
            scores.append((p, s))
            print(f"{name}: PSNR={p:.2f}  SSIM={s:.4f}")

    n = len(scores)
    print(f"\nAverage over {n} images: "
          f"PSNR={sum(p for p, _ in scores) / n:.2f}  "
          f"SSIM={sum(s for _, s in scores) / n:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Score a checkpoint against ground-truth pairs")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--low-dir", type=str, default="data/eval15_low")
    p.add_argument("--high-dir", type=str, default="data/eval15_high")
    add_checkpoint_args(p)
    evaluate(p.parse_args())
