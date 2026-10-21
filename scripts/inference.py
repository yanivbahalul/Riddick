"""Run a trained checkpoint on an image or a folder of them.

    python scripts/inference.py --checkpoint models_x/...pth --input photo.jpg -o outputs
    python scripts/inference.py --checkpoint ...pth --input data/real_night --match-input-brightness 0.0496
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from riddick.common import TRAIN_MEAN, add_checkpoint_args, build_model, pick_device


def enhance(args):
    """Enhance each input image and write it to the output directory."""
    device = pick_device()
    model = build_model(args, device)
    to_tensor, to_pil = transforms.ToTensor(), transforms.ToPILImage()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    src = Path(args.input)
    paths = [src] if src.is_file() else sorted(src.glob("*"))

    with torch.no_grad():
        for path in paths:
            x = to_tensor(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

            if args.match_input_brightness:
                # Models are calibrated to the brightness they trained on
                # (~0.05). Real phone night photos average ~0.21 and barely
                # overlap that range, so the model applies the curve it learned
                # for near-black input and overshoots ~50%, blowing out
                # highlights. Scaling fixes it with no retraining (RESEARCH 9.4).
                x = (x * (args.match_input_brightness / x.mean())).clamp(0, 1)

            out = model(x).squeeze(0).cpu().clamp(0, 1)
            dest = Path(args.output_dir) / path.name
            to_pil(out).save(dest)
            print(f"Saved {dest}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run low-light enhancement on images")
    p.add_argument("--input", type=str, required=True, help="Image file or directory")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("-o", "--output-dir", type=str, default="outputs")
    p.add_argument("--match-input-brightness", type=float, default=None, metavar="MEAN",
                   help=f"Rescale input to this mean first; use {TRAIN_MEAN} for real photos")
    add_checkpoint_args(p)
    enhance(p.parse_args())
