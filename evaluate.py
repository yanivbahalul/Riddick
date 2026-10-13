import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from metrics import psnr, ssim
from train import LowLightEnhanceNet, UNet
from train_unsupervised import CurveNet


def evaluate(args):
    # checks Apple Silicon GPU (mps) -> NVIDIA/AMD GPU (cuda - AMD ROCm builds use the same cuda API) -> cpu fallback
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    if args.curvenet:
        model = CurveNet(hidden_channels=args.hidden_channels, n_iterations=args.iterations).to(device)
    elif args.unet:
        model = UNet(hidden_channels=args.hidden_channels).to(device)
    else:
        model = LowLightEnhanceNet(
            hidden_channels=args.hidden_channels, extra_layer=args.extra_layer, batchnorm=args.batchnorm,
            skip_connection=args.skip_connection,
        ).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    to_tensor = transforms.ToTensor()

    low_dir = Path(args.low_dir)
    high_dir = Path(args.high_dir)
    filenames = sorted(f.name for f in low_dir.iterdir())

    psnr_scores = []
    ssim_scores = []

    with torch.no_grad():
        for filename in filenames:
            low_img = to_tensor(Image.open(low_dir / filename).convert("RGB")).unsqueeze(0).to(device)
            high_img = to_tensor(Image.open(high_dir / filename).convert("RGB")).unsqueeze(0).to(device)

            output = model(low_img).clamp(0, 1)

            p = psnr(output, high_img)
            s = ssim(output, high_img)
            psnr_scores.append(p)
            ssim_scores.append(s)
            print(f"{filename}: PSNR={p:.2f}  SSIM={s:.4f}")

    avg_psnr = sum(psnr_scores) / len(psnr_scores)
    avg_ssim = sum(ssim_scores) / len(ssim_scores)
    print(f"\nAverage over {len(filenames)} images: PSNR={avg_psnr:.2f}  SSIM={avg_ssim:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint against ground-truth images")
    parser.add_argument("--low-dir", type=str, default="data/eval15_low")
    parser.add_argument("--high-dir", type=str, default="data/eval15_high")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--extra-layer", action="store_true", help="Match a checkpoint trained with a 4th Conv2d+ReLU layer")
    parser.add_argument("--batchnorm", action="store_true", help="Match a checkpoint trained with BatchNorm2d")
    parser.add_argument("--skip-connection", action="store_true", help="Match a checkpoint trained with a skip connection")
    parser.add_argument("--unet", action="store_true", help="Match a checkpoint trained with the U-Net architecture (Step 10)")
    parser.add_argument("--curvenet", action="store_true", help="Match a checkpoint trained with the Phase 2 CurveNet (Step 12)")
    parser.add_argument("--iterations", type=int, default=8, help="Match the --iterations used to train a CurveNet checkpoint")
    args = parser.parse_args()

    evaluate(args)
