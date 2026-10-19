import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from train import LowLightEnhanceNet, UNet
from train_unsupervised import CurveNet


def enhance(args):
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
    to_pil = transforms.ToPILImage()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    image_paths = [input_path] if input_path.is_file() else sorted(input_path.glob("*"))

    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            tensor = to_tensor(image).unsqueeze(0).to(device)

            if args.match_input_brightness:
                # The model's behaviour is calibrated to the brightness it was
                # trained on. LOL's "low" images average ~0.05; real phone night
                # photos average ~0.21 (already auto-exposed by the phone's ISP),
                # and the two ranges barely overlap. Fed a real photo directly,
                # the model applies the curve it learned for near-black input and
                # overshoots by ~50%, blowing out highlights. Scaling the input
                # into the trained range removes that with no retraining.
                # See RESEARCH.md 9.4.
                tensor = (tensor * (args.match_input_brightness / tensor.mean())).clamp(0, 1)

            output = model(tensor).squeeze(0).cpu().clamp(0, 1)
            enhanced = to_pil(output)

            out_path = Path(args.output_dir) / image_path.name
            enhanced.save(out_path)
            print(f"Saved enhanced image to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run low-light enhancement inference")
    parser.add_argument("--input", type=str, required=True, help="Path to an image or a directory of images")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to a trained model checkpoint (.pth)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Where to save enhanced images")
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--extra-layer", action="store_true", help="Match a checkpoint trained with a 4th Conv2d+ReLU layer")
    parser.add_argument("--batchnorm", action="store_true", help="Match a checkpoint trained with BatchNorm2d")
    parser.add_argument("--skip-connection", action="store_true", help="Match a checkpoint trained with a skip connection")
    parser.add_argument("--unet", action="store_true", help="Match a checkpoint trained with the U-Net architecture (Step 10)")
    parser.add_argument("--curvenet", action="store_true", help="Match a checkpoint trained with the Phase 2 CurveNet (Step 12)")
    parser.add_argument("--iterations", type=int, default=8, help="Match the --iterations used to train a CurveNet checkpoint")
    parser.add_argument("--match-input-brightness", type=float, default=None, metavar="MEAN",
                        help="Scale input to this mean brightness before the model (use 0.0496, the training-set mean, for real photos)")
    args = parser.parse_args()

    enhance(args)
