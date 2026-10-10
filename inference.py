import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from train import LowLightEnhanceNet


def enhance(args):
    # checks Apple Silicon GPU (mps) -> NVIDIA/AMD GPU (cuda - AMD ROCm builds use the same cuda API) -> cpu fallback
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = LowLightEnhanceNet(hidden_channels=args.hidden_channels).to(device)
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
    args = parser.parse_args()

    enhance(args)
