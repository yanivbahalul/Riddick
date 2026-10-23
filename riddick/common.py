"""Shared bits used by every script that loads a trained checkpoint.

These lived as copy-paste in evaluate/inference/pipeline/assess.
Changing a default meant finding four places, and they had already
drifted apart.
"""
import sys
from pathlib import Path

import torch

from riddick.models import CurveNet, LowLightEnhanceNet, UNet

# Mean brightness of data/low. The models only behave correctly on input
# near this value - fed a brighter real photo they overshoot ~50% and blow
# out highlights (docs/RESEARCH.md 9.4).
TRAIN_MEAN = 0.0496

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
              ".dng", ".arw", ".nef", ".cr2"}


def list_images(directory):
    """Image files in a directory, sorted by name.

    Skips everything else, so a stray .DS_Store or subfolder does not crash
    a script halfway through a run.
    """
    return sorted(p for p in Path(directory).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def pick_device():
    """Apple Silicon GPU, else NVIDIA/AMD, else CPU. AMD ROCm uses the cuda API."""
    return torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )


def add_checkpoint_args(parser):
    """Flags describing how a checkpoint was trained, so it can be rebuilt.

    A checkpoint is only weights - loading it needs the same architecture
    it was trained with, hence these must match the training command.
    """
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=8, help="CurveNet: number of curve applications")
    parser.add_argument("--curvenet", action="store_true", help="Phase 2 CurveNet checkpoint")
    parser.add_argument("--unet", action="store_true", help="U-Net checkpoint")
    parser.add_argument("--extra-layer", action="store_true", help="Trained with a 4th Conv2d+ReLU")
    parser.add_argument("--batchnorm", action="store_true", help="Trained with BatchNorm2d")
    parser.add_argument("--skip-connection", action="store_true", help="Trained with a skip connection")
    return parser


def build_model(args, device, checkpoint=None):
    """Rebuild the architecture `args` describes, load weights, set eval mode."""
    if args.curvenet:
        model = CurveNet(hidden_channels=args.hidden_channels, n_iterations=args.iterations)
    elif args.unet:
        model = UNet(hidden_channels=args.hidden_channels)
    else:
        model = LowLightEnhanceNet(
            hidden_channels=args.hidden_channels, extra_layer=args.extra_layer,
            batchnorm=args.batchnorm, skip_connection=args.skip_connection,
        )
    model.load_state_dict(torch.load(checkpoint or args.checkpoint, map_location=device))
    return model.to(device).eval()


def camera_input_args(device, width, height, fps):
    """ffmpeg flags that open a live camera - each OS names cameras differently.

    device: an index on macOS ("0") and Linux ("0" or "/dev/video0"); the
    camera's name on Windows (list them with
    `ffmpeg -list_devices true -f dshow -i dummy`).
    """
    size = ["-framerate", str(fps), "-video_size", f"{width}x{height}"]
    if sys.platform == "darwin":
        return ["-f", "avfoundation", "-pixel_format", "uyvy422", *size, "-i", str(device)]
    if sys.platform == "win32":
        return ["-f", "dshow", *size, "-i", f"video={device}"]
    dev = str(device) if str(device).startswith("/dev/") else f"/dev/video{device}"
    return ["-f", "v4l2", *size, "-i", dev]
