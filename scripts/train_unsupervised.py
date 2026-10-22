"""Phase 2: self-built unsupervised (zero-reference) model, inspired by
Zero-DCE's ideas - no data/high, no paired ground truth at all.

Step 12 trained the curve-estimation network with just the exposure
control loss, to confirm the mechanism works; Step 13 then added the
other three losses one at a time (each is off until its weight is > 0).

The current best recipe (what scripts/seed_check.sh trains):

    python scripts/train_unsupervised.py --epochs 150 --spatial-weight 1.0 \
        --color-weight 1.0 --tv-weight 40.0 --exposure-target 0.45 \
        --checkpoint-dir models_p2
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from riddick.common import pick_device
from riddick.dataset import UnpairedLowLightDataset
from riddick.losses import (color_constancy_loss, exposure_loss,
                            illumination_smoothness_loss, spatial_consistency_loss)
from riddick.models import CurveNet


def train(args):
    if args.seed is not None:
        # makes a run reproducible, and lets the same config be repeated
        # with different seeds to measure how much of a result is just
        # random-init noise rather than a real effect
        torch.manual_seed(args.seed)

    device = pick_device()

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    dataset = UnpairedLowLightDataset(args.data_dir, transform=transform,
                                  exposure_jitter=tuple(args.exposure_jitter) if args.exposure_jitter else None)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = CurveNet(hidden_channels=args.hidden_channels, n_iterations=args.iterations).to(device)
    if args.init_checkpoint:
        # fine-tune from an existing checkpoint instead of random init - the
        # actual test of Phase 2's claimed advantage (adapting to real
        # footage with no paired data), as opposed to training from scratch
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))
        print(f"Fine-tuning from {args.init_checkpoint}")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        running_loss = 0.0
        running_components = {"exposure": 0.0, "spatial": 0.0, "color": 0.0, "smoothness": 0.0}
        for low_img in dataloader:
            low_img = low_img.to(device)

            optimizer.zero_grad()
            output, alphas = model(low_img, return_alphas=True)

            exp_l = exposure_loss(output, args.patch_size, args.exposure_target, args.linear_exposure)
            loss = args.exposure_weight * exp_l
            running_components["exposure"] += exp_l.item()

            if args.spatial_weight > 0:
                spa_l = spatial_consistency_loss(output, low_img, args.spatial_patch_size)
                loss = loss + args.spatial_weight * spa_l
                running_components["spatial"] += spa_l.item()
            if args.color_weight > 0:
                col_l = color_constancy_loss(output)
                loss = loss + args.color_weight * col_l
                running_components["color"] += col_l.item()
            if args.tv_weight > 0:
                tv_l = illumination_smoothness_loss(alphas)
                loss = loss + args.tv_weight * tv_l
                running_components["smoothness"] += tv_l.item()

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        parts = " ".join(
            f"{name}={val / len(dataloader):.4f}"
            for name, val in running_components.items()
            if val != 0.0
        )
        print(f"Epoch [{epoch + 1}/{args.epochs}] Loss: {avg_loss:.4f} ({parts})")

        checkpoint_path = Path(args.checkpoint_dir) / f"model_epoch{epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: train the zero-reference curve-estimation network")
    parser.add_argument("--data-dir", type=str, default="data/low", help="Folder of dark photos (no paired high needed)")
    parser.add_argument("--checkpoint-dir", type=str, default="models_unsupervised", help="Where to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=8, help="Number of curve applications (Zero-DCE uses 8)")
    parser.add_argument("--patch-size", type=int, default=16, help="Local patch size for the exposure control loss")
    parser.add_argument("--exposure-target", type=float, default=0.6, help="Target average brightness per patch")
    parser.add_argument("--exposure-weight", type=float, default=1.0, help="Weight for the exposure control loss")
    parser.add_argument("--spatial-weight", type=float, default=0.0, help="Step 13a: weight for spatial consistency loss (0 = off)")
    parser.add_argument("--spatial-patch-size", type=int, default=4, help="Local patch size for the spatial consistency loss")
    parser.add_argument("--color-weight", type=float, default=0.0, help="Step 13b: weight for color constancy loss (0 = off)")
    parser.add_argument("--tv-weight", type=float, default=0.0, help="Step 13c: weight for illumination smoothness loss (0 = off)")
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Fine-tune from this existing checkpoint instead of training from scratch")
    parser.add_argument("--exposure-jitter", type=float, nargs=2, metavar=("MIN","MAX"),
                        help="Randomly scale input brightness so the model stops depending on one input exposure (Step 28, ported from the supervised side)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed; set it to make a run reproducible or to repeat a config across seeds")
    parser.add_argument("--linear-exposure", action="store_true", help="Measure patch brightness in linear light instead of gamma-encoded sRGB (removes a contrast-dependent bias)")
    args = parser.parse_args()

    train(args)
