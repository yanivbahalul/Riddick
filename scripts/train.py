"""Phase 1: train a supervised model on paired dark/bright images.

    python scripts/train.py --checkpoint-dir models_x --epochs 150 --loss ssim --augment

Smoke-test a change first with tiny settings (--epochs 2 --batch-size 8
--image-size 64). Every epoch writes model_epochN.pth to --checkpoint-dir.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from riddick.common import pick_device
from riddick.dataset import LowLightDataset
from riddick.metrics import ssim
from riddick.models import LowLightEnhanceNet, UNet


def train(args):
    device = pick_device()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    dataset = LowLightDataset(args.data_dir, transform=transform, augment=args.augment, noise=args.noise,
                              exposure_jitter=tuple(args.exposure_jitter) if args.exposure_jitter else None)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if args.unet:
        model = UNet(hidden_channels=args.hidden_channels).to(device)
    else:
        model = LowLightEnhanceNet(
            hidden_channels=args.hidden_channels, extra_layer=args.extra_layer, batchnorm=args.batchnorm,
            skip_connection=args.skip_connection, global_context=args.global_context,
        ).to(device)
    if args.init_checkpoint:
        # resume from an existing checkpoint instead of random init - e.g.
        # continuing a run that got cut off by a disconnected Colab session
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))
        print(f"Resuming from {args.init_checkpoint}")

    l1 = nn.L1Loss()
    mse = nn.MSELoss()

    def compute_loss(output, target):
        if args.loss == "mse":
            return mse(output, target)
        if args.loss == "ssim":
            # L1 keeps overall brightness/color accurate, (1 - SSIM) pushes for structural similarity
            return l1(output, target) + (1 - ssim(output, target, reduce=False))
        return l1(output, target)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # cosine-anneals lr -> 0 over the full run, instead of the flat lr used
    # in every run so far - lets training take large steps early and fine
    # settle late, without introducing a new hyperparameter to tune.
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.lr_schedule else None
    )

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        running_loss = 0.0
        for low_img, high_img in dataloader:
            low_img, high_img = low_img.to(device), high_img.to(device)

            optimizer.zero_grad()
            output = model(low_img)
            loss = compute_loss(output, high_img)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if scheduler:
            scheduler.step()

        avg_loss = running_loss / len(dataloader)
        lr_part = f" LR: {optimizer.param_groups[0]['lr']:.6f}" if scheduler else ""
        print(f"Epoch [{epoch + 1}/{args.epochs}] Loss: {avg_loss:.4f}{lr_part}")

        checkpoint_path = Path(args.checkpoint_dir) / f"model_epoch{epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train low-light enhancement model")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to dataset root (expects low/ and high/ subfolders)")
    parser.add_argument("--checkpoint-dir", type=str, default="models", help="Where to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--extra-layer", action="store_true", help="Add a 4th Conv2d+ReLU layer")
    parser.add_argument("--batchnorm", action="store_true", help="Add BatchNorm2d after each hidden Conv2d")
    parser.add_argument("--loss", type=str, default="l1", choices=["l1", "mse", "ssim"], help="Loss function")
    parser.add_argument("--augment", action="store_true", help="Random horizontal flip + crop during training (Step 7)")
    parser.add_argument("--exposure-jitter", type=float, nargs=2, metavar=("MIN","MAX"),
                        help="Randomly scale input brightness in this range so the model stops depending on one input exposure (Step 28)")
    parser.add_argument("--noise", action="store_true", help="Add Poisson-Gaussian sensor noise to inputs so the model learns to suppress it (Step 25)")
    parser.add_argument("--skip-connection", action="store_true", help="Concat stem output into the final conv (Step 9)")
    parser.add_argument("--unet", action="store_true", help="Use the full U-Net architecture instead of LowLightEnhanceNet (Step 10)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed, to make a run reproducible")
    parser.add_argument("--global-context", action="store_true", help="Add an image-wide feature average to every pixel")
    parser.add_argument("--lr-schedule", action="store_true", help="Cosine-anneal the learning rate from --lr down to 0 over training")
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Resume from this existing checkpoint instead of training from scratch")
    args = parser.parse_args()

    train(args)
