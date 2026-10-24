"""Train the input denoiser with no clean images (Neighbor2Neighbor).

    python scripts/train_denoiser.py --data-dir data/low --out models_denoiser/denoiser.pth
    python scripts/evaluate.py --curvenet --checkpoint models_p2/model_epoch150.pth \
        --denoiser models_denoiser/denoiser.pth

Every 2x2 cell of a noisy image gives two neighbouring pixels: they see
(almost) the same scene but carry independent noise. Split the image into
two half-size sub-images that way and each can be the training target for
the other - the network cannot predict the noise in the target, so the best
it can do is predict the clean signal. No ground truth is ever used, which
means it can train on real night footage directly.

loss = |D(g1) - g2|^2 + gamma * |D(g1) - g2 - (g1(D(y)) - g2(D(y)))|^2
The second term corrects for the two pixels not being exactly the same
scene point (D(y) is computed without gradient).

Smoke-test first: --steps 20 --crop 64
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from riddick.common import list_images, pick_device
from riddick.models import Denoiser

# the 8 ordered pairs of edge-sharing positions in a 2x2 cell,
# numbered 0=(0,0) 1=(0,1) 2=(1,0) 3=(1,1)
NEIGHBOUR_PAIRS = torch.tensor([[0, 1], [1, 0], [0, 2], [2, 0], [1, 3], [3, 1], [2, 3], [3, 2]])


def cells(y):
    """(B, C, H, W) -> (B, C, 4, H/2, W/2): the four pixels of every 2x2 cell."""
    B, C, H, W = y.shape
    return F.unfold(y, 2, stride=2).view(B, C, 4, H // 2, W // 2)


def pick(y, idx):
    """Take one pixel per cell, at the positions in idx."""
    return cells(y).gather(2, idx).squeeze(2)


def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = pick_device()

    images = [torch.from_numpy(np.asarray(Image.open(f).convert("RGB"), np.float32) / 255).permute(2, 0, 1)
              for f in list_images(args.data_dir)]
    model = Denoiser().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)

    for step in range(args.steps):
        batch = []
        for _ in range(args.batch_size):
            img = random.choice(images)
            r = random.randrange(img.shape[1] - args.crop)
            c = random.randrange(img.shape[2] - args.crop)
            patch = img[:, r:r + args.crop, c:c + args.crop]
            batch.append(patch.flip(2) if random.random() < 0.5 else patch)
        y = torch.stack(batch).to(device)

        # a random neighbour pair for every cell, same choice across channels
        B, C, H, W = y.shape
        pair = NEIGHBOUR_PAIRS[torch.randint(0, 8, (B, 1, H // 2, W // 2))].to(device)
        i1 = pair[..., 0].unsqueeze(2).expand(B, C, 1, H // 2, W // 2)
        i2 = pair[..., 1].unsqueeze(2).expand(B, C, 1, H // 2, W // 2)
        g1, g2 = pick(y, i1), pick(y, i2)

        with torch.no_grad():
            full = model(y)
        out = model(g1)
        loss = ((out - g2) ** 2).mean() + args.gamma * ((out - g2 - (pick(full, i1) - pick(full, i2))) ** 2).mean()

        optimizer.zero_grad()
        # dark-image residuals are ~1e-4 in MSE; scale so Adam's early steps are not lost in eps
        (loss * 100).backward()
        optimizer.step()
        schedule.step()
        if step % 250 == 0 or step == args.steps - 1:
            print(f"Step [{step + 1}/{args.steps}] Loss: {loss.item():.6f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train the self-supervised input denoiser")
    p.add_argument("--data-dir", type=str, default="data/low", help="Folder of noisy dark images (no clean pairs needed)")
    p.add_argument("--out", type=str, default="models_denoiser/denoiser.pth")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--crop", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=2.0, help="Weight of the regulariser term")
    p.add_argument("--seed", type=int, default=1)
    train(p.parse_args())
