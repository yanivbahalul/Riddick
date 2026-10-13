"""Phase 2: self-built unsupervised (zero-reference) model, inspired by
Zero-DCE's ideas - no data/high, no paired ground truth at all.

Step 12: curve-estimation network + the light-enhancement curve formula,
trained with just the Exposure Control loss first, to confirm the
mechanism works before adding the other three losses (Step 13).
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class UnpairedLowLightDataset(Dataset):
    """Just dark photos, no paired bright ground truth - that's the point."""

    def __init__(self, low_dir, transform=None):
        self.low_dir = Path(low_dir)
        self.filenames = sorted(f.name for f in self.low_dir.iterdir())
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.low_dir / self.filenames[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


class CurveNet(nn.Module):
    """Outputs per-pixel curve parameters (not a direct image). Same small
    stem+blocks+final shape as Phase 1's LowLightEnhanceNet, but the final
    layer produces n_iterations*3 channels (one RGB alpha map per curve
    iteration) squashed to [-1, 1] with Tanh instead of an image with Sigmoid."""

    def __init__(self, in_channels=3, hidden_channels=32, n_iterations=8):
        super().__init__()
        self.n_iterations = n_iterations

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, n_iterations * 3, kernel_size=3, padding=1), nn.Tanh(),
        )

    def forward(self, x):
        alphas = self.net(x)  # (B, n_iterations*3, H, W), each value in [-1, 1]
        enhanced = x
        for i in range(self.n_iterations):
            alpha = alphas[:, i * 3:(i + 1) * 3]
            # the light-enhancement curve: push each pixel up (or down) by an
            # amount that depends on how far it already is from 0 and 1, so
            # pixels near the extremes change less than ones in the middle
            enhanced = enhanced + alpha * (enhanced - enhanced ** 2)
        return enhanced.clamp(0, 1)


def exposure_loss(output, patch_size=16, target=0.6):
    # average brightness of each non-overlapping patch should be close to
    # `target` (0.6 = well-exposed, not pitch black, not blown out) - no
    # ground-truth image needed, just the output judged against itself
    gray = output.mean(dim=1, keepdim=True)
    patch_means = F.avg_pool2d(gray, patch_size)
    return (patch_means - target).abs().mean()


def train(args):
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    dataset = UnpairedLowLightDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = CurveNet(hidden_channels=args.hidden_channels, n_iterations=args.iterations).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        running_loss = 0.0
        for low_img in dataloader:
            low_img = low_img.to(device)

            optimizer.zero_grad()
            output = model(low_img)
            loss = exposure_loss(output, args.patch_size, args.exposure_target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{args.epochs}] Loss: {avg_loss:.4f}")

        checkpoint_path = Path(args.checkpoint_dir) / f"model_epoch{epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 Step 12: train the curve-estimation network")
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
    args = parser.parse_args()

    train(args)
