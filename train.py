import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import LowLightDataset


class LowLightEnhanceNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, hidden_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        # input: dark image, 3 matrices (R, G, B)
        # 1st Conv2d: 32 filters, each does multiply+sum over the image -> 32 new feature map
        # 2nd Conv2d: 32 filters, multiply+sum over the previous 32 maps -> 32 new maps 
        # zero out negatives again
        # 3rd Conv2d: 3 filters, multiply+sum over the 32 maps -> back to 3 channels (R, G, B)
        # squash every value into range 0-1 (valid pixel range) -> final output image

    def forward(self, x):
        return self.net(x)
    
    # forward pass: push x through self.net, return output

def train(args):
    # checks Apple Silicon GPU (mps) -> NVIDIA/AMD GPU (cuda - AMD ROCm builds use the same cuda API) -> cpu fallback
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    dataset = LowLightDataset(args.data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = LowLightEnhanceNet().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        running_loss = 0.0
        for low_img, high_img in dataloader:
            low_img, high_img = low_img.to(device), high_img.to(device)

            optimizer.zero_grad()
            output = model(low_img)
            loss = criterion(output, high_img)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{args.epochs}] Loss: {avg_loss:.4f}")

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
    args = parser.parse_args()

    train(args)
