import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import LowLightDataset
from metrics import ssim


class LowLightEnhanceNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, hidden_channels=32, extra_layer=False, batchnorm=False, skip_connection=False):
        super().__init__()
        self.skip_connection = skip_connection

        def hidden_block():
            block = [nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)]
            if batchnorm:
                block.append(nn.BatchNorm2d(hidden_channels))
            block.append(nn.ReLU(inplace=True))
            return block

        stem = [nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)]
        if batchnorm:
            stem.append(nn.BatchNorm2d(hidden_channels))
        stem.append(nn.ReLU(inplace=True))

        blocks = hidden_block()
        if extra_layer:
            blocks += hidden_block()

        final_in_channels = hidden_channels * 2 if skip_connection else hidden_channels
        final = [nn.Conv2d(final_in_channels, out_channels, kernel_size=3, padding=1), nn.Sigmoid()]

        if skip_connection:
            # Keep stem/blocks/final as separate pieces so forward() can torch.cat the
            # stem's output back in before the final conv (Step 9) - the final conv then
            # sees both the raw early features and the processed ones, side by side.
            self.stem = nn.Sequential(*stem)
            self.blocks = nn.Sequential(*blocks)
            self.final = nn.Sequential(*final)
        else:
            # Unchanged from Steps 0-8: one flat pipeline, so every existing checkpoint
            # still loads (same net.0, net.1, ... keys).
            self.net = nn.Sequential(*stem, *blocks, *final)

        # input: dark image, 3 matrices (R, G, B)
        # stem: 1st Conv2d, 32 filters -> 32 feature maps (+ optional BatchNorm) -> ReLU
        # blocks: one (or two, extra_layer=True) more Conv2d+ReLU, 32->32
        # skip_connection=True: concat the stem's output with the blocks' output
        #   (32+32=64 channels) before the final conv, instead of only using blocks' output
        # final: last Conv2d back to 3 channels (R, G, B), Sigmoid squashes to valid pixel range

    def forward(self, x):
        if self.skip_connection:
            stem_out = self.stem(x)
            features = torch.cat([stem_out, self.blocks(stem_out)], dim=1)
            return self.final(features)
        return self.net(x)


class UNet(nn.Module):
    """Step 10: a real U-Net - 2 downsample/upsample stages, skip connections
    at each resolution level, not just one (Step 9 showed one skip connection
    does nothing on a shallow network; this gives it actual depth to matter)."""

    def __init__(self, in_channels=3, out_channels=3, hidden_channels=32):
        super().__init__()
        C = hidden_channels

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            )

        # Encoder: each stage halves resolution, doubles channels
        self.enc0 = conv_block(in_channels, C)       # full res,  C channels  -> skip0
        self.enc1 = conv_block(C, C * 2)              # 1/2 res,   2C channels -> skip1
        self.bottleneck = conv_block(C * 2, C * 4)    # 1/4 res,   4C channels
        self.pool = nn.MaxPool2d(2)

        # Decoder: each stage upsamples back, concats the matching skip, then convs down
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = conv_block(C * 4 + C * 2, C * 2)  # upsampled(4C) + skip1(2C) -> 2C
        self.up0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec0 = conv_block(C * 2 + C, C)          # upsampled(2C) + skip0(C) -> C

        self.final = nn.Sequential(nn.Conv2d(C, out_channels, kernel_size=3, padding=1), nn.Sigmoid())

    def forward(self, x):
        skip0 = self.enc0(x)
        skip1 = self.enc1(self.pool(skip0))
        x = self.bottleneck(self.pool(skip1))

        x = self.dec1(torch.cat([self.up1(x), skip1], dim=1))
        x = self.dec0(torch.cat([self.up0(x), skip0], dim=1))
        return self.final(x)


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

    dataset = LowLightDataset(args.data_dir, transform=transform, augment=args.augment)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    if args.unet:
        model = UNet(hidden_channels=args.hidden_channels).to(device)
    else:
        model = LowLightEnhanceNet(
            hidden_channels=args.hidden_channels, extra_layer=args.extra_layer, batchnorm=args.batchnorm,
            skip_connection=args.skip_connection,
        ).to(device)

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
    parser.add_argument("--hidden-channels", type=int, default=32)
    parser.add_argument("--extra-layer", action="store_true", help="Add a 4th Conv2d+ReLU layer")
    parser.add_argument("--batchnorm", action="store_true", help="Add BatchNorm2d after each hidden Conv2d")
    parser.add_argument("--loss", type=str, default="l1", choices=["l1", "mse", "ssim"], help="Loss function")
    parser.add_argument("--augment", action="store_true", help="Random horizontal flip + crop during training (Step 7)")
    parser.add_argument("--skip-connection", action="store_true", help="Concat stem output into the final conv (Step 9)")
    parser.add_argument("--unet", action="store_true", help="Use the full U-Net architecture instead of LowLightEnhanceNet (Step 10)")
    args = parser.parse_args()

    train(args)
