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

    def forward(self, x, return_alphas=False, coeff_scale=1):
        """coeff_scale > 1 predicts the curve coefficients at 1/scale resolution
        and upsamples them, instead of running the convolutions at full size.

        This is the standard real-time trick (bilateral grid / LUT methods hit
        4K in ~13ms with it - docs/RESEARCH.md section 1). It works because the
        coefficients describe *illumination*, which varies smoothly across a
        scene, so they do not need per-pixel resolution - while the curve
        itself is still applied at full resolution, so output detail is
        unaffected. Cost drops roughly with the square of the scale.

        Training always runs at scale 1; this is an inference-time option.
        """
        if coeff_scale > 1:
            small = F.interpolate(x, scale_factor=1 / coeff_scale, mode="bilinear",
                                  align_corners=False, recompute_scale_factor=False)
            alphas = F.interpolate(self.net(small), size=x.shape[-2:],
                                   mode="bilinear", align_corners=False)
        else:
            alphas = self.net(x)  # (B, n_iterations*3, H, W), each value in [-1, 1]

        enhanced = x
        for i in range(self.n_iterations):
            alpha = alphas[:, i * 3:(i + 1) * 3]
            # the light-enhancement curve: push each pixel up (or down) by an
            # amount that depends on how far it already is from 0 and 1, so
            # pixels near the extremes change less than ones in the middle
            enhanced = enhanced + alpha * (enhanced - enhanced ** 2)
        enhanced = enhanced.clamp(0, 1)
        if return_alphas:
            # only Step 13's illumination smoothness loss needs the raw
            # per-pixel curve parameters - evaluate.py/inference.py never
            # pass this, so they keep getting just the enhanced image back
            return enhanced, alphas
        return enhanced


GAMMA = 2.2  # sRGB is gamma-encoded; file values are not proportional to light


def exposure_loss(output, patch_size=16, target=0.6, linear=False):
    # average brightness of each non-overlapping patch should be close to
    # `target` (0.6 = well-exposed, not pitch black, not blown out) - no
    # ground-truth image needed, just the output judged against itself
    gray = output.mean(dim=1, keepdim=True)

    if linear:
        # Averaging gamma-encoded values is a biased estimate of average light.
        # s = L^(1/GAMMA) and x^(1/2.2) is concave, so by Jensen's inequality
        #     mean(s) <= (mean L)^(1/GAMMA)
        # i.e. the sRGB patch mean *understates* how much light the patch
        # actually has, and the error grows with contrast inside the patch.
        # Measured on this project's data: the bias is -0.0028 on LOL's dark
        # images but -0.0162 on real night photos (~5.8x worse), and within
        # real photos it is -0.0018 on flat patches vs -0.0395 on
        # high-contrast ones. So the loss reads a bright-light-next-to-shadow
        # patch as darker than it is and over-brightens it - which matches the
        # blown-out-to-white failure seen on real photos in Step 16.
        # Averaging in linear light removes the bias by construction.
        linear_mean = F.avg_pool2d(gray.clamp(1e-6, 1.0) ** GAMMA, patch_size)
        patch_means = linear_mean.clamp(1e-6, 1.0) ** (1.0 / GAMMA)
    else:
        patch_means = F.avg_pool2d(gray, patch_size)

    return (patch_means - target).abs().mean()


# Step 13's three remaining losses, each added on top of exposure_loss one
# at a time (13a: +spatial, 13b: +color, 13c: +smoothness), same
# one-change-at-a-time method as Phase 1.

def spatial_consistency_loss(output, original, patch_size=4):
    # keeps the enhanced image's local contrast pattern close to the
    # original's: average both into patch_size x patch_size patches, then
    # compare each patch to its 4 neighbors (left/right/up/down) before vs.
    # after enhancement - large mismatches mean the curve distorted structure
    # that brightness alone shouldn't have touched.
    orig_gray = original.mean(dim=1, keepdim=True)
    out_gray = output.mean(dim=1, keepdim=True)
    orig_patches = F.avg_pool2d(orig_gray, patch_size)
    out_patches = F.avg_pool2d(out_gray, patch_size)

    # 4 fixed 3x3 kernels, each computing "this patch minus one neighbor"
    # for every patch at once via convolution.
    neighbor_kernels = torch.tensor([
        [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],   # left
        [[0, 0, 0], [0, 1, -1], [0, 0, 0]],   # right
        [[0, -1, 0], [0, 1, 0], [0, 0, 0]],   # up
        [[0, 0, 0], [0, 1, 0], [0, -1, 0]],   # down
    ], dtype=torch.float32, device=output.device).unsqueeze(1)

    orig_diffs = F.conv2d(orig_patches, neighbor_kernels, padding=1)
    out_diffs = F.conv2d(out_patches, neighbor_kernels, padding=1)
    return ((orig_diffs - out_diffs) ** 2).mean()


def color_constancy_loss(output):
    # Gray-World assumption: a well-balanced photo's average R, G, and B
    # should be roughly equal - penalize the three channel averages for
    # drifting apart, so exposure correction doesn't also tint the image.
    mean_rgb = output.mean(dim=[2, 3])  # (B, 3)
    r, g, b = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
    return ((r - g) ** 2 + (r - b) ** 2 + (g - b) ** 2).mean()


def illumination_smoothness_loss(alphas):
    # penalizes sharp changes between neighboring pixels in the raw curve
    # parameter maps (before the curve formula is applied) - real lighting
    # changes gradually across a scene, so a noisy/blocky adjustment map is
    # a sign of overfitting to individual pixels rather than lighting.
    dh = (alphas[:, :, 1:, :] - alphas[:, :, :-1, :]).abs().mean()
    dw = (alphas[:, :, :, 1:] - alphas[:, :, :, :-1]).abs().mean()
    return dh + dw


def train(args):
    if args.seed is not None:
        # makes a run reproducible, and lets the same config be repeated
        # with different seeds to measure how much of a result is just
        # random-init noise rather than a real effect
        torch.manual_seed(args.seed)

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
    parser.add_argument("--exposure-weight", type=float, default=1.0, help="Weight for the exposure control loss")
    parser.add_argument("--spatial-weight", type=float, default=0.0, help="Step 13a: weight for spatial consistency loss (0 = off)")
    parser.add_argument("--spatial-patch-size", type=int, default=4, help="Local patch size for the spatial consistency loss")
    parser.add_argument("--color-weight", type=float, default=0.0, help="Step 13b: weight for color constancy loss (0 = off)")
    parser.add_argument("--tv-weight", type=float, default=0.0, help="Step 13c: weight for illumination smoothness loss (0 = off)")
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Fine-tune from this existing checkpoint instead of training from scratch")
    parser.add_argument("--seed", type=int, default=None, help="Random seed; set it to make a run reproducible or to repeat a config across seeds")
    parser.add_argument("--linear-exposure", action="store_true", help="Measure patch brightness in linear light instead of gamma-encoded sRGB (removes a contrast-dependent bias)")
    args = parser.parse_args()

    train(args)
