"""Phase 2's zero-reference losses - they judge the output against fixed
targets, never against a ground-truth image, which is what lets Phase 2
train on footage that has no bright counterpart.
"""
import torch
import torch.nn.functional as F


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
