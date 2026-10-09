import torch
import torch.nn.functional as F


def psnr(img1, img2, max_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * torch.log10(max_val ** 2 / mse).item()


def ssim(img1, img2, window_size=11, max_val=1.0):
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    pad = window_size // 2

    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=pad)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=pad)

    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=pad) - mu1 ** 2
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=pad) - mu2 ** 2
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=pad) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()
