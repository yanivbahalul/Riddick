"""Training datasets: paired (Phase 1) and unpaired (Phase 2)."""
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from torchvision.transforms import functional as TF

from riddick.common import list_images


def add_sensor_noise(image, full_well=2000.0, read_noise=2.0):
    """Poisson shot noise + Gaussian read noise, the standard sensor model.

    LOL's dark images are studio shots taken with a short exposure, so they
    are far cleaner than a real night frame. Training on them teaches the
    model to brighten but never to suppress noise - and brightening
    multiplies whatever is there. Measured on a real burst: the enhancement
    stage amplifies noise ~2.6x, eating most of what multi-frame merging
    buys (docs/RESEARCH.md 9.9).

    Shot noise scales with the square root of signal (photon counting is
    Poisson), read noise is constant per frame - so dark regions are
    disproportionately noisy, which is exactly the regime that matters here.
    """
    electrons = image.clamp(0, 1) * full_well
    noisy = torch.poisson(electrons) + torch.randn_like(electrons) * read_noise
    return (noisy / full_well).clamp(0, 1)


class LowLightDataset(Dataset):
    """Paired dark/bright images from root_dir/low and root_dir/high.

    A pair shares a filename, so low/x.png and high/x.png are the same scene.
    `transform` (resize + to tensor, see scripts/train.py) runs on both.
    """

    def __init__(self, root_dir, transform=None, augment=False, noise=False,
                 exposure_jitter=None):
        self.low_dir = Path(root_dir) / "low"
        self.high_dir = Path(root_dir) / "high"
        self.filenames = [p.name for p in list_images(self.low_dir)]
        self.transform = transform
        self.augment = augment
        self.noise = noise
        self.exposure_jitter = exposure_jitter  # (min, max) input brightness scale

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """Return one (low, high) pair; idx comes from the DataLoader."""
        filename = self.filenames[idx]
        low_image = Image.open(self.low_dir / filename).convert("RGB")
        high_image = Image.open(self.high_dir / filename).convert("RGB")

        if self.transform:
            low_image = self.transform(low_image)
            high_image = self.transform(high_image)

        if self.augment:
            low_image, high_image = self._augment_pair(low_image, high_image)

        if self.exposure_jitter:
            # Randomly rescale the input's brightness, leaving the target alone.
            # The model currently only behaves correctly on inputs near the
            # training set's 0.05 mean - feed it a real night photo at 0.21 and
            # it overshoots ~50% and blows out highlights (RESEARCH 9.4). That
            # is patched at inference with --match-input-brightness; this makes
            # it robust by construction instead, by showing the same scene at
            # many exposures and always asking for the same well-exposed result.
            lo, hi = self.exposure_jitter
            low_image = (low_image * random.uniform(lo, hi)).clamp(0, 1)

        if self.noise:
            # only the input gets noise - the target stays clean, so the model
            # is being asked to remove it rather than reproduce it.
            # After the jitter, so a brightened input carries proportionally
            # less noise, which is what a real brighter exposure would.
            low_image = add_sensor_noise(low_image)

        return low_image, high_image

    def _augment_pair(self, low_image, high_image):
        # low/high must get the exact same random flip/crop, or they stop being a matching pair.
        if random.random() < 0.5:
            low_image = TF.hflip(low_image)
            high_image = TF.hflip(high_image)

        size = low_image.shape[-2:]
        pad = 16
        low_image = TF.pad(low_image, pad, padding_mode="reflect")
        high_image = TF.pad(high_image, pad, padding_mode="reflect")
        i, j, h, w = RandomCrop.get_params(low_image, output_size=size)
        low_image = TF.crop(low_image, i, j, h, w)
        high_image = TF.crop(high_image, i, j, h, w)

        return low_image, high_image


class UnpairedLowLightDataset(Dataset):
    """Just dark photos, no paired bright ground truth - that's the point."""

    def __init__(self, low_dir, transform=None, exposure_jitter=None):
        self.paths = list_images(low_dir)
        self.transform = transform
        self.exposure_jitter = exposure_jitter

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.exposure_jitter:
            # Same idea as the supervised side (Step 28), but it transfers
            # BADLY here - and for the exact reason it was expected to work.
            #
            # Supervised training has a per-scene target, so jitter teaches
            # "whatever brightness comes in, produce this particular image".
            # Phase 2's losses are zero-reference: they only ask that patches
            # average `exposure_target`, with no notion of what the scene
            # should look like. Jitter on top of that teaches a brightness
            # normaliser and little else.
            #
            # Measured (Step 29, seed-matched): PSNR 18.21 -> 12.37 for only
            # a 12% drift reduction, against Phase 1's -2.34 PSNR for -42%.
            # Kept available, not recommended here. See RESEARCH 9.16.
            lo, hi = self.exposure_jitter
            image = (image * random.uniform(lo, hi)).clamp(0, 1)
        return image
