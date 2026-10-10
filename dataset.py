import os
import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from torchvision.transforms import functional as TF


class LowLightDataset(Dataset):
    def __init__(self, root_dir, transform=None, augment=False):
        self.low_dir = Path(root_dir) / "low"
        self.high_dir = Path(root_dir) / "high"
        self.filenames = sorted(os.listdir(self.low_dir))
        self.transform = transform
        self.augment = augment

    # Dataset is part of torch that help Pytorch to work
    # lines 10-12 is loaction of the dataset for the train, and sorted for a-b-c (creating the actual dataset)
    # transform is a function we can se in "tarin.py" file.

    def __len__(self):
        return len(self.filenames)
    
    # counted how much images have been taken

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        low_image = Image.open(self.low_dir / filename).convert("RGB")
        high_image = Image.open(self.high_dir / filename).convert("RGB")

        if self.transform:
            low_image = self.transform(low_image)
            high_image = self.transform(high_image)

        if self.augment:
            low_image, high_image = self._augment_pair(low_image, high_image)

        return low_image, high_image

    #idx is a number that taken from dataloader in train.py file.
    #apply transform (resize, convert to tensor).

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

