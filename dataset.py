import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class LowLightDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.low_dir = Path(root_dir) / "low"
        self.high_dir = Path(root_dir) / "high"
        self.filenames = sorted(os.listdir(self.low_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        low_image = Image.open(self.low_dir / filename).convert("RGB")
        high_image = Image.open(self.high_dir / filename).convert("RGB")

        if self.transform:
            low_image = self.transform(low_image)
            high_image = self.transform(high_image)

        return low_image, high_image
