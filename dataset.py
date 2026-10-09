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

        return low_image, high_image

    #idx is a number that taken from dataloader in train.py file.
    #apply transform (resize, convert to tensor).

