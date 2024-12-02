import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class Dataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        labels = self.dataframe.iloc[idx, -4:].to_numpy(dtype=np.float32)

        if self.transform:
            image = self.transform(image)

        return image, labels