import os

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class WIDERFaceAttr(Dataset):
    def __init__(self, csv_file, folder, transform=None):
        self.df = pd.read_csv(csv_file)
        self.folder = folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.folder, row['filename'])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        eyeglasses = torch.tensor(row['eyeglasses'], dtype=torch.float)
        male = torch.tensor(row['male'], dtype=torch.float)
        return img, eyeglasses, male
