import os

import pandas as pd
import torch
from tensorboard.compat.tensorflow_stub.errors import InvalidArgumentError
from torch.utils.data import Dataset
from PIL import Image

class WIDERFaceAttr(Dataset):
    def __init__(self, csv_file, folder, attr, transform=None):
        self.df = pd.read_csv(csv_file)
        self.folder = folder
        self.attr = attr
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.folder, row['filename'])).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if self.attr == "Eyeglasses":
            y = torch.tensor(row['eyeglasses'], dtype=torch.float)
        elif self.attr == "Male":
            y = torch.tensor(row['male'], dtype=torch.float)
        else:
            raise InvalidArgumentError

        return img, y
