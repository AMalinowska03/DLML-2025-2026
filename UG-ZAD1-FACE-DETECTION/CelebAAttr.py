import torch
from torchvision.datasets import CelebA


class CelebAAttr(torch.utils.data.Dataset):
    def __init__(self, split, attr, transform):
        self.ds = CelebA(root="data", split=split, target_type="attr", download=True)
        self.attr_idx = self.ds.attr_names.index(attr)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, attrs = self.ds[i]
        y = 1.0 if attrs[self.attr_idx] == 1 else 0.0
        img = self.transform(img)
        return img, torch.tensor(y)

    def pos_weight(self):
        attrs = self.ds.attr
        labels = (attrs[:, self.attr_idx] == 1).float()
        pos = (labels == 1).sum().item()
        neg = (labels == 0).sum().item()
        return neg / pos
