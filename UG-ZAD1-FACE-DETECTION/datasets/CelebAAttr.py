import torch
from torchvision.datasets import CelebA


class CelebAAttr(torch.utils.data.Dataset):
    def __init__(self, split, attr, transform):
        self.split = split
        self.attr = attr
        self.transform = transform

        ds = CelebA(root="data", split=split, target_type="attr", download=True)
        self.attr_idx = ds.attr_names.index(attr)
        self.length = len(ds)

    def _get_ds(self):
        if not hasattr(self, "_ds"):
            self._ds = CelebA(root="data", split=self.split, target_type="attr", download=False)
        return self._ds

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        ds = self._get_ds()
        img, attrs = ds[i]

        y = 1.0 if attrs[self.attr_idx] == 1 else 0.0

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(y, dtype=torch.float32)

    def pos_weight(self):
        ds = CelebA(root="data", split=self.split, target_type="attr", download=False)
        labels = (ds.attr[:, self.attr_idx] == 1).float()
        pos = (labels == 1).sum().item()
        neg = (labels == 0).sum().item()
        return neg / pos
