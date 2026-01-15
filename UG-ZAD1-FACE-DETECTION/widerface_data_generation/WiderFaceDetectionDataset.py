import torch
from torch.utils.data import Dataset
from torchvision.datasets import WIDERFace
import logging


class WiderFaceDetectionDataset(Dataset):
    def __init__(self, root, split="train", transform=None, base_ds=None):
        if base_ds is not None:
            self.ds = base_ds
        else:
            self.ds = WIDERFace(root=root, split=split, download=True)
        self.transform = transform
        logging.info(f"\nFiltering dataset {split}...")
        self.valid_indices = []
        for i in range(len(self.ds)):
            _, target = self.ds[i]
            if any(self.is_face_valid(target, j) for j in range(len(target['bbox']))):
                self.valid_indices.append(i)
        logging.info(f"Finished. {len(self.valid_indices)} images left.\n")

    def is_face_valid(self, target, idx, min_size=30):
        x, y, w, h = target['bbox'][idx]
        return (target['invalid'][idx] == 0 and
                target['blur'][idx] < 2 and
                w >= min_size and h >= min_size)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        img, target = self.ds[real_idx]

        boxes = []
        labels = []
        for j in range(len(target['bbox'])):
            if self.is_face_valid(target, j):
                x, y, w, h = target['bbox'][j]
                boxes.append([x, y, x + w, y + h])
                labels.append(1)

        target_dict = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64)
        }

        if self.transform:
            img = self.transform(img)

        return img, target_dict