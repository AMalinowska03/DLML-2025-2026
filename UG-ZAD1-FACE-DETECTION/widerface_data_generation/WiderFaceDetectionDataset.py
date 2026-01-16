import torch
from torch.utils.data import Dataset
from torchvision.datasets import WIDERFace
from torchvision import tv_tensors
import logging


class WiderFaceDetectionDataset(Dataset):
    def __init__(self, root, split="train", transform=None, base_ds=None, blur_less=1):
        self.blur_less = blur_less
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

    def is_face_valid(self, target, idx, min_size=30, blur_less=1):
        x, y, w, h = target['bbox'][idx]
        return (target['invalid'][idx] == 0 and
                target['blur'][idx] < blur_less and # TODO: try with 1
                w >= min_size and h >= min_size)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        img, target = self.ds[real_idx]

        width, height = img.size

        boxes = []
        labels = []
        for j in range(len(target['bbox'])):
            if self.is_face_valid(target, j, self.blur_less):
                x, y, w, h = target['bbox'][j]
                boxes.append([x, y, x + w, y + h])
                labels.append(1)

        img = tv_tensors.Image(img)

        # 2. Boxy jako BoundingBoxes z określonym formatem i rozmiarem płótna
        boxes = tv_tensors.BoundingBoxes(
            boxes,
            format="XYXY",
            canvas_size=(height, width),
            dtype=torch.float32
        )
        target_dict = {
            "boxes": boxes,
            "labels": torch.as_tensor(labels, dtype=torch.int64)
        }

        if self.transform:
            img, target_dict = self.transform(img, target_dict)

        return img, target_dict