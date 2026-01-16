import torch
import lightning as L
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.v2 as transforms
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from widerface_data_generation.WiderFaceDetectionDataset import WiderFaceDetectionDataset
from models.FaceDetector import FaceDetectorLightning

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PID %(process)d] [Thread %(threadName)s] %(message)s",
)

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def collate_fn(batch):
    return tuple(zip(*batch))


def prepare_data():
    full_train_ds = WiderFaceDetectionDataset(root="data", split="train", transform=train_transform)

    train_size = int(0.8 * len(full_train_ds))
    val_size = len(full_train_ds) - train_size

    train_ds, val_ds = random_split(full_train_ds, [train_size, val_size])

    # train_ds = WiderFaceDetectionDataset(root="data", split="train", transform=train_transform, base_ds=train_subset)
    # val_ds = WiderFaceDetectionDataset(root="data", split="train", transform=train_transform, base_ds=val_subset)

    return train_ds, val_ds


if __name__ == '__main__':
    train_ds, val_ds = prepare_data()

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=4, collate_fn=collate_fn, num_workers=2)

    model = FaceDetectorLightning()

    trainer = L.Trainer(
        max_epochs=20,
        accelerator="auto",
        devices=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=3),
            ModelCheckpoint(monitor="val_mAP_50", mode="max", filename="best-face-detector")
        ]
    )

    # Start treningu
    trainer.fit(model, train_loader, val_loader)