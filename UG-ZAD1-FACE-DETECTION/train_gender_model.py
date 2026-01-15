import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping

from models.LightningModel import LightningModel
from models.GenderCNN import GenderCNN
from datasets.train_gender_data_loaders import train_gender, train_loader_gender, val_loader_gender

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PID %(process)d] [Thread %(threadName)s] %(message)s",
)
torch.set_float32_matmul_precision("high")

gender_pos_weight = train_gender.pos_weight()

gender_model = LightningModel(GenderCNN(), torch.tensor(gender_pos_weight))

trainer = L.Trainer(
    max_epochs=30,
    callbacks=[EarlyStopping(
        monitor="val_loss",
        patience=5,
        mode="min"
    )],
    accelerator="auto",
    devices="auto",
    precision=32,
)

if __name__ == '__main__':
    trainer.fit(gender_model, train_loader_gender, val_loader_gender)
