import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping

from LightningModel import LightningModel
from GenderCNN import GenderCNN
from EyeglassesResNet import EyeglassesResNet

from data_loaders import (train_gender, train_glass, \
    train_loader_gender, val_loader_gender, test_loader_gender, \
    train_loader_glass, val_loader_glass, test_loader_glass)

torch.set_float32_matmul_precision("high")

if __name__ == "__main__":
    gender_pos_weight = train_gender.pos_weight()
    glass_pos_weight = train_glass.pos_weight()

    gender_model = LightningModel(GenderCNN(), torch.tensor(gender_pos_weight))
    glasses_model = LightningModel(EyeglassesResNet(), torch.tensor(glass_pos_weight))

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

    trainer.fit(gender_model, train_loader_gender, val_loader_gender)
    trainer.test(gender_model, test_loader_gender)

    trainer.fit(glasses_model, train_loader_glass, val_loader_glass)
    trainer.test(glasses_model, test_loader_glass)
