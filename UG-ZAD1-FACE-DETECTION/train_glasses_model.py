import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping

from models.LightningModel import LightningModel
from models.EyeglassesResNet import EyeglassesResNet
from datasets.train_glasses_data_loaders import train_glass, train_loader_glass, val_loader_glass

torch.set_float32_matmul_precision("high")

glass_pos_weight = train_glass.pos_weight()

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

if __name__ == '__main__':
    trainer.fit(glasses_model, train_loader_glass, val_loader_glass)
