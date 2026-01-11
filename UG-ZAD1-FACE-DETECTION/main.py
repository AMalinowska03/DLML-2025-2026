import torch
from LightningModel import LightningModel
from EyeglassesResNet import EyeglassesResNet
from GenderCNN import GenderCNN
from lightning import Trainer
from data_loaders import wider_male_loader

ckpt = "lightning_logs/gender_v1/checkpoints/epoch=24-step=31800.ckpt"

model = LightningModel.load_from_checkpoint(
    ckpt,
    model=GenderCNN(),
    pos_weight=torch.tensor(1.0)
)

model.eval().cuda()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.test(model, wider_male_loader)