import torch
from LightningModel import LightningModel
from EyeglassesResNet import EyeglassesResNet
from GenderCNN import GenderCNN
from lightning import Trainer
from data_loaders import wider_male_loader, wider_glasses_loader

gender_checkpoint = "lightning_logs/gender_v1/checkpoints/epoch=24-step=31800.ckpt"
glasses_checkpoint = "lightning_logs/glasses_v1/checkpoints/epoch=9-step=12720.ckpt"

gender_model = LightningModel.load_from_checkpoint(
    gender_checkpoint,
    model=GenderCNN(),
    pos_weight=torch.tensor(1.0)
)

gender_model.eval().cuda()

glasses_model = LightningModel.load_from_checkpoint(
    glasses_checkpoint,
    model=EyeglassesResNet(),
    pos_weight=torch.tensor(1.0)
)

glasses_model.eval().cuda()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.test(gender_model, wider_male_loader)
    trainer.test(glasses_model, wider_glasses_loader)