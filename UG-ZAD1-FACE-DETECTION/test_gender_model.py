import torch
from models.LightningModel import LightningModel
from models.GenderCNN import GenderCNN
from lightning import Trainer
from datasets.test_gender_data_loaders import test_loader_gender, wider_male_loader

torch.set_float32_matmul_precision("high")

gender_checkpoint = "lightning_logs/gender_v1/checkpoints/epoch=24-step=31800.ckpt"

gender_model = LightningModel.load_from_checkpoint(
    gender_checkpoint,
    model=GenderCNN(),
    pos_weight=torch.tensor(1.0)
)

gender_model.eval()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.test(gender_model, test_loader_gender)
    trainer.test(gender_model, wider_male_loader)