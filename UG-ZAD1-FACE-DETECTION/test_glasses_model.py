import torch
from models.LightningModel import LightningModel
from models.EyeglassesResNet import EyeglassesResNet
from lightning import Trainer
from datasets.test_glasses_data_loaders import test_loader_glass, wider_glasses_loader

torch.set_float32_matmul_precision("high")

glasses_checkpoint = "lightning_logs/glasses_v1/checkpoints/epoch=9-step=12720.ckpt"

glasses_model = LightningModel.load_from_checkpoint(
    glasses_checkpoint,
    model=EyeglassesResNet(),
    pos_weight=torch.tensor(1.0)
)

glasses_model.eval().cuda()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.test(glasses_model, test_loader_glass)
    trainer.test(glasses_model, wider_glasses_loader)