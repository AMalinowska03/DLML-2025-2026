import torch
from models.LightningModel import LightningModel
from models.EyeglassesResNet import EyeglassesResNet
from lightning import Trainer
from datasets.test_glasses_data_loaders import test_loader_glass, wider_glasses_loader

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PID %(process)d] [Thread %(threadName)s] %(message)s",
)
torch.set_float32_matmul_precision("high")

glasses_checkpoint_v1 = "lightning_logs/glasses_v1/checkpoints/epoch=9-step=12720.ckpt"
glasses_checkpoint_v2 = "lightning_logs/glasses_v2/checkpoints/epoch=8-step=11448.ckpt"

glasses_model = LightningModel.load_from_checkpoint(
    glasses_checkpoint_v2,
    model=EyeglassesResNet(),
    pos_weight=torch.tensor(1.0)
)

if torch.cuda.is_available():
    glasses_model.eval().cuda()
elif torch.xpu.is_available():
    glasses_model.eval().xpu()
else:
    glasses_model.eval()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.test(glasses_model, test_loader_glass)
    trainer.test(glasses_model, wider_glasses_loader)