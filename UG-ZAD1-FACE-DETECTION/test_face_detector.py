




import lightning as L
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
from widerface_data_generation.WiderFaceDetectionDataset import WiderFaceDetectionDataset
from models.FaceDetector import FaceDetectorLightning

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [PID %(process)d] [Thread %(threadName)s] %(message)s",
)

face_detector_ckpt_v1 = "lightning_logs/face_detector_v1/checkpoints/epoch=8-step=11448.ckpt" # TODO: set when generated
def collate_fn(batch):
    return tuple(zip(*batch))

test_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
])

if __name__ == '__main__':
    test_ds = WiderFaceDetectionDataset(root="data", split="val", transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=4, collate_fn=collate_fn, num_workers=2)

    model = FaceDetectorLightning.load_from_checkpoint(face_detector_ckpt_v1)
    if torch.cuda.is_available():
        model.eval().cuda()
    elif torch.xpu.is_available():
        model.eval().xpu()
    else:
        model.eval()
    trainer = L.Trainer()
    trainer.test(model, test_loader)