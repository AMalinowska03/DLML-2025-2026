import os
import sys
import cv2
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from LightningModel import LightningModel
from GenderCNN import GenderCNN
from EyeglassesResNet import EyeglassesResNet
from transformations import cnn_val_tf, resnet_train_tf


male_ckpt = "../lightning_logs/gender_v1/checkpoints/epoch=24-step=31800.ckpt"
glasses_ckpt = "../lightning_logs/glasses_v1/checkpoints/epoch=9-step=12720.ckpt"

male_model = LightningModel.load_from_checkpoint(
    male_ckpt,
    model=GenderCNN(),
    pos_weight=torch.tensor(1.0)
)
male_model.eval().cuda()

glasses_model = LightningModel.load_from_checkpoint(
    glasses_ckpt,
    model=EyeglassesResNet(),
    pos_weight=torch.tensor(1.0)
)
glasses_model.eval().cuda()


def draw_labels(frame, boxes, labels):
    for (x, y, w, h), (male, glasses) in zip(boxes, labels):
        text = f"{'Male' if male else 'Female'}, {'Glasses' if glasses else 'No Glasses'}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def predict_patch(patch):
    male_x = cnn_val_tf(patch).unsqueeze(0).cuda()
    glasses_x = resnet_train_tf(patch).unsqueeze(0).cuda()

    with torch.no_grad():
        male_logit = male_model(male_x).squeeze(1)
        glasses_logit = glasses_model(glasses_x).squeeze(1)
        male_prob = torch.sigmoid(male_logit).item()
        glasses_prob = torch.sigmoid(glasses_logit).item()

    return male_prob > 0.5, glasses_prob > 0.5
