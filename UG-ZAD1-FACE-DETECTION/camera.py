import cv2
from PIL import Image

from skimage.feature import Cascade

import torch

from models.LightningModel import LightningModel
from models.GenderCNN import GenderCNN
from models.EyeglassesResNet import EyeglassesResNet
from datasets.transforms import cnn_val_tf, resnet_train_tf


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


def detect(frame, detector):
    detections = detector.detect_multi_scale(img=frame, scale_factor=1.2, step_ratio=1,
                                             min_size=(100, 100), max_size=(200, 200))
    boxes = []
    for detection in detections:
        x = detection['c']
        y = detection['r']
        w = detection['width']
        h = detection['height']
        boxes.append((x, y, w, h))
    return boxes


def draw(frame, boxes):
    for x, y, w, h in boxes:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)


if __name__ == '__main__':
    # file = lbp_frontal_face_cascade_filename()
    file = "face.xml"
    detector = Cascade(file)

    cap = cv2.VideoCapture(0)
    skip = 5
    i = 0
    boxes = []
    while (True):
        ret, frame = cap.read()
        if i % skip == 0:
            boxes = detect(frame, detector)

        labels = []
        for (x, y, w, h) in boxes:
            patch = Image.fromarray(frame[y:y + h, x:x + w])
            labels.append(predict_patch(patch))

        draw_labels(frame, boxes, labels)
        draw(frame, boxes)
        cv2.imshow('Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1
    cap.release()
    cv2.destroyAllWindows()
