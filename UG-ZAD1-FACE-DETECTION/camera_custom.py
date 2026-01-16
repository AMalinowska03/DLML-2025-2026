import cv2
from PIL import Image
from models.FaceDetector import FaceDetectorLightning
import torchvision.transforms.v2 as transforms

import torch

from models.LightningModel import LightningModel
from models.GenderCNN import GenderCNN
from models.EyeglassesResNet import EyeglassesResNet
from datasets.transforms import cnn_val_tf, resnet_val_tf


male_ckpt = "lightning_logs/gender_v1/checkpoints/epoch=24-step=31800.ckpt"
glasses_ckpt_v1 = "lightning_logs/glasses_v1/checkpoints/epoch=9-step=12720.ckpt"
glasses_ckpt_v2 = "lightning_logs/glasses_v2/checkpoints/epoch=8-step=11448.ckpt"
face_detector_ckpt_v1 = "lightning_logs/face_detector_v1/checkpoints/epoch=4-mAP=40_77.ckpt" # TODO: set when generated

male_model = LightningModel.load_from_checkpoint(
    male_ckpt,
    model=GenderCNN(),
    pos_weight=torch.tensor(1.0)
)
male_model.eval()
male_model.to("cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu")

glasses_model = LightningModel.load_from_checkpoint(
    glasses_ckpt_v2,
    model=EyeglassesResNet(),
    pos_weight=torch.tensor(1.0)
)
glasses_model.eval()
glasses_model.to("cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu")

detector_model = FaceDetectorLightning.load_from_checkpoint(face_detector_ckpt_v1)
detector_model.eval()
detector_model.to("cuda" if torch.cuda.is_available() else "xpu" if torch.xpu.is_available() else "cpu")


def draw_labels(frame, boxes, labels):
    for (x, y, w, h), (male, glasses) in zip(boxes, labels):
        text = f"{'Male' if male else 'Female'}, {'Glasses' if glasses else 'No Glasses'}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def predict_patch(patch):
    # Sprawdzamy, na jakim urządzeniu jest model (np. male_model)
    device = next(male_model.parameters()).device

    # Tworzymy tensory i natychmiast przesyłamy je na to samo urządzenie (.to(device))
    male_x = cnn_val_tf(patch).unsqueeze(0).to(device)
    glasses_x = resnet_val_tf(patch).unsqueeze(0).to(device)

    with torch.no_grad():
        # Modele są już na 'device', teraz dane też tam są
        male_logit = male_model(male_x).squeeze(1)
        glasses_logit = glasses_model(glasses_x).squeeze(1)

        male_prob = torch.sigmoid(male_logit).item()
        glasses_prob = torch.sigmoid(glasses_logit).item()

    return male_prob > 0.5, glasses_prob > 0.5

def expand_box(x, y, w, h, img_w, img_h, margin=1):
    pad_w = int(w * margin)
    pad_h = int(h * margin)

    x1 = max(0, x - pad_w)
    y1 = max(0, y - pad_h)
    x2 = min(img_w, x + w + pad_w)
    y2 = min(img_h, y + h + pad_h)

    return x1, y1, x2 - x1, y2 - y1


def detect(frame, model, threshold=0.7):
    # frame to tensor
    img_tensor = transforms.functional.to_dtype(transforms.functional.to_image(frame), torch.float32) / 255.0
    img_tensor = img_tensor.to(next(model.parameters()).device)

    with torch.no_grad():
        prediction = model([img_tensor])[0]

    boxes = []
    for box, score in zip(prediction['boxes'], prediction['scores']):
        if score > threshold:
            x1, y1, x2, y2 = box.int().cpu().numpy()
            boxes.append((x1, y1, x2 - x1, y2 - y1))  # Konwersja na [x, y, w, h] dla reszty kodu
    return boxes

def draw(frame, boxes):
    for x, y, w, h in boxes:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    skip = 5
    i = 0
    boxes = []
    while (True):
        ret, frame = cap.read()
        if i % skip == 0:
            boxes = detect(frame, detector_model)

        labels = []
        for (x, y, w, h) in boxes:
            h_img, w_img, _ = frame.shape
            x, y, w, h = expand_box(x, y, w, h, w_img, h_img)
            patch = Image.fromarray(frame[y:y + h, x:x + w])
            labels.append(predict_patch(patch))

        draw_labels(frame, boxes, labels)
        draw(frame, boxes)
        cv2.imshow('System Detekcji i Atrybutów (Ocena 5)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        i += 1
    cap.release()
    cv2.destroyAllWindows()
