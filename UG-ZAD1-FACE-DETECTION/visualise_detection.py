import torch
import cv2
import torchvision.transforms.v2.functional as F
from torch.utils.data import DataLoader
import os
import logging
import torchvision.transforms.v2 as transforms

from models.FaceDetector import FaceDetectorLightning
from train_face_detector import collate_fn
from widerface_data_generation.WiderFaceDetectionDataset import WiderFaceDetectionDataset


face_detector_ckpt_v1 = "lightning_logs/face_detector_v1/checkpoints/epoch=4-mAP=40_77.ckpt" # TODO: set when generated
OUTPUT_DIR = "visualization_results"
THRESHOLD = 0.5  # Rysujemy tylko predykcje z pewnością > 50%
NUM_IMAGES_TO_SAVE = 10  # Ile przykładów zapisać
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


test_transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def tensor_to_cv2_image(tensor_img):
    """Konwertuje tensor PyTorch [C, H, W] na obraz OpenCV [H, W, C] BGR, cofając normalizację."""
    # Kopiujemy tensor, żeby nie zmieniać oryginału
    img = tensor_img.clone().cpu()

    # 1. Cofamy normalizację (Denormalization)
    # Wykorzystujemy te same wartości co w test_transform
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean

    # 2. Konwersja do uint8 [0, 255]
    img = torch.clamp(img, 0, 1) # upewniamy się, że zakres to [0, 1]
    img_byte = (img * 255).to(torch.uint8)

    # 3. Zmiana kolejności wymiarów i kolorów
    img_numpy = img_byte.permute(1, 2, 0).numpy()
    img_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGB2BGR)
    return img_bgr


def draw_boxes(image, boxes, color, label_text=None, scores=None):
    """Rysuje ramki na obrazie."""
    img_copy = image.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.int().cpu().numpy()
        # Rysowanie prostokąta
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)

        # Dodawanie opisu (opcjonalne)
        if label_text or scores is not None:
            text = label_text if label_text else ""
            if scores is not None:
                score = scores[i].item()
                text += f" {score:.2f}" if text else f"{score:.2f}"

            if text:
                cv2.putText(img_copy, text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img_copy


def visualize():
    # 1. Ładowanie modelu
    logging.info(f"Loading model from {face_detector_ckpt_v1}...")
    model = FaceDetectorLightning.load_from_checkpoint(face_detector_ckpt_v1)
    model.to(DEVICE)
    model.eval()

    # 2. Przygotowanie danych testowych
    # Używamy funkcji z train_detector.py żeby mieć ten sam split
    logging.info("Preparing test data...")
    test_ds = WiderFaceDetectionDataset(root="data", split="val", transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=4, collate_fn=collate_fn, num_workers=2)

    logging.info(f"Starting visualization. Saving {NUM_IMAGES_TO_SAVE} images to '{OUTPUT_DIR}'...")

    saved_count = 0
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(test_loader):
            if saved_count >= NUM_IMAGES_TO_SAVE:
                break

            # Przeniesienie na GPU
            images = [img.to(DEVICE) for img in images]
            target = targets[0]  # batch_size=1, więc bierzemy pierwszy element

            # Predykcja modelu
            outputs = model(images)
            output = outputs[0]

            # --- RYSOWANIE ---
            # 1. Konwersja tensora na obraz OpenCV
            cv2_img = tensor_to_cv2_image(images[0])

            # 2. Rysowanie GROUND TRUTH (Zielony)
            gt_boxes = target['boxes']
            img_with_gt = draw_boxes(cv2_img, gt_boxes, color=(0, 255, 0), label_text="GT")

            # 3. Filtrowanie i rysowanie PREDYKCJI (Czerwony)
            pred_boxes = []
            pred_scores = []
            for box, score in zip(output['boxes'], output['scores']):
                if score > THRESHOLD:
                    pred_boxes.append(box)
                    pred_scores.append(score)

            if pred_boxes:
                pred_boxes_tensor = torch.stack(pred_boxes)
                pred_scores_tensor = torch.stack(pred_scores)
                # Rysujemy na obrazku, który ma już GT
                final_img = draw_boxes(img_with_gt, pred_boxes_tensor,
                                       color=(0, 0, 255), scores=pred_scores_tensor)
            else:
                final_img = img_with_gt

            # Zapis do pliku
            save_path = os.path.join(OUTPUT_DIR, f"result_{batch_idx}.jpg")
            cv2.imwrite(save_path, final_img)
            saved_count += 1
            logging.info(f"Saved {save_path}")

    logging.info("Done!")


if __name__ == '__main__':
    visualize()