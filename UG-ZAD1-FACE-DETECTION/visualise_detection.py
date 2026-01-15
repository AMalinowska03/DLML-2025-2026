import torch
import cv2
import torchvision.transforms.v2.functional as F
from torch.utils.data import DataLoader
import os
import logging

from models.FaceDetector import FaceDetectorLightning
from train_face_detector import prepare_data, collate_fn

face_detector_ckpt_v1 = "lightning_logs/face_detector_v1/checkpoints/epoch=8-step=11448.ckpt" # TODO: set when generated
OUTPUT_DIR = "visualization_results"
THRESHOLD = 0.5  # Rysujemy tylko predykcje z pewnością > 50%
NUM_IMAGES_TO_SAVE = 10  # Ile przykładów zapisać
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def tensor_to_cv2_image(tensor_img):
    """Konwertuje tensor PyTorch [C, H, W] float na obraz OpenCV [H, W, C] uint8 BGR."""
    # Tensor jest w zakresie [0, 1], konwertujemy na [0, 255] byte
    img_byte = F.to_dtype(tensor_img, torch.uint8, scale=True)
    # Zmiana kolejności wymiarów z [C, H, W] na [H, W, C]
    img_numpy = img_byte.permute(1, 2, 0).cpu().numpy()
    # Konwersja RGB -> BGR (dla OpenCV)
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
    _, _, test_ds = prepare_data()
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)

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