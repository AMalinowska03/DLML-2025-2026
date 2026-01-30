import torch
import os
import cv2
import numpy as np
from tqdm import tqdm
from models.LightningModel import LightningModel
from models.GenderCNN import GenderCNN
from models.EyeglassesResNet import EyeglassesResNet

# Import loaderów (upewnij się, że ścieżki są poprawne)
from datasets.test_gender_data_loaders import test_loader_gender, wider_male_loader
from datasets.test_glasses_data_loaders import test_loader_glass, wider_glasses_loader

# Konfiguracja
OUTPUT_DIR = "results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denormalize(tensor):
    """Cofa normalizację (mean=0.5, std=0.5) do zapisu obrazu."""
    img = tensor.clone().cpu()
    img = img * 0.5 + 0.5  # Zakładając normalizację o której pisałeś w raporcie
    img = torch.clamp(img, 0, 1)
    img = (img * 255).permute(1, 2, 0).numpy().astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def save_classification_samples(model, dataloader, dataset_name, model_name):
    print(f"Przetwarzanie: {dataset_name} | Model: {model_name}...")
    model.to(DEVICE)
    model.eval()

    # Słownik do śledzenia czy już znaleźliśmy dany typ
    found = {"TP": False, "TN": False, "FP": False, "FN": False}

    with torch.no_grad():
        for batch in tqdm(dataloader):
            x, y = batch
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x).squeeze(1)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            for i in range(x.size(0)):
                label = y[i].item()
                pred = preds[i].item()

                assessment = ""
                if label == 1 and pred == 1:
                    assessment = "TP"
                elif label == 0 and pred == 0:
                    assessment = "TN"
                elif label == 0 and pred == 1:
                    assessment = "FP"
                elif label == 1 and pred == 0:
                    assessment = "FN"

                # Zapisz tylko jeśli jeszcze nie mamy tego typu dla tej kombinacji
                if not found[assessment]:
                    img_bgr = denormalize(x[i])
                    filename = f"{dataset_name}_{model_name}_{assessment}.jpg"
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    cv2.imwrite(filepath, img_bgr)
                    found[assessment] = True

            # Jeśli znaleźliśmy wszystkie 4 typy, przejdź do następnego zbioru
            if all(found.values()):
                break


if __name__ == "__main__":
    # 1. TEST GENDER
    gender_checkpoint = "lightning_logs_classification_test1/gender_v1/checkpoints/epoch=24-step=31800.ckpt"
    gender_model = LightningModel.load_from_checkpoint(
        gender_checkpoint, model=GenderCNN(), pos_weight=torch.tensor(1.0)
    )

    save_classification_samples(gender_model, test_loader_gender, "CelebA", "GenderCNN")
    save_classification_samples(gender_model, wider_male_loader, "WiderFace", "GenderCNN")

    # 2. TEST GLASSES
    glasses_checkpoint = "lightning_logs_classification_test1/glasses_v2/checkpoints/epoch=8-step=11448.ckpt"
    glasses_model = LightningModel.load_from_checkpoint(
        glasses_checkpoint, model=EyeglassesResNet(), pos_weight=torch.tensor(1.0)
    )

    save_classification_samples(glasses_model, test_loader_glass, "CelebA", "EyeGlassesRes")
    save_classification_samples(glasses_model, wider_glasses_loader, "WiderFace", "EyeGlassesRes")

    print("\nZakończono! Zdjęcia znajdują się w folderze 'results'.")