# import torch
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, classification_report
# import os

# from models.LightningModel import LightningModel
# from models.GenderCNN import GenderCNN
# from models.EyeglassesResNet import EyeglassesResNet
# from datasets.test_gender_data_loaders import test_loader_gender, wider_male_loader
# from datasets.test_glasses_data_loaders import test_loader_glass, wider_glasses_loader

# male_ckpt = "lightning_logs/gender_v1/checkpoints/epoch=24-step=31800.ckpt"
# glasses_ckpt_v1 = "lightning_logs/glasses_v1/checkpoints/epoch=9-step=12720.ckpt"
# glasses_ckpt_v2 = "lightning_logs/glasses_v2/checkpoints/epoch=8-step=11448.ckpt"
# datapath = "data/results/"

# def generate_confusion_matrix(model, loader, title, filename):
#     if not os.path.exists(datapath):
#         os.makedirs(datapath)
#     model.eval()
#     model.cuda()
#     all_preds = []
#     all_targets = []

#     with torch.no_grad():
#         for x, y in loader:
#             logits = model(x.cuda()).squeeze(1)
#             preds = (torch.sigmoid(logits) > 0.5).int().cpu().numpy()
#             all_preds.extend(preds)
#             all_targets.extend(y.int().numpy())

#     cm = confusion_matrix(all_targets, all_preds)

#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
#     plt.xlabel('Predicted')
#     plt.ylabel('Ground Truth')
#     plt.title(f'Confusion Matrix: {title}')
#     plt.savefig(f"{filename}.png")
#     print(f"Saved: {filename}.png")
#     print(classification_report(all_targets, all_preds))


# male_model = LightningModel.load_from_checkpoint(
#     male_ckpt,
#     model=GenderCNN(),
#     pos_weight=torch.tensor(1.0)
# )
# male_model.eval().cuda()

# glasses_model = LightningModel.load_from_checkpoint(
#     glasses_ckpt_v2,
#     model=EyeglassesResNet(),
#     pos_weight=torch.tensor(1.0)
# )
# glasses_model.eval().cuda()

# if __name__ == "__main__":
#     generate_confusion_matrix(male_model, test_loader_gender, "Gender (CelebA Test)", f"{datapath}cm_gender_celeba")
#     generate_confusion_matrix(glasses_model, test_loader_glass, "Eyeglasses (CelebA Test)", f"{datapath}cm_glass_celeba")

#     generate_confusion_matrix(male_model, wider_male_loader, "Gender (WIDERFace)", f"{datapath}cm_gender_wider")
#     generate_confusion_matrix(glasses_model, wider_glasses_loader, "Eyeglasses (WIDERFace)", f"{datapath}cm_glass_wider")

import pandas as pd

def generate_latex_table_rows(csv_file):
    # Wczytanie pliku CSV (nagłówek w pierwszym rzędzie)
    # Zakładamy: kolumna 0 = bibkey, kolumna 1 = tytuł
    df = pd.read_csv(csv_file)
    
    # Opcjonalnie: filtrowanie tylko zaakceptowanych prac (jeśli Twój CSV to zawiera)
    if 'status' in df.columns:
        df = df[df['status'] == 'Accepted'].reset_index(drop=True)
    
    print("% --- START GENEROWANYCH RZĘDÓW ---")
    
    for index, row in df.iterrows():
        # Pobieranie danych wg Twojej specyfikacji
        bibkey = str(row.iloc[0]) # Pierwsza kolumna
        title = str(row.iloc[1])  # Druga kolumna
        item_id = index + 1       # Numer porządkowy
        
        # Oczyszczanie tytułu ze znaków specjalnych LaTeX (np. &, _, %)
        clean_title = title.replace('&', '\\&').replace('_', '\\_').replace('%', '\\%')
        
        # Generowanie rzędu (używamy \cite, bo to odniesienie do bibliografii)
        # Jeśli upierasz się przy \ref, zmień \cite na \ref poniżej
        latex_row = f"{item_id} & {clean_title} & \\cite{{{bibkey}}} \\\\ \\hline"
        
        print(latex_row)
        
    print("% --- KONIEC GENEROWANYCH RZĘDÓW ---")

# Uruchomienie (podaj nazwę swojego pliku)
generate_latex_table_rows('articles.csv')