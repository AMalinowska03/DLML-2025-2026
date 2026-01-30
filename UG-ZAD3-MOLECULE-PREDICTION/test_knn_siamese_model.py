import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score
from dataset import BACEDataModule
from model import MoleculeSiameseLearningModel

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_knn_decision_boundary(model, dataloader, k=5, device='cuda'):
    model.eval()
    embs, targets = [], []

    # 1. Zbierz osadzenia
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            emb = model(batch)
            embs.append(emb.cpu())
            targets.append(batch.y.view(-1).cpu())

    X = torch.cat(embs).numpy()
    y = torch.cat(targets).numpy()

    # 2. Trenuj k-NN na zebranych osadzeniach
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)

    # 3. Stwórz siatkę (grid) do narysowania tła
    h = .02  # krok siatki
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predykcja dla każdego punktu siatki
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 4. Wykres
    plt.figure(figsize=(8, 6))
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])  # Kolory tła
    cmap_bold = ['#FF0000', '#0000FF']  # Kolory punktów

    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
    for i in [0, 1]:
        mask = y == i
        plt.scatter(X[mask, 0], X[mask, 1], c=cmap_bold[i], label=f'Klasa {i}', edgecolors='k')

    plt.title(f"Granica decyzyjna k-NN (k={k}) w przestrzeni osadzeń 2D")
    plt.xlabel("Wymiar 1")
    plt.ylabel("Wymiar 2")
    plt.legend()
    plt.show()

# Funkcja pomocnicza do zbierania osadzeń
def get_embeddings(model, loader):
    model.eval()
    embs, labels = [], []
    with torch.no_grad():
        for batch in loader:
            emb = model(batch)
            embs.append(emb.cpu().numpy())
            labels.append(batch.y.view(-1).cpu().numpy())
    return np.concatenate(embs), np.concatenate(labels)


def run_knn_analysis(checkpoint_path, k_values=[3, 5, 7, 11]):
    dm = BACEDataModule()
    dm.setup()

    model = MoleculeSiameseLearningModel.load_from_checkpoint(checkpoint_path)

    train_embs, train_labels = get_embeddings(model, dm.train_dataloader())
    test_embs, test_labels = get_embeddings(model, dm.test_dataloader())

    print(f"Analiza k-NN dla różnych wartości k:")
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_embs, train_labels)
        preds = knn.predict(test_embs)
        score = f1_score(test_labels, preds)
        print(f"k={k:2d} | F1-Score: {score:.4f}")

    plt.figure(figsize=(10, 8))
    plt.scatter(test_embs[:, 0], test_embs[:, 1], c=test_labels, cmap='RdYlGn', alpha=0.6, edgecolors='k')
    plt.title(f"Rozkład cząsteczek w przestrzeni osadzenia (Test Set)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.colorbar(label="Klasa (0=Nieakt, 1=Inhibitor)")
    plt.show()


if __name__ == "__main__":
    PATH = "lightning_logs_classification_test1/version_X/checkpoints/epoch=Y.ckpt"  # Uzupełnij ścieżkę
    run_knn_analysis(PATH)