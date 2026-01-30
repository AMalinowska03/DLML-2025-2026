import os
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from dataset import BACEDataModule
from model import MoleculeSiameseLearningModel

L.seed_everything(42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from lightning.pytorch.callbacks import Callback


class VisualizerCallback(Callback):
    def __init__(self, train_loader, val_loader, output_dir, k=5):
        super().__init__()
        self.val_loader = val_loader
        self.output_dir = output_dir
        self.k = k

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        emb_dim = pl_module.config['embedding_dim']

        # Generuj wykresy co 20 epok dla osadzeń 1D i 2D
        if emb_dim <= 2 and epoch % 5 == 0:
            self._plot_and_save(pl_module, epoch, emb_dim)

    def _plot_and_save(self, pl_module, epoch, dim):
        pl_module.eval()
        embs, targets = [], []
        device = pl_module.device

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(device)
                emb = pl_module(batch)
                embs.append(emb.cpu())
                targets.append(batch.y.view(-1).cpu())

        X = torch.cat(embs).numpy()
        y = torch.cat(targets).numpy()

        plt.figure(figsize=(10, 7))
        cmap_bold = ListedColormap(['#FF0000', '#0000FF'])  # Czerwony i Niebieski

        if dim == 2:
            # Granica decyzyjna
            h = .01  # Zmniejsz krok dla lepszej gładkości
            x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
            y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

            knn = KNeighborsClassifier(n_neighbors=self.k)
            knn.fit(X, y)
            Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

            plt.pcolormesh(xx, yy, Z, cmap=ListedColormap(['#FFAAAA', '#AAAAFF']), shading='auto', alpha=0.3)
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolors='k', s=30, alpha=0.8)
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

        elif dim == 1:
            jitter = np.random.normal(0, 0.02, size=len(X))
            plt.scatter(X[:, 0], jitter, c=y, cmap=cmap_bold, s=30, alpha=0.6, edgecolors='none')
            plt.axhline(0, color='black', lw=0.5, alpha=0.5)
            plt.ylim(-0.2, 0.2)

        plt.title(f"Rozkład osadzeń {dim}D | Epoka {epoch}")
        plt.grid(True, linestyle='--', alpha=0.5)

        file_path = os.path.join(self.output_dir, f"emb{dim}_epoch{epoch}.png")
        plt.savefig(file_path)
        plt.close()
        pl_module.train()

def test_different_k(model, train_loader, test_loader, k_values=[1, 3, 5, 11, 21]):
    model.eval()
    device = model.device

    def get_data(loader):
        e, l = [], []
        with torch.no_grad():
            for b in loader:
                b = b.to(device)
                e.append(model(b).cpu())
                l.append(b.y.view(-1).cpu())
        return torch.cat(e).numpy(), torch.cat(l).numpy()

    X_train, y_train = get_data(train_loader)
    X_test, y_test = get_data(test_loader)

    print(f"\nWyniki dla osadzenia {model.config['embedding_dim']}D:")
    print("k | Accuracy | F1-Score")
    print("-" * 25)
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        preds = knn.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        print(f"{k:2d} | {acc:.4f}   | {f1:.4f}")


def train_siamese_process(CONFIG, dm, output_visual_dir):
    model = MoleculeSiameseLearningModel(
        in_channels=dm.num_features,
        edge_dim=dm.num_edge_features,
        config=CONFIG
    )

    # Monitorujemy val_loss (z TripletLoss) lub val_knn_acc
    early_stop = EarlyStopping(monitor="val_loss", patience=15, mode="min")

    # Callback do generowania zdjęć do sprawozdania (tylko dla emb 1 i 2)
    visualizer = VisualizerCallback(
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        output_dir=output_visual_dir,
        k=5  # bazowe k do wizualizacji
    )

    logger = TensorBoardLogger(
        save_dir='lightning_logs',
        name=f"lightning_logs_siamese",
        version=f"siamese_emb_{CONFIG['embedding_dim']}_{CONFIG['gnn_type']}_L{CONFIG['num_layers']}_H{CONFIG['hidden_dim']}",
    )

    trainer = L.Trainer(
        max_epochs=200,  # Metric learning często wymaga więcej epok niż klasyfikacja
        callbacks=[early_stop, visualizer],
        accelerator="gpu",
        devices=1,
        logger=logger,
        log_every_n_steps=5
    )

    trainer.fit(model, datamodule=dm)
    return trainer, model


if __name__ == "__main__":
    # Katalog na wizualizacje do raportu
    VIS_DIR = "classification_visualizations_5"
    os.makedirs(VIS_DIR, exist_ok=True)

    # Inicjalizacja danych (stały podział 80/10/10 zapewniony w BACEDataModule)
    dm = BACEDataModule(batch_size=64)
    dm.setup()

    # Wybrane najlepsze konfiguracje do przetestowania w wersji Syjamskiej
    configs = [
        # Reprezentacje 1D i 2D (wymagane wizualizacje)
        {'gnn_type': "Transformer", 'num_layers': 4, 'hidden_dim': 128, 'embedding_dim': 1, 'margin': 0.2,
         'dropout_encoder': 0.1},
        {'gnn_type': "GINE", 'num_layers': 2, 'hidden_dim': 128, 'embedding_dim': 2, 'margin': 0.2,
         'dropout_encoder': 0.1},

        # # Reprezentacje wysokowymiarowe (dla porównania skuteczności)
        # {'gnn_type': "Transformer", 'num_layers': 2, 'hidden_dim': 64, 'embedding_dim': 16, 'margin': 0.3,
        #  'dropout_encoder': 0.1},
        # {'gnn_type': "GINE", 'num_layers': 2, 'hidden_dim': 128, 'embedding_dim': 32, 'margin': 0.3,
        #  'dropout_encoder': 0.1},
    ]

    for config in configs:
        print(f"\n>>> Trenowanie reprezentacji: {config['embedding_dim']}D ({config['gnn_type']})")

        trainer, trained_model = train_siamese_process(config, dm, VIS_DIR)

        test_different_k(
            model=trained_model,
            train_loader=dm.train_dataloader(),
            test_loader=dm.test_dataloader(),
            k_values=[1, 3, 5, 11, 21]
        )