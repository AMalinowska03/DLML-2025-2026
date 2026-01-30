import os

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import QM9DataModule
from model import MoleculeNetRegressionModel

L.seed_everything(42)


def visualize_regression_1d(model, dm, title="QM9 - Aproksymacja (1D)", filename=None):
    model.eval()
    embs, targets = [], []

    with torch.no_grad():
        for batch in dm.test_dataloader():
            out, emb = model(batch)
            embs.append(emb)
            targets.append(batch.y.view(-1))

    embs = torch.cat(embs).cpu().numpy().flatten()
    targets = torch.cat(targets).cpu().numpy()

    x_range = np.linspace(embs.min(), embs.max(), 200)
    x_range_tensor = torch.tensor(x_range.reshape(-1, 1), dtype=torch.float32).to(model.device)
    with torch.no_grad():
        y_range = model.predictor(x_range_tensor).cpu().numpy()

    plt.figure(figsize=(8, 6))
    plt.scatter(embs, targets, alpha=0.5, label="Dane testowe (oczekiwane)", s=10)
    plt.plot(x_range, y_range, color='red', lw=3, label="Funkcja aproksymacji")
    plt.title(title)
    plt.xlabel("Embedding (1D)")
    plt.ylabel("Wartość przewidywana")
    plt.legend()

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()


def visualize_regression_2d(model, dm, title="QM9 - Powierzchnia Aproksymacji (2D)", filename=None):
    model.eval()
    embs, targets = [], []

    with torch.no_grad():
        for batch in dm.test_dataloader():
            out, emb = model(batch)
            embs.append(emb)
            targets.append(batch.y.view(-1))

    embs = torch.cat(embs).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()

    x_min, x_max = embs[:, 0].min() - 0.5, embs[:, 0].max() + 0.5
    y_min, y_max = embs[:, 1].min() - 0.5, embs[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(model.device)
    with torch.no_grad():
        grid_values = model.predictor(grid_tensor).reshape(xx.shape).cpu().numpy()

    plt.figure(figsize=(12, 8))

    contour = plt.contourf(xx, yy, grid_values, levels=50, cmap='viridis', alpha=0.8)
    plt.colorbar(contour, label="Przewidywany Moment Dipolowy")

    lines = plt.contour(xx, yy, grid_values, levels=10, colors='white', alpha=0.4, linewidths=0.8)

    plt.clabel(lines, inline=True, fontsize=9, fmt='%.2f', colors='black')

    plt.scatter(embs[:, 0], embs[:, 1], c=targets, cmap='viridis',
                edgecolors='black', s=25, linewidths=0.5, label="Cząsteczki (wartość rzeczywista)")

    plt.title(title)
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")
    plt.grid(alpha=0.2)

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()


def visualize_regression_3d(model, dm, title="QM9 - Powierzchnia Aproksymacji (3D)", filename=None):
    model.eval()
    embs, targets = [], []

    with torch.no_grad():
        for batch in dm.test_dataloader():
            out, emb = model(batch)
            embs.append(emb)
            targets.append(batch.y.view(-1))

    embs = torch.cat(embs).cpu().numpy()
    targets = torch.cat(targets).cpu().numpy()

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(embs[:, 0], embs[:, 1], targets, c=targets, cmap='viridis', s=20)

    x = np.linspace(embs[:, 0].min(), embs[:, 0].max(), 30)
    y = np.linspace(embs[:, 1].min(), embs[:, 1].max(), 30)
    X, Y = np.meshgrid(x, y)

    grid = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float32)
    with torch.no_grad():
        Z = model.predictor(grid).reshape(X.shape).numpy()

    ax.set_title(title)
    ax.plot_surface(X, Y, Z, alpha=0.3, cmap='viridis')
    ax.set_xlabel('Embedding Dimension 1')
    ax.set_ylabel('Embedding Dimension 2')
    ax.set_zlabel('Moment Dipolowy')

    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()


CONFIG = {
    'batch_size': 256,
    'embedding_dim': 2,  # [1, 2]
    'checkpoint_path': 'lightning_logs/regression_final_plus/emb2_MLP_hidden128/checkpoints/epoch=68-step=28221.ckpt',
}

if __name__ == "__main__":
    dm = QM9DataModule(batch_size=CONFIG['batch_size'])
    dm.setup()

    model = MoleculeNetRegressionModel.load_from_checkpoint(CONFIG['checkpoint_path'])

    trainer = L.Trainer()

    trainer.test(model, datamodule=dm)

    if CONFIG['embedding_dim'] == 1:
        visualize_regression_1d(model, dm)
    elif CONFIG['embedding_dim'] == 2:
        visualize_regression_2d(model, dm)
        visualize_regression_3d(model, dm)
