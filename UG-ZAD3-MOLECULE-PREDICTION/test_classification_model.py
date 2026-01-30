import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataset import BACEDataModule
from model import MoleculeNetClassificationModel

import matplotlib
matplotlib.use('TkAgg') # Or 'Qt5Agg' if you have PyQt installed

L.seed_everything(42)


def visualize_classification_1d(model, dm, name, title="BACE - Granica decyzyjna (1D)"):
    model.eval()
    embs, labels = [], []

    with torch.no_grad():
        for batch in dm.test_dataloader():
            logits, emb = model(batch)
            embs.append(emb)
            labels.append(batch.y.view(-1))

    embs = torch.cat(embs).cpu().numpy().flatten()
    labels = torch.cat(labels).cpu().numpy()

    x_range = np.linspace(embs.min() - 0.5, embs.max() + 0.5, 500)
    x_range_tensor = torch.tensor(x_range.reshape(-1, 1), dtype=torch.float32).to(model.device)

    with torch.no_grad():
        grid_logits = model.predictor(x_range_tensor)
        grid_probs = torch.softmax(grid_logits, dim=1)[:, 1].cpu().numpy()

    boundary_idx = np.argmin(np.abs(grid_probs - 0.5))
    boundary_x = x_range[boundary_idx]

    plt.figure(figsize=(10, 6))

    plt.plot(x_range, grid_probs, color='blue', lw=2, label='Prawdopodobieństwo modelu', zorder=2)

    jitter = np.random.uniform(-0.03, 0.03, size=len(labels))
    plt.scatter(embs, labels + jitter, c=labels, cmap='RdYlGn', edgecolors='k',
                s=30, alpha=0.6, label='Cząsteczki (0=Nieakt., 1=Inhibitor)', zorder=3)

    plt.axvline(x=boundary_x, color='black', linestyle='--', lw=2,
                label=f'Granica decyzyjna (x ≈ {boundary_x:.2f})', zorder=4)

    plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)

    plt.title(title)
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Prawdopodobieństwo klasy 1")
    plt.ylim(-0.1, 1.1)
    plt.legend(loc='center right')
    plt.grid(alpha=0.3)

    plt.savefig("classification_visualization/"+name)


def visualize_classification_2d(model, dm, name, title="BACE - Granica decyzyjna (2D)"):
    model.eval()
    embs, labels = [], []

    with torch.no_grad():
        for batch in dm.test_dataloader():
            logits, emb = model(batch)
            embs.append(emb)
            labels.append(batch.y.view(-1))

    embs = torch.cat(embs).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()

    x_min, x_max = embs[:, 0].min() - 1, embs[:, 0].max() + 1
    y_min, y_max = embs[:, 1].min() - 1, embs[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(model.device)
    with torch.no_grad():
        grid_logits = model.predictor(grid_tensor)
        grid_probs = torch.softmax(grid_logits, dim=1)[:, 1].reshape(xx.shape).cpu().numpy()

    plt.figure(figsize=(10, 7))

    decision_map = (grid_probs > 0.5).astype(float)
    plt.contourf(xx, yy, decision_map, alpha=0.15, cmap='RdYlGn')

    boundary = plt.contour(xx, yy, grid_probs, levels=[0.5], colors='black', linewidths=2, linestyles='--')
    plt.clabel(boundary, inline=True, fontsize=10, fmt={0.5: 'Granica decyzyjna (p=0.5)'})

    scatter = plt.scatter(embs[:, 0], embs[:, 1], c=labels, cmap='RdYlGn', edgecolors='k', s=40, alpha=0.8)

    plt.title(title)
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")

    handles, _ = scatter.legend_elements()
    plt.legend(handles, ["Nie-inhibitor (0)", "Inhibitor (1)"], loc="upper right")

    plt.grid(alpha=0.2)
    plt.savefig("classification_visualization/"+name)


configs = [
    {
        'batch_size': 64,
        'embedding_dim': 1,
        'checkpoint_path': r'lightning_logs\classification_final\embedding_1_Transformer_layers1_dim64_MLP128_2\checkpoints\epoch=18-step=361.ckpt',
        'name': r"embedding_1_mlp"
    },
    {
        'batch_size': 64,
        'embedding_dim': 1,
        'checkpoint_path': r'lightning_logs\classification_final\embedding_1_GINE_layers4_dim128_LinearNone_1\checkpoints\epoch=61-step=1178.ckpt',
        'name': r"embedding_1_linear"
    },
    {
        'batch_size': 64,
        'embedding_dim': 2,
        'checkpoint_path': r'lightning_logs\classification_final\embedding_2_GINE_layers2_dim128_LinearNone_3\checkpoints\epoch=43-step=836.ckpt',
        'name': r"embedding_2_linear"
    },
    {
        'batch_size': 64,
        'embedding_dim': 2,
        'checkpoint_path': r'lightning_logs\classification_final\embedding_2_Transformer_layers2_dim128_MLP128_1\checkpoints\epoch=23-step=456.ckpt',
        'name': r"embedding_2_mlp"
    },
    {
        'batch_size': 64,
        'embedding_dim': 16,
        'checkpoint_path': r'lightning_logs\classification_final\embedding_16_Transformer_layers2_dim32_LinearNone_0\checkpoints\epoch=95-step=1824.ckpt',
    },
    {
        'batch_size': 64,
        'embedding_dim': 32,
        'checkpoint_path': r'lightning_logs\classification_final\embedding_32_GINE_layers2_dim128_LinearNone_2\checkpoints\epoch=38-step=741.ckpt',
    },
]


# CONFIG = {
#     'batch_size': 64,
#     'embedding_dim': 1,  # [1, 2]
#     'checkpoint_path': 'lightning_logs_classification_test1/version_13/checkpoints/epoch=36-step=370.ckpt',
# }

if __name__ == "__main__":
    for CONFIG in configs:
        dm = BACEDataModule(batch_size=CONFIG['batch_size'])
        dm.setup()

        model = MoleculeNetClassificationModel.load_from_checkpoint(CONFIG['checkpoint_path'])

        trainer = L.Trainer()

        trainer.test(model, datamodule=dm)

        if CONFIG['embedding_dim'] == 1:
            visualize_classification_1d(model, dm, CONFIG['name'])
        elif CONFIG['embedding_dim'] == 2:
            visualize_classification_2d(model, dm, CONFIG['name'])
