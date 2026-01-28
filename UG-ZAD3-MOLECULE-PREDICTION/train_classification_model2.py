import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from dataset import BACEDataModule
from model import MoleculeNetClassificationModel

L.seed_everything(42)

CONFIG = {
    'batch_size': 128,
    'gnn_type': "GCN", # ['GCN', 'Transformer']
    'hidden_dim': 32,
    'embedding_dim': 2, # [1, 2]
    'predictor_type': "MLP", # ['Linear', 'MLP']
    'mlp_hidden_dim': 64, # jeśli nie predictor_type != MLP to może być None
    'out_dim': 1,
    'num_layers': 2,
    'dropout_encoder': 0.1
}

if __name__ == "__main__":
    dm = BACEDataModule(batch_size=CONFIG['batch_size'])
    dm.setup()

    model = MoleculeNetClassificationModel(in_channels=dm.num_features, config=CONFIG, pos_weight=dm.pos_weight)

    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    trainer = L.Trainer(
        max_epochs=100,
        callbacks=[early_stop],
        accelerator="auto",
        devices=1
    )

    trainer.fit(model, datamodule=dm)
