import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping

from dataset import TextDataModule
from models import TransformerPredictor

CONFIG = {
    'batch_size': 128,
    'seq_len': 100,
    'lr': 0.001,
    'max_epochs': 10,
    'embedding_dim': 128,
    'num_layers': 2,
    'nhead': 4,
    'tokenizer_type': 'bpe',  # lub 'char'
    'data_sources': ['data/pantadeusz.txt', 'data/quovadis.txt']
}

if __name__ == "__main__":
    dm = TextDataModule(CONFIG)

    print(f"Dane przygotowane. Vocab size: {dm.vocab_size}")

    model = TransformerPredictor(
        vocab_size=dm.vocab_size,
        embedding_dim=CONFIG['embedding_dim'],
        nhead=CONFIG['nhead'],
        num_layers=CONFIG['num_layers'],
        lr=CONFIG['lr']
    )

    trainer = pl.Trainer(
        max_epochs=CONFIG['max_epochs'],
        callbacks=[EarlyStopping(monitor="val_loss", patience=3)],
        accelerator="auto",
        devices=1
    )

    trainer.fit(model, datamodule=dm)
