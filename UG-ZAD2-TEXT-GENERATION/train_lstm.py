import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping

from dataset import TextDataModule
from models import LSTMPredictor

CONFIG = {
    'batch_size': 128,
    'seq_len': 100,
    'lr': 0.001,
    'max_epochs': 100,
    'embedding_dim': 128,
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.1,
    'tokenizer_type': 'char',  # lub 'bpe'
    'data_sources': ['data/pantadeusz.txt']
}

if __name__ == "__main__":
    dm = TextDataModule(CONFIG)
    dm.setup()

    print(f"Dane przygotowane. Vocab size: {dm.vocab_size}")

    model = LSTMPredictor(
        vocab_size=dm.vocab_size,
        embedding_dim=CONFIG['embedding_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        num_layers=CONFIG['num_layers'],
        dropout=CONFIG['dropout'],
        lr=CONFIG['lr']
    )

    trainer = pl.Trainer(
        max_epochs=CONFIG['max_epochs'],
        callbacks=[EarlyStopping(monitor="val_loss", patience=3)],
        accelerator="auto",
        devices=1
    )

    trainer.fit(model, datamodule=dm)
