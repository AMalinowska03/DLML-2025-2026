from lightning import Trainer

from dataset import TextDataModule
from models import TransformerPredictor

CONFIG = {
    'batch_size': 128,
    'seq_len': 100,
    'tokenizer_type': 'char',  # lub 'bpe'
    'data_sources': ['data/pantadeusz.txt'],
    # 'model_checkpoint': 'lightning_logs/transformer_1_v1/checkpoints/epoch=84-step=266305.ckpt',
    'model_checkpoint': 'lightning_logs/transformer_1_nlay1_nhead1_dim64/checkpoints/epoch=45-step=128110.ckpt',
}

if __name__ == "__main__":
    trainer = Trainer()

    dm = TextDataModule(CONFIG)
    dm.setup()

    model = TransformerPredictor.load_from_checkpoint(CONFIG['model_checkpoint'])

    trainer.test(model, datamodule=dm)
