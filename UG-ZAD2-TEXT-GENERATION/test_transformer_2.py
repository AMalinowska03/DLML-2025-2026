from lightning import Trainer

from dataset import TextDataModule
from models import TransformerPredictor

CONFIG = {
    'batch_size': 128,
    'seq_len': 100,
    'tokenizer_type': 'char',  # lub 'bpe'
    'data_sources': ['data/pantadeusz.txt'],
    'model_checkpoint': 'lightning_logs/transformer_2_v1',
}

if __name__ == "__main__":
    trainer = Trainer()

    dm = TextDataModule(CONFIG)
    dm.setup()

    model = TransformerPredictor.load_from_checkpoint(CONFIG['model_checkpoint'])

    result = trainer.test(model, datamodule=dm)
    print(result)
