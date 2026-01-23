from lightning import Trainer

from dataset import TextDataModule
from models import LSTMPredictor

CONFIG = {
    'batch_size': 128,
    'seq_len': 100,
    'tokenizer_type': 'char',
    'data_sources': ['data/pantadeusz.txt'],
    # 'model_checkpoint': 'lightning_logs/lstm_v2_128_512_nlay4/checkpoints/epoch=30-step=86335.ckpt',
    'model_checkpoint': 'lightning_logs/lstm_v1/checkpoints/epoch=96-step=270145.ckpt',

}

if __name__ == "__main__":
    trainer = Trainer()

    dm = TextDataModule(CONFIG)
    dm.setup()

    model = LSTMPredictor.load_from_checkpoint(CONFIG['model_checkpoint'])

    trainer.test(model, datamodule=dm)
