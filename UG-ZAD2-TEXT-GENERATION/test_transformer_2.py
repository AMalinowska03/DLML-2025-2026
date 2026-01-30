from lightning import Trainer

from dataset import TextDataModule
from models import TransformerPredictor

CONFIG = {
    'batch_size': 128,
    'seq_len': 100,
    'vocab_size': 10000, # 5000
    'tokenizer_type': 'custom',  # lub 'bpe'
    'tokenizer_path': 'lightning_logs_classification_test1/custom_tokenizer',
    'data_sources': ['data/pantadeusz.txt', 'data/quovadis.txt', 'data/potoptompierwszy.txt'],
    'model_checkpoint': 'lightning_logs_classification_test1/transformer_2_v1_10k_vocab_part_2/checkpoints/epoch=81-step=305286.ckpt',
    # 'model_checkpoint': 'lightning_logs_classification_test1/transformer_2_v1_5k_vocab/checkpoints/epoch=66-step=280060.ckpt',
}

if __name__ == "__main__":
    trainer = Trainer()

    dm = TextDataModule(CONFIG)
    dm.setup()

    model = TransformerPredictor.load_from_checkpoint(CONFIG['model_checkpoint'])

    trainer.test(model, datamodule=dm)
