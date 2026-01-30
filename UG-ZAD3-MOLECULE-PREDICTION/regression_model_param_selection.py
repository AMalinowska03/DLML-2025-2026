import lightning as L
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from dataset import QM9DataModule
from model import MoleculeNetRegressionModel

import itertools
import copy


L.seed_everything(42)

CONFIG = {
    'batch_size': 256,
    'gnn_type': "GMM", # ['GMM', 'Transformer']
    'num_layers': 6,
    'hidden_dim': 128,
    'embedding_dim': 2, # [1, 2]
    'predictor_type': "Linear", # ['Linear', 'MLP']
    'mlp_hidden_dim': 128,  # jeśli nie predictor_type != MLP to może być None
    'out_dim': 1,
    'dropout_encoder': 0.1
}

search_space = {
    'hidden_dim': [128, 256],
    'embedding_dim': [1, 2, 16, 32],
    'predictor_type': ["Linear", "MLP"]
}

if __name__ == "__main__":
    dm = QM9DataModule(batch_size=CONFIG['batch_size'])
    dm.setup()

    keys, values = zip(*search_space.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for i, exp_params in enumerate(experiments):
        current_config = copy.deepcopy(CONFIG)
        current_config.update(exp_params)

        model = MoleculeNetRegressionModel(
            in_channels=dm.num_features,
            edge_dim=dm.num_edge_features,
            config=current_config
        )

        early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")

        version_name = f"emb{current_config['embedding_dim']}_{current_config['predictor_type']}_hidden{current_config['hidden_dim']}"

        logger = TensorBoardLogger(
            save_dir="lightning_logs",
            name="regression_param_selection",
            version=version_name
        )

        trainer = L.Trainer(
            max_epochs=10,
            callbacks=[early_stop],
            accelerator="auto",
            devices=1,
            logger=logger
        )

        trainer.fit(model, datamodule=dm)
