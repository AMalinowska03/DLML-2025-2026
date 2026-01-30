import copy

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from dataset import BACEDataModule
from model import MoleculeNetClassificationModel

L.seed_everything(42)


# relu->norm, mean pool
# 25 - hd 16, mlphd 64, lay 3, dropout 0.1
# 26 - hd 16, mlphd 64, lay 3, dropout 0.2
# 27 - hd 32, mlphd 64, lay 3, dropout 0.1
# 28 - hd 32, mlphd 64, lay 3, dropout 0.2

# changed order to norm->relu
# 29 - hd 32, mlphd 64, lay 3, dropout 0.1
# 30 - hd 32, mlphd 64, lay 2, dropout 0.1

# max pool
# 31 - hd 32, mlphd 64, lay 2, dropout 0.1
# 32 - hd 32, mlphd 64, lay 3, dropout 0.1


def train_process(CONFIG, logger = None):
    dm = BACEDataModule(batch_size=CONFIG['batch_size'])
    dm.setup()

    model = MoleculeNetClassificationModel(in_channels=dm.num_features, edge_dim=dm.num_edge_features, config=CONFIG,
                                           pos_weight=dm.pos_weight)

    early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    if logger is None:
        logger = TensorBoardLogger(
            save_dir='lightning_logs',
            name=f"embedding_{CONFIG['embedding_dim']}",
            version=f"{CONFIG['gnn_type']}_layers{CONFIG['num_layers']}_dim{CONFIG['hidden_dim']}_{CONFIG['predictor_type']}{CONFIG['mlp_hidden_dim']}",
            )

    trainer = L.Trainer(
        max_epochs=1000,
        callbacks=[early_stop],
        accelerator="auto",
        devices=1,
        logger=logger,
    )

    trainer.fit(model, datamodule=dm)
    return trainer


def check_values(BEST_CONFIG, CONFIG, best_auroc, current_auroc):
    if current_auroc > best_auroc:
        best_auroc = current_auroc
        return CONFIG, best_auroc
    else:
        return BEST_CONFIG, best_auroc


if __name__ == "__main__":
    part = 2

    if part == 1:
        for gnn_type in ['GINE']:
            for embedding_dim in [1, 2, 16, 32]:
                for hidden_dim in [8, 16, 32, 64, 128]:
                    for num_layers in [1, 2, 4]:
                        for mlp_hidden_dim in [64, 128]:
                            CONFIG = {'batch_size': 64, 'gnn_type': gnn_type, 'num_layers': num_layers,
                                      'hidden_dim': hidden_dim, 'embedding_dim': embedding_dim, 'predictor_type': "MLP",
                                      'mlp_hidden_dim': mlp_hidden_dim, 'out_dim': 2, 'dropout_encoder': 0.1}
                            train_process(CONFIG)

                        CONFIG = {
                            'batch_size': 64,
                            'gnn_type': gnn_type,  # ['GCN', 'Transformer', "GINE"]
                            'num_layers': num_layers,
                            'hidden_dim': hidden_dim,
                            'embedding_dim': embedding_dim,  # [1, 2]
                            'predictor_type': "Linear",  # ['Linear', 'MLP']
                            'mlp_hidden_dim': None,  # jeśli nie predictor_type != MLP to może być None
                            'out_dim': 2,
                            'dropout_encoder': 0.1
                        }
                        train_process(CONFIG)

    if part == 2:
        configs = [
            # emb 1
            {'batch_size': 64, 'gnn_type': "Transformer", 'num_layers': 1,
             'hidden_dim': 64, 'embedding_dim': 1, 'predictor_type': "MLP",
             'mlp_hidden_dim': 128, 'out_dim': 2, 'dropout_encoder': 0.1},
            # linear emb 1
            {'batch_size': 64, 'gnn_type': "GINE", 'num_layers': 4,
             'hidden_dim': 128, 'embedding_dim': 1, 'predictor_type': "Linear",
             'mlp_hidden_dim': None, 'out_dim': 2, 'dropout_encoder': 0.1},
            # mlp emb 2
            {'batch_size': 64, 'gnn_type': "Transformer", 'num_layers': 2,
             'hidden_dim': 128, 'embedding_dim': 2, 'predictor_type': "MLP",
             'mlp_hidden_dim': 128, 'out_dim': 2, 'dropout_encoder': 0.1},
            # linear emb 2
            {'batch_size': 64, 'gnn_type': "GINE", 'num_layers': 2,
             'hidden_dim': 128, 'embedding_dim': 2, 'predictor_type': "Linear",
             'mlp_hidden_dim': None, 'out_dim': 2, 'dropout_encoder': 0.1},
            # emb 16
            {'batch_size': 64, 'gnn_type': "Transformer", 'num_layers': 2,
             'hidden_dim': 32, 'embedding_dim': 16, 'predictor_type': "Linear",
             'mlp_hidden_dim': None, 'out_dim': 2, 'dropout_encoder': 0.1},
            # emb 32
            {'batch_size': 64, 'gnn_type': "GINE", 'num_layers': 2,
             'hidden_dim': 128, 'embedding_dim': 32, 'predictor_type': "Linear",
             'mlp_hidden_dim': None, 'out_dim': 2, 'dropout_encoder': 0.1},
        ]

        for CONFIG in configs:
            for i in range(4):
                logger = TensorBoardLogger(
                    save_dir='lightning_logs',
                    name=f"classification_final",
                    version=f"embedding_{CONFIG['embedding_dim']}_{CONFIG['gnn_type']}_layers{CONFIG['num_layers']}_dim{CONFIG['hidden_dim']}_{CONFIG['predictor_type']}{CONFIG['mlp_hidden_dim']}_{i}",
                )

                train_process(CONFIG, logger)
