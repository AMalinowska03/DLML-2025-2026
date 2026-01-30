import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torchmetrics
from torch_geometric.nn import GINEConv, GMMConv, TransformerConv, global_mean_pool, global_add_pool, global_max_pool
from pytorch_metric_learning import losses, miners, distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class GNNEncoder(nn.Module):
    def __init__(self, in_channels, edge_dim, hidden_dim, embedding_dim, gnn_type="GINE", num_layers=2, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.gnn_type = gnn_type
        self.dropout = dropout

        self.node_init = nn.Linear(in_channels, hidden_dim)

        if gnn_type != "GMM":
            self.edge_init = nn.Linear(edge_dim, hidden_dim)

        for _ in range(num_layers):
            if gnn_type == "GINE":
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.layers.append(GINEConv(mlp, edge_dim=hidden_dim))
            elif gnn_type == "Transformer":
                self.layers.append(TransformerConv(hidden_dim, hidden_dim, heads=4, concat=False, edge_dim=hidden_dim))
            elif gnn_type == "GMM":
                self.layers.append(GMMConv(hidden_dim, hidden_dim, dim=edge_dim, kernel_size=5))
            else:
                raise AttributeError("'gnn_type' must be GINE or Transformer or GMM")

            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.final_proj = nn.Linear(hidden_dim, embedding_dim)
        self.act = nn.ReLU()

    def forward(self, batch):
        x = self.node_init(batch.x.float())
        edge_index = batch.edge_index

        if self.gnn_type == "GMM":
            edge_info = batch.edge_attr.float()
        else:
            edge_info = self.edge_init(batch.edge_attr.float())

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_info)

            x = self.norms[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        pool_fn = global_add_pool if self.gnn_type == "GMM" else global_max_pool
        molecule_emb = pool_fn(x, batch.batch)

        return self.final_proj(molecule_emb)

# ------------------------------------------------------------------------------------------------------------------

class Predictor(nn.Module):
    def __init__(self, embedding_dim, out_dim, predictor_type="Linear", hidden_dim=128):
        super().__init__()
        if predictor_type == "Linear":
            self.net = nn.Linear(embedding_dim, out_dim)
        elif predictor_type == "MLP":
            self.net = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, out_dim)
            )
        else:
            raise AttributeError("'predictor_type' must be Linear or MLP")

    def forward(self, x):
        return self.net(x)

# ------------------------------------------------------------------------------------------------------------------


class MoleculeNetClassificationModel(L.LightningModule):
    def __init__(self, in_channels, edge_dim, config, pos_weight):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.encoder = GNNEncoder(in_channels, edge_dim, config['hidden_dim'], config['embedding_dim'],
                                  config['gnn_type'], config['num_layers'], config['dropout_encoder'])

        self.predictor = Predictor(config['embedding_dim'], config['out_dim'], config['predictor_type'],
                                   config.get('mlp_hidden_dim'))

        # weights = torch.tensor([1.0, pos_weight.item()])
        self.register_buffer("class_weights", pos_weight)
        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

        self.metrics = torchmetrics.MetricCollection([
            torchmetrics.Accuracy(task="binary"),
            torchmetrics.Precision(task="binary"),
            torchmetrics.Recall(task="binary"),
            torchmetrics.F1Score(task="binary"),
            torchmetrics.AUROC(task="binary")
        ])

    def forward(self, batch):
        emb = self.encoder(batch)
        logits = self.predictor(emb)
        return logits, emb

    def training_step(self, batch, batch_idx):
        logits, _ = self(batch)
        y = batch.y.view(-1).long()

        loss = self.criterion(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch.num_graphs)
        self.training_step_outputs.append(loss)

        return loss

    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.log("training_loss_epoch", epoch_average)

        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        logits, _ = self(batch)
        y = batch.y.view(-1).long()

        loss = self.criterion(logits, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch.num_graphs)
        self.validation_step_outputs.append(loss)

        probs = torch.softmax(logits, dim=1)[:, 1]
        output = self.metrics(probs, y)
        self.log_dict(output, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch.num_graphs)

    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log("val_loss_epoch", epoch_average)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        logits, _ = self(batch)
        y = batch.y.view(-1).long()

        loss = self.criterion(logits, y)
        self.log("test_loss", loss, prog_bar=True, batch_size=batch.num_graphs)

        probs = torch.softmax(logits, dim=1)[:, 1]
        output = self.metrics(probs, y)
        self.log_dict(output, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# ------------------------------------------------------------------------------------------------------------------

class MoleculeNetRegressionModel(L.LightningModule):
    def __init__(self, in_channels, edge_dim, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.training_step_outputs = []
        self.validation_step_outputs = []

        self.encoder = GNNEncoder(in_channels, edge_dim, config['hidden_dim'], config['embedding_dim'],
                                  config['gnn_type'], config['num_layers'], config['dropout_encoder'])

        self.predictor = Predictor(config['embedding_dim'], config['out_dim'], config['predictor_type'],
                                   config.get('mlp_hidden_dim'))

        self.criterion = nn.MSELoss()
        self.mae = torchmetrics.MeanAbsoluteError()

    def forward(self, batch):
        emb = self.encoder(batch)
        out = self.predictor(emb)
        return out, emb

    def training_step(self, batch, batch_idx):
        preds, _ = self(batch)
        preds = preds.view(-1)
        y = batch.y.view(-1).float()

        loss = self.criterion(preds, y)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch.num_graphs)

        return loss

    def validation_step(self, batch, batch_idx):
        preds, _ = self(batch)
        preds = preds.view(-1)
        y = batch.y.view(-1).float()

        val_loss = self.criterion(preds, y)
        self.mae(preds, y)
        self.log_dict({"val_loss": val_loss, "val_mae": self.mae},
                      on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)

    def test_step(self, batch, batch_idx):
        preds, _ = self(batch)
        preds = preds.view(-1)
        y = batch.y.view(-1).float()

        val_loss = self.criterion(preds, y)
        self.mae(preds, y)
        self.log_dict({"test_loss": val_loss, "test_mae": self.mae},
                      on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


# ---------------------------------------------------------------------------------------------------------------
#
# class MoleculeSiameseLearningModel(L.LightningModule):
#     def __init__(self, in_channels, edge_dim, config):
#         super().__init__()
#         self.save_hyperparameters()
#         self.config = config
#
#         # Wykorzystujemy Twój istniejący GNNEncoder
#         self.encoder = GNNEncoder(
#             in_channels,
#             edge_dim,
#             config['hidden_dim'],
#             config['embedding_dim'],
#             config['gnn_type'],
#             config['num_layers'],
#             config['dropout_encoder']
#         )
#
#         self.loss_fn = losses.TripletMarginLoss(margin=config.get('margin', 0.1))
#         self.miner = miners.TripletMarginMiner(margin=config.get('margin', 0.1), type_of_triplets="semi-hard")
#
#     def forward(self, batch):
#         emb = self.encoder(batch)
#         return F.normalize(emb, p=2, dim=1)
#
#     def training_step(self, batch, batch_idx):
#         emb = self(batch)
#         y = batch.y.view(-1)
#
#         indices_tuple = self.miner(emb, y)
#         loss = self.loss_fn(emb, y, indices_tuple)
#
#         self.log("train_metric_loss", loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch.num_graphs)
#         return loss
#
#     def validation_step(self, batch, batch_idx):
#         emb = self(batch)
#         y = batch.y.view(-1)
#         indices_tuple = self.miner(emb, y)
#         loss = self.loss_fn(emb, y, indices_tuple)
#         self.log("val_metric_loss", loss, prog_bar=True, batch_size=batch.num_graphs)
#
#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=0.001)

class MoleculeSiameseLearningModel(L.LightningModule):
    def __init__(self, in_channels, edge_dim, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.encoder = GNNEncoder(
            in_channels,
            edge_dim,
            config['hidden_dim'],
            config['embedding_dim'],
            config['gnn_type'],
            config['num_layers'],
            config['dropout_encoder']
        )

        margin = config.get('margin', 0.2)
        self.loss_fn = losses.TripletMarginLoss(margin=margin)
        self.miner = miners.TripletMarginMiner(margin=margin, type_of_triplets="semi-hard")

        self.validation_step_outputs = []
        self.training_step_outputs = []

    def forward(self, batch):
        emb = self.encoder(batch)
        return F.normalize(emb, p=2, dim=1)

    def training_step(self, batch, batch_idx):
        emb = self(batch)
        y = batch.y.view(-1)

        indices_tuple = self.miner(emb, y)
        loss = self.loss_fn(emb, y, indices_tuple)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, batch_size=batch.num_graphs)

        self.training_step_outputs.append({
            "emb": emb.detach().cpu(),
            "y": y.detach().cpu()
        })
        return loss

    def validation_step(self, batch, batch_idx):
        emb = self(batch)
        y = batch.y.view(-1)

        self.validation_step_outputs.append({
            "emb": emb.detach().cpu(),
            "y": y.detach().cpu()
        })

        indices_tuple = self.miner(emb, y)
        loss = self.loss_fn(emb, y, indices_tuple)
        self.log("val_loss", loss, prog_bar=True, batch_size=batch.num_graphs)

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        all_emb = torch.cat([x["emb"] for x in self.validation_step_outputs]).cpu().numpy()
        all_y = torch.cat([x["y"] for x in self.validation_step_outputs]).cpu().numpy()

        knn = KNeighborsClassifier(n_neighbors=self.config.get('k_neighbors', 5))
        knn.fit(all_emb, all_y)
        preds = knn.predict(all_emb)
        acc = accuracy_score(all_y, preds)

        self.log("val_knn_acc", acc, prog_bar=True)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.get('lr', 0.001))
        return optimizer