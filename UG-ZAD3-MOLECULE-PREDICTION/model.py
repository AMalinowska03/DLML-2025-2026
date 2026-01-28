import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch_geometric.nn import GCNConv, TransformerConv, global_mean_pool


class GNNEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, embedding_dim, gnn_type="GCN", num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.gnn_type = gnn_type

        current_dim = in_channels
        for _ in range(num_layers):
            if gnn_type == "GCN":
                self.layers.append(GCNConv(current_dim, hidden_dim))
            elif gnn_type == "Transformer":
                # TransformerConv wymaga edge_attr (opcjonalnie) i pos (opcjonalnie)
                self.layers.append(TransformerConv(current_dim, hidden_dim, heads=2, concat=False))
            current_dim = hidden_dim

        # Projekcja do przestrzeni embeddingu (np. 1D, 2D lub więcej)
        self.final_proj = nn.Linear(hidden_dim, embedding_dim)
        self.act = nn.ReLU()

    def forward(self, x, edge_index, batch):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = self.act(x)
            x = F.dropout(x, p=0.2, training=self.training)

        # Global Pooling (graf -> wektor)
        x = global_mean_pool(x, batch)

        embedding = self.final_proj(x)
        return embedding


class Predictor(nn.Module):
    def __init__(self, embedding_dim, out_dim, predictor_type="Linear", hidden_dim=None):
        super().__init__()
        if predictor_type == "Linear":
            self.net = nn.Linear(embedding_dim, out_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim)
            )

    def forward(self, x):
        return self.net(x)


class MoleculeNetClassificationModel(L.LightningModule):
    def __init__(self, in_channels, config, pos_weight):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.encoder = GNNEncoder(in_channels, config['hidden_dim'], config['embedding_dim'], config['gnn_type'])
        self.predictor = Predictor(config['embedding_dim'], config['out_dim'], config['predictor_type'], config.get('mlp_hidden_dim', None))

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.metric = torchmetrics.Accuracy(task="binary")

    def forward(self, batch):
        emb = self.encoder(batch.x.float(), batch.edge_index, batch.batch)
        out = self.predictor(emb)
        return out, emb

    def training_step(self, batch, batch_idx):
        preds, _ = self(batch)
        preds = preds.view(-1)
        y = batch.y.float().view(-1)

        loss = self.criterion(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, _ = self(batch)
        preds = preds.view(-1)
        y = batch.y.float().view(-1)

        val_loss = self.criterion(preds, y)
        self.log("val_loss", val_loss, prog_bar=True)

        self.metric(torch.sigmoid(preds), y.int())
        self.log("val_metric", self.metric, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class MoleculeNetRegressionModel(L.LightningModule):
    def __init__(self, in_channels, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.encoder = GNNEncoder(in_channels, config['hidden_dim'], config['embedding_dim'], config['gnn_type'])
        self.predictor = Predictor(config['embedding_dim'], config['out_dim'], config['predictor_type'], config.get('mlp_hidden_dim', None))

        self.criterion = nn.MSELoss()
        self.metric = torchmetrics.MeanAbsoluteError()

    def forward(self, batch):
        emb = self.encoder(batch.x.float(), batch.edge_index, batch.batch)
        out = self.predictor(emb)
        return out, emb

    def training_step(self, batch, batch_idx):
        preds, _ = self(batch)
        preds = preds.view(-1)
        y = batch.y.float().view(-1)

        loss = self.criterion(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, _ = self(batch)
        preds = preds.view(-1)
        y = batch.y.float().view(-1)

        val_loss = self.criterion(preds, y)
        self.log("val_loss", val_loss, prog_bar=True)

        self.metric(preds, y)
        self.log("val_metric", self.metric, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
