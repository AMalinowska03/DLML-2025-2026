import lightning as L
import torch
import torchmetrics
from torch import nn


class LightningModel(L.LightningModule):
    def __init__(self, model, pos_weight):
        super().__init__()
        self.model = model
        self.loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.acc = torchmetrics.classification.BinaryAccuracy()

        for p in self.model.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(1)
        loss = self.loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(1)
        loss = self.loss(logits, y)
        predictions = torch.sigmoid(logits)
        acc = self.acc(predictions, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze(1)
        predictions = torch.sigmoid(logits)
        acc = self.acc(predictions, y)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
