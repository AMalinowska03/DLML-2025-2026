import math

import torch
import torchmetrics
from torch import nn

import lightning as L


class BaseLitModel(L.LightningModule):
    def __init__(self, vocab_size, lr=0.001):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.vocab_size = vocab_size
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=vocab_size)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=vocab_size)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if isinstance(logits, tuple): logits = logits[0]
        # Reshape dla Loss: (Batch * Seq, Vocab) vs (Batch * Seq)
        loss = self.loss_fn(logits.reshape(-1, self.vocab_size), y.reshape(-1))
        self.log("train_loss", loss, prog_bar=True)
        self.train_acc(logits.reshape(-1, self.vocab_size).argmax(dim=1), y.reshape(-1))
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        if isinstance(logits, tuple): logits = logits[0]
        loss = self.loss_fn(logits.reshape(-1, self.vocab_size), y.reshape(-1))
        self.log("val_loss", loss, prog_bar=True)
        self.val_acc(logits.reshape(-1, self.vocab_size).argmax(dim=1), y.reshape(-1))
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        return loss


class LSTMPredictor(BaseLitModel):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, lr):
        super().__init__(vocab_size, lr)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embeds = self.embedding(x)
        out, hidden = self.lstm(embeds, hidden)
        logits = self.fc(out)
        # Zwracamy logits. Jeśli potrzebujemy hidden (do generacji), musimy to obsłużyć osobno
        if self.training or hidden is None:
            return logits
        return logits, hidden


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (Batch, Seq_Len, Emb_Dim)
        return x + self.pe[:, :x.size(1), :]


class TransformerPredictor(BaseLitModel):
    def __init__(self, vocab_size, embedding_dim, nhead, num_layers, lr, max_len=500):
        super().__init__(vocab_size, lr)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_len)

        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embedding_dim, vocab_size)

    def generate_square_subsequent_mask(self, sz, device):
        # Maska kauzalna: pozwala patrzeć tylko na tokeny przeszłe
        # Zwraca macierz (sz, sz) z -inf powyżej przekątnej
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self, x):
        # x: (Batch, Seq_Len)
        seq_len = x.size(1)

        # 1. Embedding + Positional Encoding
        embeds = self.embedding(x) * math.sqrt(self.hparams.embedding_dim)
        embeds = self.pos_encoder(embeds)

        # 2. Maska atencji (Causal Mask)
        mask = self.generate_square_subsequent_mask(seq_len, x.device)

        # 3. Transformer Decoder
        # Ponieważ robimy Language Modeling (GPT-style), nie mamy pamięci z encodera.
        # W PyTorch TransformerDecoder: tgt=input, memory=input (lub None, zależnie od implementacji)
        # Tutaj używamy samego decodera bez cross-attention do encodera
        output = self.transformer_decoder(tgt=embeds, memory=embeds, tgt_mask=mask, memory_mask=mask)

        logits = self.fc(output)
        return logits