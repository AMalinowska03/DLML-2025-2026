import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch_geometric.nn import GCNConv, TransformerConv, global_mean_pool


class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()