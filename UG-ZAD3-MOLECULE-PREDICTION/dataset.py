import lightning as L
import torch
from torch_geometric.datasets import MoleculeNet, QM9
from torch_geometric.loader import DataLoader


class BACEDataModule(L.LightningDataModule):
    def __init__(self, batch_size=64, root="data"):
        super().__init__()
        self.batch_size = batch_size
        self.root = root
        self.num_features = None
        self.num_classes = 2
        self.pos_weight = None

    def setup(self, stage=None):
        dataset = MoleculeNet(root=self.root, name="BACE")

        labels = torch.tensor([d.y.item() for d in dataset])
        num_pos = labels.sum()
        num_neg = len(labels) - num_pos
        self.pos_weight = torch.tensor([num_neg / num_pos])

        self.num_features = dataset.num_features

        total_len = len(dataset)
        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)
        test_len = total_len - train_len - val_len
        self.train_ds, self.val_ds, self.test_ds = torch.utils.data.random_split(
            dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=2)


class QM9DataModule(L.LightningDataModule):
    def __init__(self, batch_size=64, root="data"):
        super().__init__()
        self.batch_size = batch_size
        self.root = root
        self.num_features = None
        self.num_classes = 1

    def setup(self, stage=None):
        dataset = QM9(root=self.root)
        dataset.data.y = dataset.data.y[:, 0:1]  # QM9 target: bierzemy indeks 0 (moment dipolowy)

        self.num_features = dataset.num_features

        total_len = len(dataset)
        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)
        test_len = total_len - train_len - val_len

        self.train_ds, self.val_ds, self.test_ds = torch.utils.data.random_split(
            dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=2)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=2)
