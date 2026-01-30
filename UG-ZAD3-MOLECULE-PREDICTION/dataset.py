import lightning as L
import torch
from torch_geometric.datasets import MoleculeNet, QM9
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T


class BACEDataModule(L.LightningDataModule):
    def __init__(self, batch_size=64, root="data"):
        super().__init__()
        self.batch_size = batch_size
        self.root = root
        self.num_features = None
        self.num_edge_features = None
        self.num_classes = 2
        self.pos_weight = None

    def setup(self, stage=None):
        dataset = MoleculeNet(root=self.root, name="BACE")

        # Stały podział danych
        total_len = len(dataset)
        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)
        test_len = total_len - train_len - val_len

        self.train_ds, self.val_ds, self.test_ds = torch.utils.data.random_split(
            dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42)
        )

        # Obliczanie wag tylko dla zbioru treningowego
        train_labels = torch.tensor([data.y.item() for data in self.train_ds])
        num_pos = train_labels.sum().float()
        num_neg = len(train_labels) - num_pos

        # Wagi odwrotnie proporcjonalne do liczności klas
        # Klasa rzadsza otrzymuje większą wagę
        weight_neg = len(train_labels) / num_neg
        weight_pos = len(train_labels) / num_pos

        # Normalizacja wag, aby ich suma była równa liczbie klas (opcjonalne, ale zalecane)
        self.pos_weight = torch.tensor([weight_neg, weight_pos])

        self.num_features = dataset.num_features
        self.num_edge_features = dataset.num_edge_features

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=2, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=2, persistent_workers=True)


class QM9DataModule(L.LightningDataModule):
    def __init__(self, batch_size=64, root="data"):
        super().__init__()
        self.batch_size = batch_size
        self.root = root
        self.num_features = None
        self.num_edge_features = None
        self.num_classes = 1

    def setup(self, stage=None):
        transform = T.Compose([
            T.Cartesian(norm=True, cat=True)
        ])

        dataset = QM9(root=self.root, transform=transform)

        dataset.data.y = dataset.data.y[:, 0:1]  # QM9 target: bierzemy indeks 0 (moment dipolowy)

        self.num_features = dataset.num_features
        self.num_edge_features = dataset.num_edge_features

        total_len = len(dataset)
        train_len = int(0.8 * total_len)
        val_len = int(0.1 * total_len)
        test_len = total_len - train_len - val_len

        self.train_ds, self.val_ds, self.test_ds = torch.utils.data.random_split(
            dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=2, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=2, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=2, persistent_workers=True)
