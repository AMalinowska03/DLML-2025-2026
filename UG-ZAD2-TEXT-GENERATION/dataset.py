import lightning as L
import torch
from torch.utils.data import DataLoader, random_split, Dataset

from dataset_tokenizers import CharTokenizer, HFTokenizerWrapper, LiteHFTokenizer


class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len):
        self.seq_len = seq_len
        self.data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        print(f"Dataset utworzony. Ilość tokenów: {len(self.data)}")

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]  # Input, Target


class TextDataModule(L.LightningDataModule):
    def __init__(self, config, tokenizer_cls=None):
        super().__init__()
        self.cfg = config
        self.tokenizer_cls = tokenizer_cls
        self.tokenizer = None
        self.vocab_size = 0

    def setup(self, stage=None):
        full_text = ""
        for fname in self.cfg['data_sources']:
            with open(fname, 'r', encoding='utf-8') as f:
                full_text += f.read().lower() + "\n"

        if self.cfg['tokenizer_type'] == 'custom':
            self.tokenizer = LiteHFTokenizer(full_text)
        if self.cfg['tokenizer_type'] == 'bpe':
            self.tokenizer = HFTokenizerWrapper()
        elif self.cfg['tokenizer_type'] == 'char':
            self.tokenizer = CharTokenizer(full_text)

        self.vocab_size = self.tokenizer.vocab_size

        full_dataset = TextDataset(full_text, self.tokenizer, self.cfg['seq_len'])

        total_len = len(full_dataset)
        train_size = int(0.8 * total_len)
        val_size = int(0.1 * total_len)
        test_size = total_len - train_size - val_size

        self.train_set, self.val_set, self.test_set = random_split(
            full_dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)  # Ważne dla powtarzalności testów!
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.cfg['batch_size'],
            shuffle=True,
            num_workers=2,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.cfg['batch_size'],
            num_workers=2,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.cfg['batch_size'],
            num_workers=2,
            persistent_workers=True
        )