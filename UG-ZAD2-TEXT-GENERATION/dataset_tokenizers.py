from abc import ABC, abstractmethod

import torch
from transformers import AutoTokenizer


class BaseTokenizer(ABC):
    @abstractmethod
    def encode(self, text):
        pass

    @abstractmethod
    def decode(self, tokens):
        pass

    @property
    @abstractmethod
    def vocab_size(self):
        pass


class CharTokenizer(BaseTokenizer):
    def __init__(self, text):
        self.chars = sorted(list(set(text.lower())))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.char_to_idx.get(c, 0) for c in text.lower() if c in self.char_to_idx]

    def decode(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return ''.join([self.idx_to_char.get(t, '') for t in tokens])

    @property
    def vocab_size(self):
        return len(self.chars)


class HFTokenizerWrapper(BaseTokenizer):
    def __init__(self, model_name='allegro/herbert-base-cased'):
        # Używamy gotowego tokenizera dla języka polskiego (np. HerBERT), ale uczymy model od zera
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size


class LiteHFTokenizer(BaseTokenizer):
    def __init__(self, texts, load_from_path=None, vocab_size=5000):
        if load_from_path is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(load_from_path)
        else:
            print(f"Tworzenie lekkiego tokenizera (baza: GPT-2, cel: {vocab_size} tokenów)...")

            # Używamy 'gpt2', bo świetnie radzi sobie z generacją i nie robi [UNK]
            base_tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=False)

            def batch_iterator():
                # Dzielimy tekst na kawałki po 1000 znaków, żeby nie zapchać RAMu
                for i in range(0, len(texts), 1000):
                    yield texts[i: i + 1000]

            # train_new_from_iterator automatycznie tworzy nowy słownik idealnie pod Twój tekst
            self.tokenizer = base_tokenizer.train_new_from_iterator(
                batch_iterator(),
                vocab_size=vocab_size
            )

            # Dodajemy token PAD, bo GPT-2 domyślnie go nie ma, a potrzebujemy do batchy
            self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"Tokenizer gotowy. Rozmiar: {self.tokenizer.vocab_size}")

    def encode(self, text):
        return self.tokenizer.encode(text)

    def decode(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size