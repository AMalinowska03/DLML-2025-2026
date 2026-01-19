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
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode(self, tokens):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self.tokenizer.decode(tokens)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size