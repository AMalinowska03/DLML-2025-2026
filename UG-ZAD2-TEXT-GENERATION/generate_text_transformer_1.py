import torch
import torch.nn.functional as F

from dataset import TextDataModule
from models import TransformerPredictor


def generate_text_transformer(model, start_text, tokenizer, length=100, temperature=1.0, device='cpu', max_ctx_len=100):
    model.eval()
    model.to(device)

    input_ids = tokenizer.encode(start_text)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)  # (1, Seq)

    # Lista wynikowa
    generated_ids = list(input_ids)

    with torch.no_grad():
        for _ in range(length):
            # A. PRZYCIĘCIE OKNA (Sliding Window)
            # Transformer nie może przyjąć więcej niż max_ctx_len (długość, na której był trenowany),
            # bo nie ma wytrenowanych embeddingów pozycyjnych dla dalszych pozycji.
            cond_idxs = input_tensor[:, -max_ctx_len:]

            # B. FORWARD PASS
            # Podajemy całą sekwencję (okno kontekstowe)
            logits = model(cond_idxs)

            # C. WYBÓR
            # Interesuje nas tylko predykcja dla ostatniego elementu sekwencji
            # logits shape: (1, Seq_Len, Vocab_Size) -> bierzemy [0, -1, :]
            next_token_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=0)

            next_token = torch.multinomial(probs, 1).item()
            generated_ids.append(next_token)

            # D. AKTUALIZACJA
            # Doklejamy nowy token do tenora wejściowego
            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=device)], dim=1)

    return tokenizer.decode(generated_ids)


CONFIG = {
    'batch_size': 128,
    'seq_len': 100,
    'tokenizer_type': 'char',  # lub 'bpe'
    'data_sources': ['data/pantadeusz.txt'],
    'model_checkpoint': 'lightning_logs/transformer_1_v1',
}

if __name__ == "__main__":
    dm = TextDataModule(CONFIG)
    model = TransformerPredictor.load_from_checkpoint(CONFIG['model_checkpoint'])
    generated_text = generate_text_transformer(model, "Tadeusz", dm.tokenizer, max_ctx_len=CONFIG['seq_len'])
