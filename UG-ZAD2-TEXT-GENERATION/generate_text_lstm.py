import torch
import torch.nn.functional as F

from dataset import TextDataModule
from models import LSTMPredictor


def generate_text_lstm(model, start_text, tokenizer, length=100, temperature=1.0, device='cpu'):
    model.eval()
    model.to(device)

    # 1. Przygotowanie promptu
    input_ids = tokenizer.encode(start_text)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)  # (1, Seq)

    generated_ids = list(input_ids)

    with torch.no_grad():
        # A. WARM-UP: Przepuszczamy cały prompt, aby zbudować stan ukryty (hidden state)
        # Model LSTM zwraca: output, (h_n, c_n)
        _, hidden = model(input_tensor)

        # Ostatni token promptu to nasz pierwszy input do pętli generującej
        last_token = input_tensor[:, -1].unsqueeze(1)  # (1, 1)

        # B. PĘTLA GENERACJI
        for _ in range(length):
            # Podajemy tylko OSTATNI token i STARY stan ukryty
            output, hidden = model(last_token, hidden)

            # output ma wymiar (1, 1, VocabSize)
            logits = output[0, -1, :] / temperature
            probs = F.softmax(logits, dim=0)

            # Losowanie
            next_token = torch.multinomial(probs, 1).item()
            generated_ids.append(next_token)

            # Aktualizacja wejścia na kolejny krok
            last_token = torch.tensor([[next_token]], device=device)

    return tokenizer.decode(generated_ids)


CONFIG = {
    'batch_size': 128,
    'seq_len': 100,
    'tokenizer_type': 'char',  # lub 'bpe'
    'data_sources': ['data/pantadeusz.txt'],
    'model_checkpoint': 'lightning_logs/lstm_v1',
}

if __name__ == "__main__":
    dm = TextDataModule(CONFIG)
    model = LSTMPredictor.load_from_checkpoint(CONFIG['model_checkpoint'])
    generated_text = generate_text_lstm(model, "Tadeusz", dm.tokenizer)
