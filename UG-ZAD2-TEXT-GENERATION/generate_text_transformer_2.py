import torch
import torch.nn.functional as F

from dataset import TextDataModule
from models import TransformerPredictor


def generate_text_transformer(model, start_text, tokenizer, length=100, temperature=1.0, device='cpu', max_ctx_len=100):
    model.eval()
    model.to(device)

    input_ids = tokenizer.encode(start_text)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)  # (1, Seq)

    generated_ids = list(input_ids)

    with torch.no_grad():
        for _ in range(length):
            cond_idxs = input_tensor[:, -max_ctx_len:]

            logits = model(cond_idxs)

            next_token_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=0)

            next_token = torch.multinomial(probs, 1).item()
            generated_ids.append(next_token)

            input_tensor = torch.cat([input_tensor, torch.tensor([[next_token]], device=device)], dim=1)

    return tokenizer.decode(generated_ids)


CONFIG = {
    'batch_size': 128,
    'seq_len': 100,
    'vocab_size': 10000, # 5000
    'tokenizer_type': 'custom',  # lub 'bpe'
    'data_sources': ['data/pantadeusz.txt', 'data/quovadis.txt', 'data/potoptompierwszy.txt'],
    'model_checkpoint': 'lightning_logs/transformer_2_v1_10k_vocab_part_2/checkpoints/epoch=81-step=305286.ckpt',
    # 'model_checkpoint': 'lightning_logs/transformer_2_v1_5k_vocab/checkpoints/epoch=66-step=280060.ckpt',
}

if __name__ == "__main__":
    dm = TextDataModule(CONFIG)
    dm.setup()
    model = TransformerPredictor.load_from_checkpoint(CONFIG['model_checkpoint'])
    generated_text = generate_text_transformer(model, "Litwo", dm.tokenizer, max_ctx_len=CONFIG['seq_len'])
    print(generated_text)
