import torch
import torch.nn.functional as F
from src.tokenizer import Vinux
from src.model import VinuxLanguageModel

tokenizer = Vinux()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VinuxLanguageModel(vocab_size=len(tokenizer.vocab))
weights = torch.load('models/model.pt',map_location=device)
model = model.to(device)
model.load_state_dict(weights)
model.eval()
inv_vocab = {v: k for k, v in tokenizer.vocab.items()}


print("[Vinux-AI] Скрипт готов к работе!")

while True:
    answer = input('Введите начальный промпт >>> ')
    if answer in ['exit', 'выход']:
        break

    else:
        with torch.no_grad():
            for i in range(50):
                input_ids = [tokenizer.vocab.get(char, 0) for char in answer]
                cond_tokens = input_ids[-32:]
                x = torch.tensor([cond_tokens], dtype=torch.long).to(device)
                x_cond = x[:, -32:]
                logits, _ = model(x_cond)
                next_token_logits = logits[:, -1, :]
                temp = next_token_logits / 0.85
                probability = F.softmax(temp, dim=-1)
                next_token_id = torch.multinomial(probability, 1).item()
                next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
                x = torch.cat((x, next_token_tensor), dim=1)
                token = inv_vocab.get(next_token_id, "")
                print(token, end="", flush=True)
