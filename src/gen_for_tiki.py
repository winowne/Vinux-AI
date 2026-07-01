import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BLOCK_SIZE = 64
N_EMBED = 384
N_HEAD = 6
N_LAYER = 6
VOCAB_SIZE = 30000


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBED, N_EMBED)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class VinuxLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks = nn.Sequential(*[Block(N_EMBED, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, None


vocab_path = 'vocabs/tigr_vocab.json'
weights_path = 'models/tigr.pt'

with open(vocab_path, 'r', encoding='utf-8') as f:
    kaggle_vocab = json.load(f)

vocab = {str(k): int(v) for k, v in kaggle_vocab.items()}
id_to_word = {int(v): str(k) for k, v in kaggle_vocab.items()}

special_tokens = [("<pad>", 0), ("<unk>", 1), ("<start>", 2), ("<transition>", 3), ("<end>", 4)]
for token, idx in special_tokens:
    vocab[token] = idx
    id_to_word[idx] = token

model = VinuxLanguageModel(vocab_size=VOCAB_SIZE)
weights = torch.load(weights_path, map_location=device)

model_dict = model.state_dict()
weights = {k: v for k, v in weights.items() if k in model_dict and model_dict[k].shape == v.shape}
model.load_state_dict(weights, strict=True)
model = model.to(device)
model.eval()


def clean_and_format_text(text):
    bad_tokens = ["<unk>", "<transition>", "<start>", "<end>", "[pad]", "[unk]", "<pad>"]
    for token in bad_tokens:
        text = text.replace(token, "")
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    if text:
        text = text[0].upper() + text[1:]
    return text


def generate_response(user_text):
    input_ids = [vocab["<start>"]]
    for word in user_text.lower().split():
        input_ids.append(vocab.get(word, vocab["<unk>"]))
    input_ids.append(vocab["<transition>"])

    x = torch.tensor([input_ids], dtype=torch.long).to(device)
    response_words = []

    with torch.no_grad():
        for i in range(40):
            x_cond = x[:, -BLOCK_SIZE:]
            logits, _ = model(x_cond)
            next_token_logits = logits[:, -1, :]

            temp = next_token_logits / 0.8
            top_k = 10
            v, ix = torch.topk(temp, top_k)
            temp = torch.full_like(temp, float('-inf')).scatter_(-1, ix, v)
            probability = F.softmax(temp, dim=-1)
            next_token_id = torch.multinomial(probability, 1).item()

            if next_token_id == vocab["<end>"]:
                break
            if next_token_id == vocab["<pad>"] and i > 0:
                break

            next_token_tensor = torch.tensor([[next_token_id]], dtype=torch.long).to(device)
            x = torch.cat((x, next_token_tensor), dim=1)

            word = id_to_word.get(next_token_id, "<unk>")
            response_words.append(word)

    raw_output = " ".join(response_words)
    final_text = clean_and_format_text(raw_output)

    return final_text if final_text else "[Нейросеть задумалась]"


if __name__ == '__main__':
    while True:
        prompt = input("Введи текст (или 'exit'): ")
        if prompt.lower() == 'exit':
            break
        print(f">> {generate_response(prompt)}\n")