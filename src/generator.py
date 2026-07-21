import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_EXPERIMENTAL_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
import json
import string
from dataclasses import dataclass
from transformers import GPT2Tokenizer

@dataclass
class Config:
    block_size: int = 256
    vocab_size: int = 50257
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.1
    bias: bool = False

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        bias_val = getattr(cfg, 'bias', False)
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=bias_val)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=bias_val)
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
                                        .view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        bias_val = getattr(cfg, 'bias', False)
        self.c_fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=bias_val)
        self.c_proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=bias_val)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(cfg.vocab_size, cfg.n_embd),
            wpe=nn.Embedding(cfg.block_size, cfg.n_embd),
            drop=nn.Dropout(cfg.dropout),
            h=nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)]),
            ln_f=nn.LayerNorm(cfg.n_embd),
        ))
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(0, T, device=idx.device, dtype=torch.long)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = logits[:, [-1], :]
            loss = None
            
        return logits, loss

class LightningGPT(pl.LightningModule):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or Config()
        self.gpt = GPT(self.cfg)
        self.save_hyperparameters()

def setup_model(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    import src.generator as gen
    torch.serialization.add_safe_globals([Config, gen.Config])
    
    model_lightning = LightningGPT.load_from_checkpoint(
        checkpoint_path,
        weights_only=False
    )
    model = model_lightning.gpt
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def has_repetition(text, threshold=4):
    words = text.split()
    if len(words) < threshold:
        return False
    for i in range(len(words) - threshold + 1):
        if len(set(words[i:i+threshold])) == 1:
            return True
    return False

def is_meaningful(text):
    if len(text.strip()) < 2:
        return False
    if text.strip() in ['.', ',', '!', '?', '...', '', '..', '??']:
        return False
    return True


def generate_answer(model, tokenizer, device, context_ids, max_tokens=50, temp=0.5, temp_end=0.3, top_k=40, repetition_penalty=1.2, length_penalty=-0.5):
    generated_tokens = []
    input_ids = context_ids.clone()
    
    with torch.no_grad():
        for i in range(max_tokens):
            idx_cond = input_ids if input_ids.size(1) <= model.cfg.block_size else input_ids[:, -model.cfg.block_size:]
            
            temp_i = temp + (temp_end - temp) * (i / max_tokens)
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temp_i
            
            logits = logits + length_penalty * (i / max_tokens)
            
            for token in set(generated_tokens):
                if logits[0, token] > 0:
                    logits[0, token] /= repetition_penalty
                else:
                    logits[0, token] *= repetition_penalty

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            
            if i == 0:
                probs[0, tokenizer.eos_token_id] = 0.0
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                    
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
            token_id = next_token.item()
            generated_tokens.append(token_id)

            if token_id == tokenizer.eos_token_id:
                break
                
    return generated_tokens

model = None
tokenizer = None
device = None
history_ids = None
is_loaded = False
total_input_tokens = 0
total_output_tokens = 0
load_progress = 0

def reset_token_counts():
    global total_input_tokens, total_output_tokens
    total_input_tokens = 0
    total_output_tokens = 0

def load_model_background():
    global model, tokenizer, device, history_ids, is_loaded, load_progress
    if is_loaded:
        load_progress = 100
        return

    load_progress = 5
    CHECKPOINT_PATH = "models/checkpoint.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    load_progress = 10
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    load_progress = 30

    import src.generator as gen
    torch.serialization.add_safe_globals([Config, gen.Config])
    load_progress = 35

    model_lightning = LightningGPT.load_from_checkpoint(
        CHECKPOINT_PATH,
        weights_only=False
    )
    load_progress = 80

    model = model_lightning.gpt
    model.to(device)
    load_progress = 90
    model.eval()
    load_progress = 95

    history_ids = torch.tensor([[]], dtype=torch.long, device=device)
    is_loaded = True
    load_progress = 100

def generate_response(user_text: str) -> str:
    global history_ids, total_input_tokens, total_output_tokens

    while not is_loaded:
        time.sleep(0.1)

    if user_text.lower() == 'clear':
        history_ids = torch.tensor([[]], dtype=torch.long, device=device)
        total_input_tokens = 0
        total_output_tokens = 0
        return "[System] Контекст общения полностью очищен."

    new_user_ids = tokenizer.encode(user_text + "\n", return_tensors="pt").to(device)
    question_tokens = new_user_ids.size(1)
    total_input_tokens += question_tokens
    history_ids = torch.cat((history_ids, new_user_ids), dim=1)
    
    if history_ids.size(1) > model.cfg.block_size:
        history_ids = history_ids[:, -model.cfg.block_size:]
    
    if question_tokens <= 10:
        temp = 0.3
    elif question_tokens <= 15:
        temp = 0.4
    elif question_tokens <= 20:
        temp = 0.5
    elif question_tokens <= 22:
        temp = 0.6
    else:
        temp = 0.7
        
    top_k = int(max(10, 100 - (question_tokens / model.cfg.block_size) * 90))
    
    max_tokens = min(80, question_tokens + 50)
    
    if temp < 0.4:
        length_penalty = -0.7
    elif temp > 0.6:
        length_penalty = -0.3
    else:
        length_penalty = -0.5

    candidates = []
    candidates_raw = []
    t0 = time.time()
    
    for _ in range(3):
        bot_tokens = generate_answer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            context_ids=history_ids,
            max_tokens=max_tokens,
            temp=temp,
            temp_end=0.3,
            top_k=top_k,
            repetition_penalty=1.2,
            length_penalty=length_penalty
        )
        resp_str = tokenizer.decode(bot_tokens, skip_special_tokens=True).strip()
        resp_str = resp_str.lstrip(string.punctuation)
        candidates.append((bot_tokens, resp_str))
        candidates_raw.append(resp_str)

    t1 = time.time()
    
    filtered_candidates = [c for c in candidates if is_meaningful(c[1]) and not has_repetition(c[1])]
    if not filtered_candidates:
        filtered_candidates = candidates if candidates else [(torch.tensor([tokenizer.eos_token_id], device=device), "")]
        
    bot_tokens, response = max(filtered_candidates, key=lambda x: len(x[0]))
    
    new_bot_ids = torch.tensor([bot_tokens], dtype=torch.long, device=device)
    total_output_tokens += len(bot_tokens)
    history_ids = torch.cat((history_ids, new_bot_ids), dim=1)
    
    log_entry = {
        "question": user_text,
        "answer": response,
        "candidates": candidates_raw,
        "temperature": round(temp, 2),
        "top_k": top_k,
        "length_penalty": length_penalty,
        "thinking_time_s": round(t1 - t0, 2),
        "response_tokens": len(bot_tokens),
        "question_tokens": question_tokens,
    }
    with open("chat_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        
    return response