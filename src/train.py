import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.tokenizer import Vinux
from src.model import VinuxLanguageModel

sys.stdout.reconfigure(encoding='utf-8')

DATA_PATH = "data.txt"
MODEL_SAVE_PATH = "models/model.pt"
BATCH_SIZE = 64
BLOCK_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 3e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"[Vinux-AI] Обучение будет запущено на устройстве: {DEVICE}")

tokenizer = Vinux()
VOCAB_SIZE = len(tokenizer.vocab)
print(f"[Vinux-AI] Новый вокабуляр успешно загружен. Размер словаря: {VOCAB_SIZE} токенов")


class LazyTextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.lines = []

        print("[Загрузка] Индексируем строки датасета...")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.lines.append(line.strip())
        print(f"[Успешно] Индексация завершена. Всего строк: {len(self.lines)}")

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]

        token_ids = [self.tokenizer.vocab.get(char, 0) for char in line]

        if len(token_ids) <= self.block_size:
            token_ids = token_ids + [0] * (self.block_size + 1 - len(token_ids))
        else:
            token_ids = token_ids[:self.block_size + 1]

        x = torch.tensor(token_ids[:self.block_size], dtype=torch.long)
        y = torch.tensor(token_ids[1:self.block_size + 1], dtype=torch.long)
        return x, y


dataset = LazyTextDataset(DATA_PATH, tokenizer, BLOCK_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

model = VinuxLanguageModel(vocab_size=VOCAB_SIZE).to(DEVICE)

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

print("\n[Vinux-AI] Начинаем обучение с чистого листа...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0

    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

        logits, loss = model(x_batch, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if batch_idx % 100 == 0 and batch_idx > 0:
            avg_loss = total_loss / (batch_idx + 1)
            print(f"Эпоха [{epoch}/{EPOCHS}] | Шаг {batch_idx}/{len(dataloader)} | Текущий Loss: {avg_loss:.4f}")

    epoch_loss = total_loss / len(dataloader)
    print(f"\n[Эпоха {epoch} Завершена!] Средний Loss: {epoch_loss:.4f}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"[Сохранение] Веса модели сохранены в {MODEL_SAVE_PATH}\n")

print("[Vinux-AI] Обучение полностью успешно завершено!")