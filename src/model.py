import torch
import torch.nn as nn
import torch.nn.functional as F

class VinuxLanguageModel(nn.Module):
    def __init__(self, vocab_size):  # Добавили vocab_size сюда
        super().__init__()
        self.table = nn.Embedding(vocab_size, 128)  # Заменили 5000 на vocab_size
        self.lm_head = nn.Linear(128, vocab_size)  # Заменили 5000 на vocab_size
        self.query = nn.Linear(128, 128)
        self.key = nn.Linear(128, 128)
        self.value = nn.Linear(128, 128)


    def forward(self, idx,targets=None):
        x = self.table(idx)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        matrix_linkage = q @ k.transpose(-2, -1)
        meaning = torch.softmax(matrix_linkage, dim =-1)
        out = meaning @ v
        logits = self.lm_head(out)
        if targets is None:
            loss = None

        else:
            B,T,C = logits.shape
            flat_logits = logits.view(B*T,C)
            flat_targets = targets.view(B*T)
            loss = F.cross_entropy(flat_logits, flat_targets)

        return logits, loss

if __name__ == '__main__':
    model = VinuxLanguageModel(vocab_size=5000)  # Передаем тестовое число сюда
    targets = None
    x = torch.randint(0, 5000, (4, 32))