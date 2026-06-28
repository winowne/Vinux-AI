import torch
from src.tokenizer import Vinux
from torch.utils.data import DataLoader

class VinuxDataset(torch.utils.data.Dataset):
    def __init__(self,tokenizer,max_length):
        self.max_length = max_length
        with open('data/data.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        self.tokens = tokenizer.encode(text)

    def __len__(self):
        return len(self.tokens) // self.max_length

    def __getitem__(self, idx):
        start_idx = idx * self.max_length
        end_idx = start_idx + self.max_length + 1
        chunk = self.tokens[start_idx:end_idx]
        x = chunk[:-1]
        y = chunk[1:]
        return torch.tensor(x),torch.tensor(y)

if __name__ == '__main__':
    tokenizer = Vinux()
    dataset = VinuxDataset(tokenizer,32)
    dataloader = DataLoader(dataset,batch_size=4,shuffle=True)
    x_batch, y_batch = next(iter(dataloader))
    print(x_batch.shape)
    print(y_batch.shape)