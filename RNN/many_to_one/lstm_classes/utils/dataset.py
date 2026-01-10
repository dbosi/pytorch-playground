import torch

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        super().__init__()

        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        return self.texts[index], self.labels[index]