import torch
import typing

EMBEDDING_SIZE = 64

HIDDEN_DIM = 64

class ModelLSTM(torch.nn.Module):
    def __init__(self: typing.Self, vocab_size: int, padding_idx: int, n_classes: int) -> None:
        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, EMBEDDING_SIZE, padding_idx=padding_idx)

        self.lstm = torch.nn.LSTM(EMBEDDING_SIZE, HIDDEN_DIM, batch_first=True)

        self.fc = torch.nn.Linear(HIDDEN_DIM, n_classes)

    def forward(self: typing.Self, text: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths=lengths.cpu(), batch_first=True, enforce_sorted=False)

        out, (h, c) = self.lstm(packed)

        return self.fc(h[-1]).squeeze()