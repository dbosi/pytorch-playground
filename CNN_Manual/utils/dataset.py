import torch
import typing

def train_val_split(data: torch.Tensor, labels: torch.Tensor, train_split: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_examples = data.size(0)

    rand_idxs = torch.randperm(n_examples)

    train_examples = int(n_examples * train_split)

    train_idxs, val_idxs = rand_idxs[:train_examples], rand_idxs[train_examples:]

    return data[train_idxs], labels[train_idxs], data[val_idxs], labels[val_idxs]

def batch_loader(data: torch.Tensor, labels: torch.Tensor, batch_size: int) -> typing.Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    n_examples = data.size(0)

    for i in range(0, n_examples, batch_size):
        yield data[i: i + batch_size], labels[i: i + batch_size]