import torch
import typing

def train_val_split(x: torch.Tensor, y: torch.Tensor, train_split: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_examples = x.size(0)

    n_train = int(train_split * n_examples)

    rand_idxs = torch.randperm(n_examples)

    return x[rand_idxs[:n_train]], y[rand_idxs[:n_train]], x[rand_idxs[n_train:]], y[rand_idxs[n_train:]]


def batch_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int, device: torch.device) -> typing.Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    n_examples = x.size(0)

    rand_idxs = torch.randperm(n_examples)

    x, y = x[rand_idxs], y[rand_idxs]

    for i in range(0, n_examples, batch_size):
        yield x[i:i+batch_size], y[i:i+batch_size]