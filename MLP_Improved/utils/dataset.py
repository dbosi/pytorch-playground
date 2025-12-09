import numpy as np
import pathlib
import struct
import torch
import typing

def load_idx(filename: pathlib.Path, dtype:torch.dtype=torch.float32) -> torch.Tensor:
    with open(filename, 'rb') as f:
        magic = f.read(4)
        data_type = struct.unpack('>B', magic[2:3])[0]
        dims = struct.unpack('>B', magic[3:4])[0]
        shape = []
        for _ in range(dims):
            shape.append(struct.unpack('>I', f.read(4))[0])
        
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(shape)

    return torch.tensor(data, dtype=dtype)

def train_val_split(dataset: torch.Tensor, labels: torch.Tensor, train_percentage: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_examples = dataset.size(0)

    index_perms = torch.randperm(n_examples)

    n_train_examples = int(n_examples*train_percentage)

    train_indexes = index_perms[:n_train_examples]
    val_indexes = index_perms[n_train_examples:]

    return dataset[train_indexes], labels[train_indexes], dataset[val_indexes], labels[val_indexes]

def batch_loader(dataset: torch.Tensor, labels: torch.Tensor, batch_size: int) -> typing.Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
    for i in range(0, dataset.size(0), batch_size):
        yield dataset[i: i + batch_size,], labels[i: i + batch_size]
