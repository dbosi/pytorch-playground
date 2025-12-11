import numpy as np
import pathlib
import struct
import torch

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