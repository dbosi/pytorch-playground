import torch

def dropout(x: torch.Tensor, dropout_rate: float) -> torch.Tensor:
    if dropout_rate == 0:
        return x
    
    mask = (torch.rand_like(x) > dropout_rate).to(torch.float32)

    mask = mask / (1 - dropout_rate)

    return x * mask