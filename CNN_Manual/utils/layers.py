import torch

def dropout(x: torch.Tensor, p: float) -> torch.Tensor:
    if p == 0.0:
        return x
    
    mask = (torch.rand_like(x) > p).to(torch.float32)
    mask = mask / (1.0 - p)

    return x * mask