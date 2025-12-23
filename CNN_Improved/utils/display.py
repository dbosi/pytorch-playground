import torch
import typing
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from matplotlib.axes import Axes

def display_predictions(images: torch.Tensor, predictions: typing.Optional[torch.Tensor]=None, n: int=5, classes: typing.Optional[list]=None, rand: bool=True, grayscale: bool=True) -> None:
    _, axes = plt.subplots(1, n, figsize=(15, 15))

    axes: list[Axes]

    n_examples = images.size(0)

    images = images.permute(0, 2, 3, 1)

    if rand:
        rand_idxs = torch.randperm(n_examples)
        
        images = images[rand_idxs]
        
        if predictions is not None:
            predictions = predictions[rand_idxs]

    for i in range(n):
        axes[i].imshow(images[i].to(torch.cpu.current_device()), cmap="gray" if grayscale else None)
        axes[i].set_title(classes[predictions[i]] if predictions is not None else i+1)
        axes[i].axis("off")

    plt.show()