import torch
import typing
import matplotlib.pyplot as plt

def display_predictions(images: torch.Tensor, predictions: torch.Tensor, classes: typing.Union[torch.Tensor, None], n: int, rand: bool=True, gray: bool=True) -> None:
    fig, axes = plt.subplots(1, n, figsize=(15, 15))

    n_examples = predictions.size(0)

    if rand:
        rand_idxs = torch.randperm(n_examples)

        images = images[rand_idxs]
        predictions = predictions[rand_idxs]

    for i in range(n):
        axes[i].imshow(images[i], cmap="gray" if gray else "viridis")
        axes[i].set_title(str(predictions[i] if classes is None else classes[predictions[i]]))
        axes[i].axis("off")

    plt.show()