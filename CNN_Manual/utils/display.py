import torch
import matplotlib.pyplot as plt
import typing

def display_images(images: torch.Tensor, n: int, rand=False, grayscale=False) -> None:
    fig, axes = plt.subplots(1, n)

    if rand:
        rand_idxs = torch.randperm(images.size(0))
        images = images[rand_idxs]

    for i in range(n):
        axes[i].imshow(images[i], cmap="gray" if grayscale else "viridis")
        axes[i].axis("off")

    plt.show()

def display_predictions(images: torch.Tensor, preds: torch.Tensor, n: int, classes: typing.Union[torch.Tensor, None], rand=False, grayscale=False) -> None:
    fix, axes = plt.subplots(1, n, figsize=(15, 15))

    if rand:
        rand_idxs = torch.randperm(images.size(0))

        images = images[rand_idxs]
        preds = preds[rand_idxs]

    for i in range(n):
        axes[i].imshow(images[i], cmap="gray" if grayscale else "viridis")
        axes[i].set_title(f"Pred: {preds[i] if classes == None else classes[preds[i]]}")
        axes[i].axis("off")

    plt.tight_layout(pad=4.0)
    plt.show()