import torch
import matplotlib.pyplot as plt

def display_img_and_prediction(imgs: torch.Tensor, preds: torch.Tensor, n: int) -> None:
    fig, axes = plt.subplots(1, n, figsize=(12, 3))

    indices = torch.randperm(imgs.shape[0])
    selected_indices = indices[:n]

    for i, idx in enumerate(selected_indices):
        axes[i].imshow(imgs[idx].reshape(28, 28), cmap='gray')
        axes[i].set_title(f"Pred: {preds[idx]}")
        axes[i].axis('off')

    plt.show()