import numpy as np
import struct
import pathlib
import torch
import matplotlib.pyplot as plt

DATASET_PATH = pathlib.Path("dataset")

def load_idx(filename: pathlib.Path):
    with open(filename, 'rb') as f:
        magic = f.read(4)
        data_type = struct.unpack('>B', magic[2:3])[0]
        dims = struct.unpack('>B', magic[3:4])[0]
        shape = []
        for _ in range(dims):
            shape.append(struct.unpack('>I', f.read(4))[0])
        
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(shape)

    return data

def display_img_and_prediction(imgs, preds, n):
    fig, axes = plt.subplots(1, n, figsize=(12, 3))

    indices = torch.randperm(imgs.shape[0])
    selected_indices = indices[:n]

    for i, idx in enumerate(selected_indices):
        axes[i].imshow(imgs[idx].reshape(28, 28), cmap='gray')
        axes[i].set_title(f"Pred: {preds[idx].argmax()}")
        axes[i].axis('off')

    plt.show()

loaded_params = torch.load("model.pth")

w1 = loaded_params["w1"]
b1 = loaded_params["b1"]
w2 = loaded_params["w2"]
b2 = loaded_params["b2"]

test_images = torch.Tensor(load_idx(DATASET_PATH / "t10k-images.idx3-ubyte"))
test_images = test_images.reshape(test_images.shape[0], -1)
test_labels = torch.Tensor(load_idx(DATASET_PATH / "t10k-labels.idx1-ubyte"))

with torch.no_grad():
    z1 = torch.matmul(test_images, w1) + b1
    a1 = torch.maximum(z1, torch.tensor(0.0))

    z2 = torch.matmul(a1, w2) + b2
    z2 = z2 - torch.max(z2, dim=1, keepdim=True)[0]
    exp_z2 = torch.exp(z2)
    smax_z2 = exp_z2 / torch.sum(exp_z2, dim=1, keepdim=True)

correct_predictions = (smax_z2.argmax(dim=1) == test_labels).sum().item()
accuracy = correct_predictions / test_labels.shape[0]

print("Accuracy:", accuracy)

display_img_and_prediction(test_images, smax_z2, 5)