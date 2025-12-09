import numpy as np
import struct
import pathlib
import torch
import matplotlib.pyplot as plt

from utils.SimpleMLP import SimpleMLP

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

test_images = torch.Tensor(load_idx(DATASET_PATH / "t10k-images.idx3-ubyte"))
test_images = test_images.reshape(test_images.shape[0], -1)
test_labels = torch.Tensor(load_idx(DATASET_PATH / "t10k-labels.idx1-ubyte"))

n_examples = test_images.shape[0]
n_features = test_images.shape[1]

state_dict = torch.load("model.pth")

model = SimpleMLP(n_features, 32, 10)
model.load_state_dict(state_dict)
model.eval()

with torch.no_grad():
    output = model(test_images)
    predictions = torch.softmax(output, dim=1)

correct_predictions = (predictions.argmax(dim=1) == test_labels).sum().item()
accuracy = correct_predictions / test_labels.shape[0]

print("Accuracy:", accuracy)

display_img_and_prediction(test_images, predictions, 5)