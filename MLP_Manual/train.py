import numpy as np
import struct
import pathlib
import torch

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

images = load_idx(DATASET_PATH / "train-images.idx3-ubyte")
images = images.reshape(images.shape[0], -1)
images = torch.tensor(images, dtype=torch.float32)

labels = load_idx(DATASET_PATH / "train-labels.idx1-ubyte")

n_examples = images.shape[0]
n_features = images.shape[1]

epoch = 35
learning_rate = 1

images = images / 255

w1 = torch.randn(n_features, 32) * 0.1
b1 = torch.randn(1, 32)
w2 = torch.randn(32, 10) * 0.1
b2 = torch.randn(1, 10)

parameters = [w1, b1, w2, b2]

for p in parameters:
    p.requires_grad = True

for i in range(epoch):
    z1 = torch.matmul(images, w1) + b1
    a1 = torch.maximum(z1, torch.tensor(0.0))

    z2 = torch.matmul(a1, w2) + b2
    z2 = z2 - torch.max(z2, dim=1, keepdim=True)[0]
    exp_z2 = torch.exp(z2)
    smax_z2 = exp_z2 / torch.sum(exp_z2, dim=1, keepdim=True)

    log_probs = torch.log(smax_z2)

    one_hot_encoded = torch.nn.functional.one_hot(torch.tensor(labels, dtype=torch.long), num_classes=10)

    cat_cross_entropy_loss = -torch.sum(log_probs * one_hot_encoded, dim=1).mean()

    print("epoch:", i+1, "\tloss:", cat_cross_entropy_loss.item())

    for p in parameters:
        p.grad = None

    cat_cross_entropy_loss.backward()

    for p in parameters:
        p.data -= learning_rate * p.grad

model_params = {
    "w1": w1,
    "b1": b1,
    "w2": w2,
    "b2": b2
}

torch.save(model_params, "model.pth")