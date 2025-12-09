import numpy as np
import struct
import pathlib
import torch

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

images = load_idx(DATASET_PATH / "train-images.idx3-ubyte")
images = images.reshape(images.shape[0], -1)
images = torch.tensor(images, dtype=torch.float32)

images = images / 255.0

labels = torch.tensor(load_idx(DATASET_PATH / "train-labels.idx1-ubyte"), dtype=torch.long)

n_examples = images.shape[0]
n_features = images.shape[1]

epoch = 50
learning_rate = 1

model = SimpleMLP(n_features, 32, 10)
cross_ent_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for i in range(epoch):
    o = model(images)
    
    loss = cross_ent_loss(o, labels)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    print("epoch:", i+1, "\tloss:", loss.item())

torch.save(model.state_dict(), "model.pth")