import numpy as np
import torch
import utils.dataset
import utils.models
import pathlib

DATASET_PATH = pathlib.Path("dataset")
TRAIN_DATA_FILE = "train-images.idx3-ubyte"
TRAIN_LABELS_FILE = "train-labels.idx1-ubyte"

TRAIN_DATA_PERCENTAGE = .9

BATCH = 32
EPOCHS = 10
LEARNING_RATE = 0.01

NEURONS_L1 = 100
NEURONS_L2 = 50
NEURONS_OUTPUT = 10


### PREPARE DATASET ###

images = utils.dataset.load_idx(DATASET_PATH / TRAIN_DATA_FILE, dtype=torch.float32)
labels = utils.dataset.load_idx(DATASET_PATH / TRAIN_LABELS_FILE, dtype=torch.long)

n_examples, img_width, img_height = images.shape

images = images / 255.0
images = images.flatten(1, 2)

dataset = torch.utils.data.TensorDataset(images, labels)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(n_examples * TRAIN_DATA_PERCENTAGE), int(n_examples - (n_examples * TRAIN_DATA_PERCENTAGE))])

loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH, shuffle=True)

### TRAINING ###

model = utils.models.ImprovedMLP(img_width*img_height, NEURONS_L1, NEURONS_L2, NEURONS_OUTPUT)

cce_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

for i in range(EPOCHS):
    for x, y in loader:
        o = model(x)

        loss = cce_loss(o, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    with torch.no_grad():
        x_val, y_val = val_dataset.dataset.tensors

        o = model(x_val)

        val_loss = cce_loss(o, y_val)

        predictions = torch.softmax(o, dim=1).argmax(dim=1)

        acc = (predictions == y_val).sum() / y_val.size(0)

        print("Epoch", i+1, "\tval_loss:", val_loss.item(), "\tval_acc:", acc.item())

torch.save(model.state_dict(), "model.pth")