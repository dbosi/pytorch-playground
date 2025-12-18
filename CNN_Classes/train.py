import numpy as np
import torch
import torchvision
import torchmetrics
import pathlib
from utils.models import SimpleCNN


DATASET_PATH = pathlib.Path("dataset")

TRAIN_DATA_PERCENTAGE = 0.9

BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001

### DATASET ###

train_dataset = torchvision.datasets.FashionMNIST(DATASET_PATH, train=True, download=True, transform=torchvision.transforms.ToTensor())

classes = train_dataset.classes
n_total = len(train_dataset)

train_set, val_set = torch.utils.data.random_split(train_dataset, [int(n_total * TRAIN_DATA_PERCENTAGE), int(n_total - (n_total * TRAIN_DATA_PERCENTAGE))])

train_loader = torch.utils.data.DataLoader(train_set, BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, BATCH_SIZE)

### TRAINING ###

model = SimpleCNN()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
metrics = torchmetrics.Accuracy(task="multiclass", num_classes=len(classes))

for i in range(EPOCHS):
    print(f"Epoch {i+1}")

    model.train()

    train_loss = 0.0
    metrics.reset()

    for j, (x, y) in enumerate(train_loader):
        o = model(x)

        batch_loss = criterion(o, y)
        train_loss += batch_loss.item()

        metrics.update(o, y)

        optimizer.zero_grad()

        batch_loss.backward()

        optimizer.step()

        print(f"\rtrain_loss: {(train_loss / (j+1)):.4f} \tBatch: {j+1}/{len(train_loader)}", end="")

    print(f"\rtrain_loss: {(train_loss / (j+1)):.4f} \ttrain_acc: {metrics.compute().item():.4f}", end="")

    with torch.no_grad():
        model.eval()

        val_loss = 0.0
        metrics.reset()

        for x, y in val_loader:
            o = model(x)

            val_loss += criterion(o, y).item()

            metrics.update(o, y)

        print(f"\nval_loss:   {(val_loss / len(val_loader)):.4f} \tval_acc:   {metrics.compute().item():.4f}\n")

torch.save(model.state_dict(), "model.pth")