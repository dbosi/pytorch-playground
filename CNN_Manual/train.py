import numpy as np
import torch
import torchvision
import utils.dataset
import utils.layers
import utils.convolution
import pathlib

DATASET_PATH = pathlib.Path("dataset")

TRAIN_DATA_PERCENTAGE = .9

BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.01

INPUT_SIZE = 800
NEURONS_L1 = 128
NEURONS_L2 = 64
NEURONS_OUTPUT = 10

CHANNELS_1 = 16
CHANNELS_2 = 32

KERNEL_STRIDE = 1
KERNEL_SIZE = 3

POOL_SIZE = 2
POOL_STRIDE = 2

DROPOUT_RATE = 0.5

### DATASET ###

train_dataset = torchvision.datasets.FashionMNIST(DATASET_PATH, train=True, download=True)

images, labels = train_dataset.data, train_dataset.targets
classes = train_dataset.classes

n_total, img_h, img_w = images.shape

images = images / 255.0

x_train, y_train, x_val, y_val = utils.dataset.train_val_split(images, labels, TRAIN_DATA_PERCENTAGE)

### TRAINING ###

kernel_w1 = torch.randn(CHANNELS_1, 1, KERNEL_SIZE, KERNEL_SIZE) * (2 / (KERNEL_SIZE * KERNEL_SIZE)) ** 0.5
conv_b1 = torch.zeros(CHANNELS_1, dtype=torch.float32)

kernel_w2 = torch.randn(CHANNELS_2, CHANNELS_1, KERNEL_SIZE, KERNEL_SIZE) * (2 / (CHANNELS_1 * KERNEL_SIZE * KERNEL_SIZE)) ** 0.5
conv_b2 = torch.zeros(CHANNELS_2, dtype=torch.float32)

w1 = torch.randn(INPUT_SIZE, NEURONS_L1) * (2 / INPUT_SIZE) ** 0.5
b1 = torch.zeros(1, NEURONS_L1, dtype=torch.float32)

w2 = torch.randn(NEURONS_L1, NEURONS_L2) * (2 / NEURONS_L1) ** 0.5
b2 = torch.zeros(1, NEURONS_L2, dtype=torch.float32)

w3 = torch.randn(NEURONS_L2, NEURONS_OUTPUT)
b3 = torch.zeros(1, NEURONS_OUTPUT, dtype=torch.float32)

parameters = [kernel_w1, kernel_w2, conv_b1, w1, b1, w2, b2, w3, b3]

criterion = torch.nn.CrossEntropyLoss()

for p in parameters:
    p.requires_grad = True

for i in range(EPOCHS):
    for j, batch in enumerate(utils.dataset.batch_loader(x_train, y_train, BATCH_SIZE)):
        x, y = batch

        x = x.unsqueeze(1)

        # Convolution 1 #
        o = torch.nn.functional.conv2d(x, kernel_w1, conv_b1)
        o = torch.relu(o)
        o = torch.nn.functional.max_pool2d(o, POOL_SIZE, POOL_STRIDE)

        # Convolution 2 #
        o = torch.nn.functional.conv2d(o, kernel_w2, conv_b2)
        o = torch.relu(o)
        o = torch.nn.functional.max_pool2d(o, POOL_SIZE, POOL_STRIDE)

        # Flatten #
        o = o.flatten(1)

        # Dense 1 #
        o = torch.matmul(o, w1) + b1
        o = torch.relu(o)
        
        # Dropout 1 #
        o = utils.layers.dropout(o, DROPOUT_RATE)
        
        # Dense 2 #
        o = torch.matmul(o, w2) + b2
        o = torch.relu(o)

        # Dropout 2 #
        o = utils.layers.dropout(o, DROPOUT_RATE)

        # Output Layer #
        o = torch.matmul(o, w3) + b3

        # Backpropagation #

        train_loss = criterion(o, y)

        for p in parameters:
            p.grad = None

        train_loss.backward()

        for p in parameters:
            p.data -= LEARNING_RATE * p.grad
        
        print(f"\rEpoch {i+1}, Batch {j}/{len(x_train)//BATCH_SIZE}, batch_loss: {train_loss.item():.4f}", end="")

    with torch.no_grad():
        x, y = x_val.unsqueeze(1), y_val

        # Convolution 1 #
        o = torch.nn.functional.conv2d(x, kernel_w1, conv_b1)        
        o = torch.relu(o)
        o = torch.nn.functional.max_pool2d(o, POOL_SIZE, POOL_STRIDE)

        # Convolution 2 #
        o = torch.nn.functional.conv2d(o, kernel_w2, conv_b2)
        o = torch.relu(o)
        o = torch.nn.functional.max_pool2d(o, POOL_SIZE, POOL_STRIDE)

        # Flatten #
        o = o.flatten(1)

        # Dense 1 #
        o = torch.matmul(o, w1) + b1
        o = torch.relu(o)
        
        # Dense 2 #
        o = torch.matmul(o, w2) + b2
        o = torch.relu(o)

        # Output Layer #
        o = torch.matmul(o, w3) + b3

        val_loss = criterion(o, y)

        predictions = torch.softmax(o, dim=1).argmax(dim=1)
        acc = (predictions == y).sum() / y.size(0)

        print(f"\nEpoch {i+1}, val_loss: {val_loss.item():.4f} \tval_acc: {acc.item():.4f}\n")

state_dict = {
    "kernel_w1": kernel_w1,
    "conv_b1": conv_b1,
    "kernel_w2": kernel_w2,
    "conv_b2": conv_b2,
    "w1": w1,
    "b1": b1,
    "w2": w2,
    "b2": b2,
    "w3": w3,
    "b3": b3
}

torch.save(state_dict, "model.pth")