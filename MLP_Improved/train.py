import numpy as np
import torch
import utils.dataset
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

x_train, y_train, x_val, y_val = utils.dataset.train_val_split(images, labels, TRAIN_DATA_PERCENTAGE)

x_train, x_val = x_train.flatten(1,2), x_val.flatten(1,2)

### TRAINING ###

w1 = torch.randn(img_width * img_height, NEURONS_L1) * (2 / (img_width * img_height)) ** 0.5 
b1 = torch.zeros(1, NEURONS_L1)

w2 = torch.randn(NEURONS_L1, NEURONS_L2) * (2 / NEURONS_L1) ** 0.5
b2 = torch.zeros(1, NEURONS_L2)

w3 = torch.randn(NEURONS_L2, NEURONS_OUTPUT) * (2 / NEURONS_L2) ** 0.5
b3 = torch.zeros(1, NEURONS_OUTPUT)

parameters = [w1, b1, w2, b2, w3, b3]

for p in parameters:
    p.requires_grad = True

loss_function = torch.nn.CrossEntropyLoss()

for i in range(EPOCHS):
    for batch in utils.dataset.batch_loader(x_train, y_train, BATCH):
        x, y = batch

        z1 = torch.matmul(x, w1) + b1
        a1 = torch.relu(z1)

        z2 = torch.matmul(a1, w2) + b2
        a2 = torch.relu(z2)

        z3 = torch.matmul(a2, w3) + b3
        
        loss = loss_function(z3, y)

        for p in parameters:
            p.grad = None

        loss.backward()

        for p in parameters:
            p.data -= LEARNING_RATE * p.grad

    with torch.no_grad():
        z1 = torch.matmul(x_val, w1) + b1
        a1 = torch.relu(z1)

        z2 = torch.matmul(a1, w2) + b2
        a2 = torch.relu(z2)

        z3 = torch.matmul(a2, w3) + b3

        val_loss = loss_function(z3, y_val)

        predictions = torch.softmax(z3, dim=1).argmax(dim=1)
        acc = (predictions == y_val).sum() / y_val.size(0)

        print("Epoch", i+1, "\tval_loss:", val_loss.item(), "\tval_acc:", acc.item())

    
model_params = {
    "w1": w1,
    "b1": b1,
    "w2": w2,
    "b2": b2,
    "w3": w3,
    "b3": b3
}

torch.save(model_params, "model.pth")