import torch
import torchvision
import torchmetrics
import pathlib
from utils.models import SimpleCNN
from utils.display import display_predictions

DATASET_PATH = pathlib.Path("dataset")

BATCH_SIZE = 32

### DATASET ###

test_dataset = torchvision.datasets.FashionMNIST(DATASET_PATH, train=False, download=True, transform=torchvision.transforms.ToTensor())

classes = test_dataset.classes

test_loader = torch.utils.data.DataLoader(test_dataset, BATCH_SIZE)

### EVALUATION ###

state_dict = torch.load("model.pth")

model = SimpleCNN()
model.load_state_dict(state_dict)
model.eval()

metric = torchmetrics.Accuracy(task="multiclass", num_classes=len(classes))

all_outputs = []

with torch.no_grad():
    for x, y in test_loader:
        o = model(x)

        metric.update(o, y)

        all_outputs.append(o)

    print(f"Test Set Accuracy: {metric.compute().item():.4f}")

    predictions = torch.cat(all_outputs, dim=0).softmax(dim=1).argmax(dim=1)

    display_predictions(test_dataset.data, predictions, classes, 5)