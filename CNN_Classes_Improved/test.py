import torch
import torchvision
import torchmetrics
import pathlib
import utils.models
import utils.display

### CONSTANTS ###

CUDA_DEVICE = "cuda"
CPU_DEVICE = "cpu"

DATASET_PATH = pathlib.Path("dataset")

MODEL_NAME = pathlib.Path("model.pth")

BATCH_SIZE = 32


### VARIABLES ###

device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else CPU_DEVICE)


### DATASET ###

cifar10_test_dataset = torchvision.datasets.CIFAR10(DATASET_PATH, train=False, download=True)

all_test_examples = torch.Tensor(cifar10_test_dataset.data).to(torch.float32)
all_test_labels = torch.Tensor(cifar10_test_dataset.targets).to(torch.long)

classes = cifar10_test_dataset.classes

n_examples, img_h, img_w, n_channels = all_test_examples.shape

all_test_examples = all_test_examples / 255.0

channels_mean = torch.mean(all_test_examples, dim=(0, 1, 2))
channels_std = torch.std(all_test_examples, dim=(0, 1, 2))

all_test_examples = (all_test_examples - channels_mean.view(1, 1, 1, n_channels)) / channels_std.view(1, 1, 1, n_channels)
all_test_examples = all_test_examples.permute(0, 3, 1, 2)

test_set = torch.utils.data.TensorDataset(all_test_examples, all_test_labels)

test_set_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE)


### INFERENCE ###

model = utils.models.ImprovedCNN((n_channels, img_h, img_w), len(classes))

state_dict = torch.load(MODEL_NAME)

model.load_state_dict(state_dict)

model = model.to(device)

metric = torchmetrics.Accuracy(task="multiclass", num_classes=len(classes)).to(device)

with torch.no_grad():
    model.eval()

    all_predictions = []

    for x, y in test_set_loader:
        x = x.to(device)
        y = y.to(device)

        o = model(x)

        metric.update(o, y)

        all_predictions.append(o)

    print(f"Accuracy: {metric.compute().item():.4f}")

    predictions = torch.cat(all_predictions, dim=0).softmax(dim=1).argmax(dim=1)

    utils.display.display_predictions(torch.Tensor(cifar10_test_dataset.data) / 255.0, predictions, classes, 5, gray=False)