import torch
import torchvision
import torchmetrics
import pathlib
import utils.models
import utils.display

### CONSTANTS ###

CPU_DEVICE = "cpu"
CUDA_DEVICE = "cuda"

DATASET_PATH = pathlib.Path("dataset")
MODEL_NAME = pathlib.Path("model.pth")
INFERENCE_VALUES = pathlib.Path("inference_values.pth")

BATCH_SIZE = 32


### VARIABLES ###

device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else CPU_DEVICE)

model_state_dict = torch.load(MODEL_NAME)
inference_values = torch.load(INFERENCE_VALUES)

channel_mean: torch.Tensor = inference_values["channel_mean"]
channel_std: torch.Tensor = inference_values["channel_std"]


### DATASET ###

test_set = torchvision.datasets.CIFAR10(DATASET_PATH, train=False, download=True)

test_x = torch.tensor(test_set.data, dtype=torch.float32)
test_y = torch.tensor(test_set.targets, dtype=torch.float32)
classes = test_set.classes

test_x = test_x.permute(0, 3, 1, 2)

n_examples, n_channels, h_pixels, w_pixels = test_x.shape

test_x = test_x / 255.0
test_x = (test_x - channel_mean.view(1, n_channels, 1, 1)) / channel_std.view(1, n_channels, 1, 1)

test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)


### MODEL ###

model = utils.models.ResidualNetwork(in_channels=n_channels, n_classes=len(classes)).to(device)
model.load_state_dict(model_state_dict)

accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=len(classes)).to(device)

with torch.no_grad():
    model.eval()

    all_predictions = []

    for x, y in test_loader:
        x: torch.Tensor
        y: torch.Tensor

        x = x.to(device)
        y = y.to(device)

        o = model(x)

        accuracy.update(o, y)
        all_predictions.append(o)

    print(f"Test Set Accuracy: {accuracy.compute().item():.4f}")

    predictions = torch.cat(all_predictions, dim=0).softmax(dim=1).argmax(dim=1)

    utils.display.display_predictions(torch.tensor(test_set.data) / 255.0, predictions, classes, 5, gray=False)