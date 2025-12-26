import torch
import torchvision
import torchmetrics
import pathlib
import utils.models
import kornia

### CONSTANTS ###

CUDA_DEVICE = "cuda"
CPU_DEVICE = "cpu"

DATASET_PATH = pathlib.Path("dataset")

MODEL_NAME = pathlib.Path("model.pth")

TRAIN_SET_PERCENTAGE = 0.8

BATCH_SIZE = 32

EPOCHS = 50
LEARNING_RATE = 0.001

### VARIABLES ###

device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else CPU_DEVICE)


### DATASET ###

cifar10_train_dataset = torchvision.datasets.CIFAR10(DATASET_PATH, train=True, download=True)

all_train_examples = torch.Tensor(cifar10_train_dataset.data).to(torch.float32)
all_train_targets = torch.Tensor(cifar10_train_dataset.targets).to(torch.long)

classes = cifar10_train_dataset.classes

n_examples, img_h, img_w, n_channels = all_train_examples.shape

all_train_examples = all_train_examples / 255.0

channels_mean = all_train_examples.mean(dim=(0, 1, 2))
channels_std = all_train_examples.std(dim=(0, 1, 2))

all_train_examples = all_train_examples.permute(0, 3, 1, 2)

dataset = torch.utils.data.TensorDataset(all_train_examples, all_train_targets)

n_train = int(TRAIN_SET_PERCENTAGE * len(cifar10_train_dataset))
n_val = len(cifar10_train_dataset) - n_train

train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE)

train_augmentation = kornia.augmentation.AugmentationSequential(
    kornia.augmentation.RandomHorizontalFlip(0.5),
    kornia.augmentation.RandomRotation(15),
    kornia.augmentation.ColorJitter(0.2, 0.2, 0.2, 0.1),
    kornia.augmentation.Normalize(mean=channels_mean, std=channels_std)
).to(device)

val_augmentation = kornia.augmentation.AugmentationSequential(
    kornia.augmentation.Normalize(mean=channels_mean, std=channels_std)
).to(device)

### MODEL ###

model = utils.models.ImprovedCNN(in_shape=(n_channels, img_h, img_w), n_classes=len(classes))
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
metric = torchmetrics.Accuracy(task="multiclass", num_classes=len(classes)).to(device)

for epoch in range(EPOCHS):
    print(f"Epoch: {epoch+1}")

    model.train()

    metric.reset()
    train_loss = 0.0

    for b, (x, y) in enumerate(train_loader):
        x = x.to(device)
        x = train_augmentation(x)
        y = y.to(device)

        o = model(x)

        loss: torch.Tensor = criterion(o, y)

        optimizer.zero_grad()

        loss.backward()

        train_loss += loss.item()

        optimizer.step()

        metric.update(o, y)

        print(f"\rtrain_loss: {(train_loss / (b+1)):.4f}\tBatch: {b+1}/{len(train_loader)}", end="")

    print(f"\rtrain_loss: {(train_loss / (b+1)):.4f}\ttrain_acc: {metric.compute().item():.4f}", end="")

    with torch.no_grad():
        model.eval()

        metric.reset()
        val_loss = 0.0

        for x, y in val_loader:
            x = x.to(device)
            x = val_augmentation(x)
            y = y.to(device)

            o = model(x)

            val_loss += criterion(o, y).item()

            metric.update(o, y)

        print(f"\nval_loss: {(val_loss / len(val_loader)):.4f}\tval_acc: {metric.compute().item():.4f}\n")

state_dict = model.state_dict()

torch.save(state_dict, MODEL_NAME)