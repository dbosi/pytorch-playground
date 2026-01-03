import torch
import torchvision
import torchmetrics
import pathlib
import kornia
import utils.models


### CONSTANTS ###

CPU_DEVICE = "cpu"
CUDA_DEVICE = "cuda"

DATASET_FOLDER = pathlib.Path("dataset")
MODEL_NAME = pathlib.Path("model.pth")
INFERENCE_VALUES = pathlib.Path("inference_values.pth")


TRAIN_SPLIT_PERCENTAGE = .8
BATCH_SIZE = 128


LEARNING_RATE = 0.001
EPOCHS = 200

### VARIABLES ###

device = torch.device(CUDA_DEVICE if torch.cuda.is_available() else CPU_DEVICE)


### DATASET ###

train_dataset = torchvision.datasets.CIFAR10(DATASET_FOLDER, train=True, download=True)

dataset_x = torch.tensor(train_dataset.data, dtype=torch.float32)
dataset_y = torch.tensor(train_dataset.targets, dtype=torch.long)

classes = train_dataset.classes


dataset_x = dataset_x.permute(0, 3, 1, 2)

dataset_x = dataset_x / 255.0

channels_values = {
    "channel_mean": dataset_x.mean(dim=(0, 2, 3)),
    "channel_std":  dataset_x.std(dim=(0, 2, 3))
}


n_examples, n_channels, h_pixels, w_pixels = dataset_x.shape

tensor_dataset = torch.utils.data.TensorDataset(dataset_x, dataset_y)

n_train = int(n_examples * TRAIN_SPLIT_PERCENTAGE)
n_val = n_examples - n_train

train_split, val_split = torch.utils.data.random_split(tensor_dataset, [n_train, n_val])

train_loader = torch.utils.data.DataLoader(train_split, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_split, batch_size=BATCH_SIZE)


train_augmentation = kornia.augmentation.AugmentationSequential(
    kornia.augmentation.RandomCrop((32, 32), padding=4),
    kornia.augmentation.RandomHorizontalFlip(0.5),
    kornia.augmentation.ColorJitter(0.2, 0.2, 0.2, 0.1),
    kornia.augmentation.Normalize(mean=channels_values["channel_mean"], std=channels_values["channel_std"])
).to(device)

val_augmentation = kornia.augmentation.AugmentationSequential(
    kornia.augmentation.Normalize(mean=channels_values["channel_mean"], std=channels_values["channel_std"])
).to(device)


### MODEL ###

model = utils.models.ResidualNetwork(n_channels, len(classes)).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=len(classes)).to(device)
mean_loss = torchmetrics.MeanMetric().to(device)

best_acc = 0


### TRAINING ###

for epoch in range(EPOCHS):
    print(f"Epoch: {epoch+1}")

    model.train()
    accuracy.reset()
    mean_loss.reset()

    for b, (x, y) in enumerate(train_loader):
        x: torch.Tensor
        y: torch.Tensor
        
        x = x.to(device)
        x = train_augmentation(x)
        y = y.to(device)

        o = model(x)

        batch_loss: torch.Tensor = criterion(o, y)
        mean_loss.update(batch_loss)

        optimizer.zero_grad()

        batch_loss.backward()

        optimizer.step()

        accuracy.update(o, y)

        print(f"\rtrain_loss: {mean_loss.compute().item():.4f}\tBatch: {b+1}/{len(train_loader)}", end="")

    print(f"\rtrain_loss: {mean_loss.compute().item():.4f}\ttrain_acc: {accuracy.compute().item():.4f}", end="")

    with torch.no_grad():
        model.eval()
        accuracy.reset()
        mean_loss.reset()

        for x, y in val_loader:
            x: torch.Tensor
            y: torch.Tensor

            x = x.to(device)
            x = val_augmentation(x)
            y = y.to(device)

            o = model(x)

            batch_loss = criterion(o, y)
            mean_loss.update(batch_loss)

            accuracy.update(o, y)

        print(f"\nval_loss: {mean_loss.compute().item():.4f}\tval_acc: {accuracy.compute().item():.4f}\n")

        if accuracy.compute().item() > best_acc:
            best_acc = accuracy.compute().item()

            print(f"Saved Model with {accuracy.compute().item():.4f} accuracy!\n")
            
            state_dict = model.state_dict()
            torch.save(state_dict, MODEL_NAME)

torch.save(channels_values, INFERENCE_VALUES)