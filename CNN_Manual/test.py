import torch
import torchvision
import pathlib
import utils.display

DATASET_PATH = pathlib.Path("dataset")

POOL_SIZE = 2
POOL_STRIDE = 2

# DATASET #

test_dataset = torchvision.datasets.FashionMNIST(DATASET_PATH, train=False, download=True)

images, labels = test_dataset.data, test_dataset.targets
classes = test_dataset.classes

images = images / 255.0

# MODEL #

state_dict = torch.load("model.pth")

kernel_w1 = state_dict["kernel_w1"]
conv_b1 = state_dict["conv_b1"]

kernel_w2 = state_dict["kernel_w2"]
conv_b2 = state_dict["conv_b2"]

w1 = state_dict["w1"]
b1 = state_dict["b1"]

w2 = state_dict["w2"]
b2 = state_dict["b2"]

w3 = state_dict["w3"]
b3 = state_dict["b3"]

# FORWARD PASS #

with torch.no_grad():
    x, y = torch.unsqueeze(images, 1), labels

    # Convolution 1 #
    o = torch.conv2d(x, kernel_w1, conv_b1)
    o = torch.relu(o)
    o = torch.max_pool2d(o, POOL_SIZE, POOL_STRIDE)

    # utils.display.display_images(o[0], 5)
    
    # Convolution 2 #
    o = torch.conv2d(o, kernel_w2, conv_b2)
    o = torch.relu(o)
    o = torch.max_pool2d(o, POOL_SIZE, POOL_STRIDE)

    # utils.display.display_images(o[0], 5)

    # Flatten #
    o = o.flatten(1)

    # Dense 1 #
    o = torch.matmul(o, w1) + b1
    o = torch.relu(o)

    # Dense 2 #
    o = torch.matmul(o, w2) + b2
    o = torch.relu(o)

    # Dense 3 #
    o = torch.matmul(o, w3) + b3

    # Predictions #
    predictions = torch.softmax(o, dim=1).argmax(dim=1)
    accuracy = (predictions == y).sum() / y.size(0)

    print(f"Test Set Accuracy: {accuracy:.4f}")

    utils.display.display_predictions(images, predictions, 5, classes, rand=True, grayscale=True)