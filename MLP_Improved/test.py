import torch
import pathlib
import utils.dataset
import utils.display

DATASET_PATH = pathlib.Path("dataset")
MODEL_PATH = "model.pth"

TEST_DATA_FILE = "t10k-images.idx3-ubyte"
TEST_LABELS_FILE = "t10k-labels.idx1-ubyte"


loaded_params = torch.load(MODEL_PATH)

w1 = loaded_params["w1"]
b1 = loaded_params["b1"]
w2 = loaded_params["w2"]
b2 = loaded_params["b2"]
w3 = loaded_params["w3"]
b3 = loaded_params["b3"]


dataset = utils.dataset.load_idx(DATASET_PATH / TEST_DATA_FILE, dtype=torch.float32)
dataset = dataset.flatten(1, 2)

labels = utils.dataset.load_idx(DATASET_PATH / TEST_LABELS_FILE, dtype=torch.long)

with torch.no_grad():
    z1 = torch.matmul(dataset, w1) + b1
    a1 = torch.relu(z1)
    z2 = torch.matmul(a1, w2) + b2
    a2 = torch.relu(z2)
    z3 = torch.matmul(a2, w3) + b3

    predictions = torch.softmax(z3, dim=1).argmax(dim=1)
    acc = (predictions == labels).sum() / labels.size(0)

    print("Test Set Accuracy:", acc)

    utils.display.display_img_and_prediction(dataset, predictions, 5)
