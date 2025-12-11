import torch
import pathlib
import utils.dataset
import utils.models
import utils.display

DATASET_PATH = pathlib.Path("dataset")
TEST_DATA_FILE = "t10k-images.idx3-ubyte"
TEST_LABELS_FILE = "t10k-labels.idx1-ubyte"

MODEL_PATH = "model.pth"

NEURONS_L1 = 100
NEURONS_L2 = 50
NEURONS_OUTPUT = 10

### PREPARE DATASET ###

images = utils.dataset.load_idx(DATASET_PATH / TEST_DATA_FILE, dtype=torch.float32)
labels = utils.dataset.load_idx(DATASET_PATH / TEST_LABELS_FILE, dtype=torch.long)

images = images.flatten(1, 2)

n_examples, n_features = images.shape

model = utils.models.ImprovedMLP(n_features, NEURONS_L1, NEURONS_L2, NEURONS_OUTPUT)

state_dict = torch.load("model.pth")
model.load_state_dict(state_dict)

model.eval()

with torch.no_grad():
    o = model(images)

    predictions = torch.softmax(o, dim=1).argmax(dim=1)

    acc = (predictions == labels).sum() / n_examples

    print("Accuracy:", acc.item())

    utils.display.display_img_and_prediction(images, predictions, 5)