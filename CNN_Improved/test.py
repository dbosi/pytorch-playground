import torch
import torchvision
import pathlib
import utils.display


MODEL_NAME = pathlib.Path("model.pth")

KERNEL_OUT_C_1 = 32
KERNEL_OUT_C_2 = 64
KERNEL_OUT_C_3 = 128

POOL_SIZE = 2
POOL_STRIDE = 2

EPSILON = 1e-8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


### DATASET ###

DATASET_PATH = pathlib.Path("dataset")

test_dataset = torchvision.datasets.CIFAR10(DATASET_PATH, train=False, download=True)

x_test, y_test = torch.Tensor(test_dataset.data).to(torch.float32).to(device), torch.Tensor(test_dataset.targets).to(torch.long).to(device)
classes = list(test_dataset.classes)

x_test = x_test.permute(0, 3, 1, 2)

channels_mean = x_test.mean(dim=(0, 2, 3)) / 255.0
channels_std = x_test.std(dim=(0, 2, 3)) / 255.0

x_test = x_test / 255.0
x_test = (x_test - channels_mean.view(1, 3, 1, 1)) / channels_std.view(1, 3, 1, 1)


### MODEL ###

if MODEL_NAME.is_file():
    state_dict = torch.load(MODEL_NAME)
else:
    raise FileNotFoundError(f"No model named '{str(MODEL_NAME)}' found! Use train.py to train the model!")

kernel_w1 = state_dict["kernel_w1"].to(device)
bn_gamma_1 = state_dict["bn_gamma_1"].to(device)
bn_beta_1 = state_dict["bn_beta_1"].to(device)
running_mean_1 = state_dict["running_mean_1"].to(device)
running_var_1 = state_dict["running_var_1"].to(device)

kernel_w2 = state_dict["kernel_w2"].to(device)
bn_gamma_2 = state_dict["bn_gamma_2"].to(device)
bn_beta_2 = state_dict["bn_beta_2"].to(device)
running_mean_2 = state_dict["running_mean_2"].to(device)
running_var_2 = state_dict["running_var_2"].to(device)

kernel_w3 = state_dict["kernel_w3"].to(device)
bn_gamma_3 = state_dict["bn_gamma_3"].to(device)
bn_beta_3 = state_dict["bn_beta_3"].to(device)
running_mean_3 = state_dict["running_mean_3"].to(device)
running_var_3 = state_dict["running_var_3"].to(device)

dense_w1 = state_dict["dense_w1"].to(device)
bn_gamma_4 = state_dict["bn_gamma_4"].to(device)
bn_beta_4 = state_dict["bn_beta_4"].to(device)
running_mean_4 = state_dict["running_mean_4"].to(device)
running_var_4 = state_dict["running_var_4"].to(device)

dense_w2 = state_dict["dense_w2"].to(device)
bn_gamma_5 = state_dict["bn_gamma_5"].to(device)
bn_beta_5 = state_dict["bn_beta_5"].to(device)
running_mean_5 = state_dict["running_mean_5"].to(device)
running_var_5 = state_dict["running_var_5"].to(device)

dense_w3 = state_dict["dense_w3"].to(device)
bn_gamma_6 = state_dict["bn_gamma_6"].to(device)
bn_beta_6 = state_dict["bn_beta_6"].to(device)
running_mean_6 = state_dict["running_mean_6"].to(device)
running_var_6 = state_dict["running_var_6"].to(device)

dense_w4 = state_dict["dense_w4"].to(device)
bn_gamma_7 = state_dict["bn_gamma_7"].to(device)
bn_beta_7 = state_dict["bn_beta_7"].to(device)
running_mean_7 = state_dict["running_mean_7"].to(device)
running_var_7 = state_dict["running_var_7"].to(device)

dense_w5 = state_dict["dense_w5"].to(device)
dense_b5 = state_dict["dense_b5"].to(device)


### INFERENCE ###

with torch.no_grad():
    o = torch.nn.functional.pad(x_test, (1, 1, 1, 1), value=0)
    o = torch.conv2d(o, kernel_w1)

    o = (o - running_mean_1.view(1, KERNEL_OUT_C_1, 1, 1)) / torch.sqrt(running_var_1.view(1, KERNEL_OUT_C_1, 1, 1) + EPSILON)
    o = bn_gamma_1.view(1, KERNEL_OUT_C_1, 1, 1) * o + bn_beta_1.view(1, KERNEL_OUT_C_1, 1, 1)
    
    o = torch.relu(o)
    o = torch.max_pool2d(o, POOL_SIZE, POOL_STRIDE)
    

    o = torch.nn.functional.pad(o, (1, 1, 1, 1), value=0)
    o = torch.conv2d(o, kernel_w2)

    o = (o - running_mean_2.view(1, KERNEL_OUT_C_2, 1, 1)) / torch.sqrt(running_var_2.view(1, KERNEL_OUT_C_2, 1, 1) + EPSILON)
    o = bn_gamma_2.view(1, KERNEL_OUT_C_2, 1, 1) * o + bn_beta_2.view(1, KERNEL_OUT_C_2, 1, 1)

    o = torch.relu(o)
    o = torch.max_pool2d(o, POOL_SIZE, POOL_STRIDE)
    

    o = torch.nn.functional.pad(o, (1, 1, 1, 1), value=0)
    o = torch.conv2d(o, kernel_w3)

    o = (o - running_mean_3.view(1, KERNEL_OUT_C_3, 1, 1)) / torch.sqrt(running_var_3.view(1, KERNEL_OUT_C_3, 1, 1) + EPSILON)
    o = bn_gamma_3.view(1, KERNEL_OUT_C_3, 1, 1) * o + bn_beta_3.view(1, KERNEL_OUT_C_3, 1, 1)

    o = torch.relu(o)
    o = torch.max_pool2d(o, POOL_SIZE, POOL_STRIDE)


    o = o.flatten(1)


    o = torch.matmul(o, dense_w1)

    o = (o - running_mean_4) / torch.sqrt(running_var_4 + EPSILON)
    o = bn_gamma_4 * o + bn_beta_4

    o = torch.relu(o)


    o = torch.matmul(o, dense_w2)

    o = (o - running_mean_5) / torch.sqrt(running_var_5 + EPSILON)
    o = bn_gamma_5 * o + bn_beta_5

    o = torch.relu(o)


    o = torch.matmul(o, dense_w3)

    o = (o - running_mean_6) / torch.sqrt(running_var_6 + EPSILON)
    o = bn_gamma_6 * o + bn_beta_6

    o = torch.relu(o)


    o = torch.matmul(o, dense_w4)

    o = (o - running_mean_7) / torch.sqrt(running_var_7 + EPSILON)
    o = bn_gamma_7 * o + bn_beta_7

    o = torch.relu(o)


    o = torch.matmul(o, dense_w5) + dense_b5

    predictions = torch.softmax(o, dim=1).argmax(dim=1)
    accuracy = (predictions == y_test).sum() / y_test.size(0)

    print(f"Test Set Accuracy: {accuracy.item():.4f}")

    utils.display.display_predictions(torch.Tensor(test_dataset.data).permute(0, 3, 1, 2) / 255.0, predictions, 5, classes, grayscale=False)