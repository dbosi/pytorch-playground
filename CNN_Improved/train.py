import torch
import torchvision
import pathlib
import utils.dataset
import utils.layers


MODEL_NAME = pathlib.Path("model.pth")
DATASET_PATH = pathlib.Path("dataset")

TRAIN_SPLIT = 0.8


# CONVOLUTION CONSTS #

KERNEL_OUT_C_1 = 32
KERNEL_SIZE_1 = 3

KERNEL_IN_C_2 = KERNEL_OUT_C_1
KERNEL_OUT_C_2 = 64
KERNEL_SIZE_2 = 3

KERNEL_IN_C_3 = KERNEL_OUT_C_2
KERNEL_OUT_C_3 = 128
KERNEL_SIZE_3 = 3

PADDING_SIZE = 1

POOL_SIZE = 2
POOL_STRIDE = POOL_SIZE

# DENSE CONSTS #

NEURONS_L1 = 512
NEURONS_L2 = 256
NEURONS_L3 = 128
NEURONS_L4 = 64

# TRAINING CONSTS #

BATCH_SIZE = 32

EPOCHS = 75

DROPOUT_RATE = 0.3

LEARNING_RATE = 0.001
BETA_1 = 0.9
BETA_2 = 0.999 
EPSILON = 1e-8

BN_MOMENTUM = 0.1

### DATASET ###

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = torchvision.datasets.CIFAR10(DATASET_PATH, train=True, download=True)

data, targets = torch.Tensor(train_dataset.data).to(torch.float32), torch.Tensor(train_dataset.targets).to(torch.long)
data = data.permute(0, 3, 1, 2)

classes = train_dataset.classes

n_total, n_channels, img_h, img_w = data.shape

channels_mean = data.mean(dim=(0, 2, 3)) / 255.0
channels_std = data.std(dim=(0, 2, 3)) / 255.0

x_train, y_train, x_val, y_val = utils.dataset.train_val_split(data, targets, TRAIN_SPLIT)

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(lambda x: x / 255.0),
    torchvision.transforms.RandomHorizontalFlip(0.5),
    torchvision.transforms.RandomRotation(15),
    torchvision.transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    torchvision.transforms.Normalize(mean=channels_mean, std=channels_std)
])

val_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(lambda x: x / 255.0),
    torchvision.transforms.Normalize(mean=channels_mean, std=channels_std)
])


### LAYERS ###

# CONVOLUTIONS PARAMETERS #

kernel_w1 = torch.randn(KERNEL_OUT_C_1, n_channels, KERNEL_SIZE_1, KERNEL_SIZE_1, device=device) * (2 / (n_channels * KERNEL_SIZE_1 * KERNEL_SIZE_1)) ** 0.5

kernel_w2 = torch.randn(KERNEL_OUT_C_2, KERNEL_IN_C_2, KERNEL_SIZE_2, KERNEL_SIZE_2, device=device) * (2 / (KERNEL_IN_C_2 * KERNEL_SIZE_2 * KERNEL_SIZE_2)) ** 0.5

kernel_w3 = torch.randn(KERNEL_OUT_C_3, KERNEL_IN_C_3, KERNEL_SIZE_3, KERNEL_SIZE_3, device=device) * (2 / (KERNEL_IN_C_3 * KERNEL_SIZE_3 * KERNEL_SIZE_3)) ** 0.5


conv_1_o_height = ((img_h + 2 * PADDING_SIZE - KERNEL_SIZE_1 + 1) - POOL_SIZE) // POOL_STRIDE + 1
conv_1_o_width = ((img_w + 2 * PADDING_SIZE - KERNEL_SIZE_1 + 1) - POOL_SIZE) // POOL_STRIDE + 1

conv_2_o_height = ((conv_1_o_height + 2 * PADDING_SIZE - KERNEL_SIZE_2 + 1) - POOL_SIZE) // POOL_STRIDE + 1
conv_2_o_width = ((conv_1_o_width + 2 * PADDING_SIZE - KERNEL_SIZE_2 + 1) - POOL_SIZE) // POOL_STRIDE + 1

conv_3_o_height = ((conv_2_o_height + 2 * PADDING_SIZE - KERNEL_SIZE_3 + 1) - POOL_SIZE) // POOL_STRIDE + 1
conv_3_o_width = ((conv_2_o_width + 2 * PADDING_SIZE - KERNEL_SIZE_3 + 1) - POOL_SIZE) // POOL_STRIDE + 1

dense_input_size = KERNEL_OUT_C_3 * conv_3_o_height * conv_3_o_width

# DENSE PARAMETERS #

dense_w1 = torch.randn(dense_input_size, NEURONS_L1, device=device) * (2 / dense_input_size) ** 0.5

dense_w2 = torch.randn(NEURONS_L1, NEURONS_L2, device=device) * (2 / NEURONS_L1) ** 0.5

dense_w3 = torch.randn(NEURONS_L2, NEURONS_L3, device=device) * (2 / NEURONS_L2) ** 0.5

dense_w4 = torch.randn(NEURONS_L3, NEURONS_L4, device=device) * (2 / NEURONS_L3) ** 0.5

dense_w5 = torch.randn(NEURONS_L4, len(classes), device=device)
dense_b5 = torch.zeros(len(classes), dtype=torch.float32, device=device)


# BATCH NORM PARAMETERS #

bn_gamma_1 = torch.ones(KERNEL_OUT_C_1, device=device)
bn_beta_1 = torch.zeros(KERNEL_OUT_C_1, device=device)
running_mean_1 = torch.zeros(KERNEL_OUT_C_1, device=device)
running_var_1 = torch.ones(KERNEL_OUT_C_1, device=device)

bn_gamma_2 = torch.ones(KERNEL_OUT_C_2, device=device)
bn_beta_2 = torch.zeros(KERNEL_OUT_C_2, device=device)
running_mean_2 = torch.zeros(KERNEL_OUT_C_2, device=device)
running_var_2 = torch.ones(KERNEL_OUT_C_2, device=device)

bn_gamma_3 = torch.ones(KERNEL_OUT_C_3, device=device)
bn_beta_3 = torch.zeros(KERNEL_OUT_C_3, device=device)
running_mean_3 = torch.zeros(KERNEL_OUT_C_3, device=device)
running_var_3 = torch.ones(KERNEL_OUT_C_3, device=device)


bn_gamma_4 = torch.ones(NEURONS_L1, device=device)
bn_beta_4  = torch.zeros(NEURONS_L1, device=device)
running_mean_4 = torch.zeros(NEURONS_L1, device=device)
running_var_4  = torch.ones(NEURONS_L1, device=device)

bn_gamma_5 = torch.ones(NEURONS_L2, device=device)
bn_beta_5  = torch.zeros(NEURONS_L2, device=device)
running_mean_5 = torch.zeros(NEURONS_L2, device=device)
running_var_5  = torch.ones(NEURONS_L2, device=device)

bn_gamma_6 = torch.ones(NEURONS_L3, device=device)
bn_beta_6  = torch.zeros(NEURONS_L3, device=device)
running_mean_6 = torch.zeros(NEURONS_L3, device=device)
running_var_6  = torch.ones(NEURONS_L3, device=device)

bn_gamma_7 = torch.ones(NEURONS_L4, device=device)
bn_beta_7  = torch.zeros(NEURONS_L4, device=device)
running_mean_7 = torch.zeros(NEURONS_L4, device=device)
running_var_7  = torch.ones(NEURONS_L4, device=device)


parameters: list[torch.Tensor] = [
    kernel_w1, kernel_w2, kernel_w3, dense_w1, dense_w2, dense_w3, dense_w4, dense_w5, dense_b5,
    bn_gamma_1, bn_beta_1, bn_gamma_2, bn_beta_2, bn_gamma_3, bn_beta_3, bn_gamma_4, bn_beta_4, bn_gamma_5, bn_beta_5, bn_gamma_6, bn_beta_6, bn_gamma_7, bn_beta_7
]

m_state = [torch.zeros_like(p) for p in parameters]
v_state = [torch.zeros_like(p) for p in parameters]

for p in parameters:
    p.requires_grad = True


### TRAINING ###

criterion = torch.nn.CrossEntropyLoss()

t = 0

for i in range(EPOCHS):
    print(f"\nEpoch: {i+1}")

    train_loss = 0
    train_correct = 0

    for b, (x, y) in enumerate(utils.dataset.batch_loader(x_train, y_train, BATCH_SIZE, device)):
        x = x.to(device)
        y = y.to(device)

        x = torch.stack([train_transforms(img) for img in x])


        # Conv Layer 1
        o = torch.nn.functional.pad(x, (1, 1, 1, 1), value=0)
        o = torch.conv2d(o, kernel_w1)
        
        mean_o = o.mean(dim=(0, 2, 3))
        var_o = o.var(dim=(0, 2, 3), unbiased=False)
        o = (o - mean_o.view(1, KERNEL_OUT_C_1, 1, 1)) / torch.sqrt(var_o.view(1, KERNEL_OUT_C_1, 1, 1) + EPSILON)
        o = bn_gamma_1.view(1, KERNEL_OUT_C_1, 1, 1) * o + bn_beta_1.view(1, KERNEL_OUT_C_1, 1, 1)

        with torch.no_grad():
            running_mean_1.mul_(1-BN_MOMENTUM).add_(BN_MOMENTUM* mean_o)
            running_var_1.mul_(1-BN_MOMENTUM).add_(BN_MOMENTUM* var_o)
        
        o = torch.relu(o)
        o = torch.max_pool2d(o, POOL_SIZE, POOL_STRIDE)


        # Conv Layer 2
        o = torch.nn.functional.pad(o, (1, 1, 1, 1), value=0)
        o = torch.conv2d(o, kernel_w2)

        mean_o = o.mean(dim=(0, 2, 3))
        var_o = o.var(dim=(0, 2, 3), unbiased=False)
        o = (o - mean_o.view(1, KERNEL_OUT_C_2, 1, 1)) / torch.sqrt(var_o.view(1, KERNEL_OUT_C_2, 1, 1) + EPSILON)
        o = bn_gamma_2.view(1, KERNEL_OUT_C_2, 1, 1) * o + bn_beta_2.view(1, KERNEL_OUT_C_2, 1, 1)

        with torch.no_grad():
            running_mean_2.mul_(1-BN_MOMENTUM).add_(BN_MOMENTUM* mean_o)
            running_var_2.mul_(1-BN_MOMENTUM).add_(BN_MOMENTUM* var_o)

        o = torch.relu(o)
        o = torch.max_pool2d(o, POOL_SIZE, POOL_STRIDE)


        # Conv Layer 3
        o = torch.nn.functional.pad(o, (1, 1, 1, 1), value=0)
        o = torch.conv2d(o, kernel_w3)

        mean_o = o.mean(dim=(0, 2, 3))
        var_o = o.var(dim=(0, 2, 3), unbiased=False)
        o = (o - mean_o.view(1, KERNEL_OUT_C_3, 1, 1)) / torch.sqrt(var_o.view(1, KERNEL_OUT_C_3, 1, 1) + EPSILON)
        o = bn_gamma_3.view(1, KERNEL_OUT_C_3, 1, 1) * o + bn_beta_3.view(1, KERNEL_OUT_C_3, 1, 1)

        with torch.no_grad():
            running_mean_3.mul_(1-BN_MOMENTUM).add_(BN_MOMENTUM* mean_o)
            running_var_3.mul_(1-BN_MOMENTUM).add_(BN_MOMENTUM* var_o)

        o = torch.relu(o)
        o = torch.max_pool2d(o, POOL_SIZE, POOL_STRIDE)


        # Flatten
        o = o.flatten(1)


        # Dense Layer 1
        o = torch.matmul(o, dense_w1)

        mean_o = o.mean(dim=0)
        var_o = o.var(dim=0, unbiased=False)
        o = (o - mean_o) / torch.sqrt(var_o + EPSILON)
        o = bn_gamma_4 * o + bn_beta_4

        with torch.no_grad():
            running_mean_4.mul_(1-BN_MOMENTUM).add_(BN_MOMENTUM*mean_o)
            running_var_4.mul_(1-BN_MOMENTUM).add_(BN_MOMENTUM*var_o)

        o = torch.relu(o)

        # Dropout Layer 1
        o = utils.layers.dropout(o, DROPOUT_RATE)


        # Dense Layer 2
        o = torch.matmul(o, dense_w2)

        mean_o = o.mean(dim=0)
        var_o = o.var(dim=0, unbiased=False)
        o = (o - mean_o) / torch.sqrt(var_o + EPSILON)
        o = bn_gamma_5 * o + bn_beta_5

        with torch.no_grad():
            running_mean_5.mul_(1-BN_MOMENTUM).add_(BN_MOMENTUM*mean_o)
            running_var_5.mul_(1-BN_MOMENTUM).add_(BN_MOMENTUM*var_o)

        o = torch.relu(o)

        # Dropout Layer 2
        o = utils.layers.dropout(o, DROPOUT_RATE)


        # Dense Layer 3
        o = torch.matmul(o, dense_w3)

        mean_o = o.mean(dim=0)
        var_o = o.var(dim=0, unbiased=False)
        o = (o - mean_o) / torch.sqrt(var_o + EPSILON)
        o = bn_gamma_6 * o + bn_beta_6

        with torch.no_grad():
            running_mean_6.mul_(1-BN_MOMENTUM).add_(BN_MOMENTUM*mean_o)
            running_var_6.mul_(1-BN_MOMENTUM).add_(BN_MOMENTUM*var_o)

        o = torch.relu(o)

        # Dropout Layer 3
        o = utils.layers.dropout(o, DROPOUT_RATE)


        # Dense Layer 4
        o = torch.matmul(o, dense_w4)

        mean_o = o.mean(dim=0)
        var_o = o.var(dim=0, unbiased=False)
        o = (o - mean_o) / torch.sqrt(var_o + EPSILON)
        o = bn_gamma_7 * o + bn_beta_7

        with torch.no_grad():
            running_mean_7.mul_(1-BN_MOMENTUM).add_(BN_MOMENTUM*mean_o)
            running_var_7.mul_(1-BN_MOMENTUM).add_(BN_MOMENTUM*var_o)

        o = torch.relu(o)

        # Dropout Layer 4
        o = utils.layers.dropout(o, DROPOUT_RATE)


        # Dense Layer 5
        o = torch.matmul(o, dense_w5) + dense_b5

        # Learning 
        loss: torch.Tensor = criterion(o, y)
        train_loss += loss.item()

        train_correct += (torch.softmax(o, dim=1).argmax(dim=1) == y).sum().item()

        for p in parameters:
            p.grad = None

        loss.backward()
        
        with torch.no_grad():
            t += 1

            for i in range(len(parameters)):
                m_state[i] = BETA_1 * m_state[i] + (1 - BETA_1) * parameters[i].grad
                v_state[i] = BETA_2 * v_state[i] + (1 - BETA_2) * parameters[i].grad**2

                m_hat = m_state[i] / (1 - BETA_1**t)
                v_hat = v_state[i] / (1 - BETA_2**t)

                parameters[i] -= LEARNING_RATE * m_hat / (v_hat**0.5 + EPSILON)

        print(f"\rtrain_loss: {(train_loss / (b+1)):.4f}\tBatch: {b+1}/{x_train.size(0) // BATCH_SIZE}", end="")

    print(f"\rtrain_loss: {(train_loss / (b+1)):.4f}\ttrain_accuracy: {(train_correct / x_train.size(0)):.4f}")
    
    val_loss = 0
    val_correct = 0

    with torch.no_grad():
        for x, y in utils.dataset.batch_loader(x_val, y_val, BATCH_SIZE, device):
            x = x.to(device)
            y = y.to(device)

            x = torch.stack([val_transforms(img) for img in x])
            
            # Conv Layer 1
            o = torch.nn.functional.pad(x, (1, 1, 1, 1), value=0)
            o = torch.conv2d(o, kernel_w1)

            o = (o - running_mean_1.view(1, KERNEL_OUT_C_1, 1, 1)) / torch.sqrt(running_var_1.view(1, KERNEL_OUT_C_1, 1, 1) + EPSILON)
            o = bn_gamma_1.view(1, KERNEL_OUT_C_1, 1, 1) * o + bn_beta_1.view(1, KERNEL_OUT_C_1, 1, 1)

            o = torch.relu(o)
            o = torch.max_pool2d(o, POOL_SIZE, POOL_STRIDE)


            # Conv Layer 2
            o = torch.nn.functional.pad(o, (1, 1, 1, 1), value=0)
            o = torch.conv2d(o, kernel_w2)

            o = (o - running_mean_2.view(1, KERNEL_OUT_C_2, 1, 1)) / torch.sqrt(running_var_2.view(1, KERNEL_OUT_C_2, 1, 1) + EPSILON)
            o = bn_gamma_2.view(1, KERNEL_OUT_C_2, 1, 1) * o + bn_beta_2.view(1, KERNEL_OUT_C_2, 1, 1)

            o = torch.relu(o)
            o = torch.max_pool2d(o, POOL_SIZE, POOL_STRIDE)


            # Conv Layer 3
            o = torch.nn.functional.pad(o, (1, 1, 1, 1), value=0)
            o = torch.conv2d(o, kernel_w3)

            o = (o - running_mean_3.view(1, KERNEL_OUT_C_3, 1, 1)) / torch.sqrt(running_var_3.view(1, KERNEL_OUT_C_3, 1, 1) + EPSILON)
            o = bn_gamma_3.view(1, KERNEL_OUT_C_3, 1, 1) * o + bn_beta_3.view(1, KERNEL_OUT_C_3, 1, 1)

            o = torch.relu(o)
            o = torch.max_pool2d(o, POOL_SIZE, POOL_STRIDE)


            # Flatten
            o = o.flatten(1)


            # Dense Layer 1
            o = torch.matmul(o, dense_w1)

            o = (o - running_mean_4) / torch.sqrt(running_var_4 + EPSILON)
            o = bn_gamma_4 * o + bn_beta_4

            o = torch.relu(o)


            # Dense Layer 2
            o = torch.matmul(o, dense_w2)

            o = (o - running_mean_5) / torch.sqrt(running_var_5 + EPSILON)
            o = bn_gamma_5 * o + bn_beta_5

            o = torch.relu(o)


            # Dense Layer 3
            o = torch.matmul(o, dense_w3)

            o = (o - running_mean_6) / torch.sqrt(running_var_6 + EPSILON)
            o = bn_gamma_6 * o + bn_beta_6

            o = torch.relu(o)


            # Dense Layer 4
            o = torch.matmul(o, dense_w4)

            o = (o - running_mean_7) / torch.sqrt(running_var_7 + EPSILON)
            o = bn_gamma_7 * o + bn_beta_7

            o = torch.relu(o)


            # Dense Layer 5
            o = torch.matmul(o, dense_w5) + dense_b5

            loss: torch.Tensor = criterion(o, y)
            val_loss += loss.item()

            val_correct += (torch.softmax(o, dim=1).argmax(dim=1) == y).sum().item()

        print(f"\rval_loss: {(val_loss / (x_val.size(0) // BATCH_SIZE)):.4f}\tval_accuracy: {(val_correct / x_val.size(0)):.4f}")

state_dict = {
    "kernel_w1": kernel_w1,
    "kernel_w2": kernel_w2,
    "kernel_w3": kernel_w3,
    "dense_w1": dense_w1,
    "dense_w2": dense_w2,
    "dense_w3": dense_w3,
    "dense_w4": dense_w4,
    "dense_w5": dense_w5,
    "dense_b5": dense_b5,
    
    "bn_gamma_1": bn_gamma_1,
    "bn_beta_1": bn_beta_1,
    "running_mean_1": running_mean_1,
    "running_var_1": running_var_1,
    
    "bn_gamma_2": bn_gamma_2,
    "bn_beta_2": bn_beta_2,
    "running_mean_2": running_mean_2,
    "running_var_2": running_var_2,

    "bn_gamma_3": bn_gamma_3,
    "bn_beta_3": bn_beta_3,
    "running_mean_3": running_mean_3,
    "running_var_3": running_var_3,

    "bn_gamma_4": bn_gamma_4,
    "bn_beta_4": bn_beta_4,
    "running_mean_4": running_mean_4,
    "running_var_4": running_var_4,
    
    "bn_gamma_5": bn_gamma_5,
    "bn_beta_5": bn_beta_5,
    "running_mean_5": running_mean_5,
    "running_var_5": running_var_5,
    
    "bn_gamma_6": bn_gamma_6,
    "bn_beta_6": bn_beta_6,
    "running_mean_6": running_mean_6,
    "running_var_6": running_var_6,
    
    "bn_gamma_7": bn_gamma_7,
    "bn_beta_7": bn_beta_7,
    "running_mean_7": running_mean_7,
    "running_var_7": running_var_7
}

torch.save(state_dict, MODEL_NAME)