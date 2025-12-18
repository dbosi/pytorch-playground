import torch

INPUT_SIZE = 800
NEURONS_L1 = 128
NEURONS_L2 = 64
NEURONS_OUTPUT = 10

CHANNELS_1 = 16
CHANNELS_2 = 32

KERNEL_STRIDE = 1
KERNEL_SIZE = 3

POOL_SIZE = 2
POOL_STRIDE = 2

DROPOUT_RATE = 0.5

class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv_layer = torch.nn.Sequential(
            torch.nn.Conv2d(1, CHANNELS_1, (KERNEL_SIZE, KERNEL_SIZE)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((POOL_SIZE, POOL_SIZE), POOL_SIZE),
            
            torch.nn.Conv2d(CHANNELS_1, CHANNELS_2, (KERNEL_SIZE, KERNEL_SIZE)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d((POOL_SIZE, POOL_SIZE), POOL_SIZE)
        )

        self.dense_layer = torch.nn.Sequential(
            torch.nn.Flatten(),

            torch.nn.Linear(INPUT_SIZE, NEURONS_L1),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=DROPOUT_RATE),
            
            torch.nn.Linear(NEURONS_L1, NEURONS_L2),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=DROPOUT_RATE),

            torch.nn.Linear(NEURONS_L2, NEURONS_OUTPUT)
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layer(x)
        x = self.dense_layer(x)

        return x