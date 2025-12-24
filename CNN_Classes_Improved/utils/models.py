import torch
import typing

### CONSTANTS ###

# Conv Layer #

KERNEL_IN_C_1 = 3
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

# DENSE LAYER #

NEURONS_L1 = 512
NEURONS_L2 = 256
NEURONS_L3 = 128
NEURONS_L4 = 64

# OTHER #

DROPOUT_RATE = 0.3

### MODEL ###

class ImprovedCNN(torch.nn.Module):
    def __init__(self, in_shape: tuple[int, int, int], n_classes: int):
        super(ImprovedCNN, self).__init__()

        self.in_shape = in_shape
        self.n_classes = n_classes

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=KERNEL_IN_C_1, out_channels=KERNEL_OUT_C_1, kernel_size=KERNEL_SIZE_1, padding=PADDING_SIZE, bias=False),
            torch.nn.BatchNorm2d(KERNEL_OUT_C_1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=POOL_SIZE, stride=POOL_STRIDE),

            torch.nn.Conv2d(in_channels=KERNEL_IN_C_2, out_channels=KERNEL_OUT_C_2, kernel_size=KERNEL_SIZE_2, padding=PADDING_SIZE, bias=False),
            torch.nn.BatchNorm2d(KERNEL_OUT_C_2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=POOL_SIZE, stride=POOL_STRIDE),

            torch.nn.Conv2d(in_channels=KERNEL_IN_C_3, out_channels=KERNEL_OUT_C_3, kernel_size=KERNEL_SIZE_3, padding=PADDING_SIZE, bias=False),
            torch.nn.BatchNorm2d(KERNEL_OUT_C_3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=POOL_SIZE, stride=POOL_STRIDE),

            torch.nn.Flatten()
        )

        self._init_dense_layer()

    def _init_dense_layer(self):
        with torch.no_grad():
            n_channels, img_h, img_w = self.in_shape

            dummy = torch.zeros(1, n_channels, img_h, img_w)

            o = self.conv(dummy)
        
            _, flattened_size = o.shape

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(flattened_size, NEURONS_L1, bias=False),
            torch.nn.BatchNorm1d(NEURONS_L1),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT_RATE),

            torch.nn.Linear(NEURONS_L1, NEURONS_L2, bias=False),
            torch.nn.BatchNorm1d(NEURONS_L2),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT_RATE),

            torch.nn.Linear(NEURONS_L2, NEURONS_L3, bias=False),
            torch.nn.BatchNorm1d(NEURONS_L3),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT_RATE),

            torch.nn.Linear(NEURONS_L3, NEURONS_L4, bias=False),
            torch.nn.BatchNorm1d(NEURONS_L4),
            torch.nn.ReLU(),
            torch.nn.Dropout(DROPOUT_RATE),

            torch.nn.Linear(NEURONS_L4, self.n_classes)
        )

        self._init_weights()

    
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


    def forward(self, x) -> torch.Tensor:
        o = self.conv(x)
        o = self.dense(o)

        return o