import torch
import typing

### CONSTANTS ###

DENSE_NEURONS_1 = 256

### RESIDUAL BLOCKS ###

class ResidualBlock(torch.nn.Module):
    def __init__(self: typing.Self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)

        self.conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = torch.nn.Identity()

        self.relu = torch.nn.ReLU()

    def forward(self: typing.Self, x: torch.Tensor) -> torch.Tensor:
        o = self.conv1(x)
        o = self.bn1(o)
        o = self.relu(o)

        o = self.conv2(o)
        o = self.bn2(o)
        o += self.shortcut(x)
        o = self.relu(o)

        return o

### MODEL ###

class ResidualNetwork(torch.nn.Module):
    def __init__(self: typing.Self, in_channels: int, n_classes: int) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes

        self.conv = torch.nn.Sequential(
            ResidualBlock(self.in_channels, 64),
            ResidualBlock(64, 64),

            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),

            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, DENSE_NEURONS_1)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(DENSE_NEURONS_1, self.n_classes),
        )

        self._init_weights()


    def _init_weights(self: typing.Self) -> None:
        for module in self.modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(module.weight)

                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)


    def forward(self: typing.Self, x: torch.Tensor) -> torch.Tensor:
        o = self.conv(x)
        o = self.classifier(o)

        return o