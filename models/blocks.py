import torch
import torch.nn as nn


class DownSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, use_bn: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, 1, bias=False)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.3, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, dropout: bool = False):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, 1, bias=False), nn.BatchNorm2d(out_channels)]
        if dropout:
            layers.append(nn.Dropout(0.2))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpSampleBilinear(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.3, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.activation(x)
        return x


class DownPool(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool_size: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(pool_size),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.3, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BigConv(nn.Module):
    def __init__(self, in_channels: int, filters: int):
        super().__init__()
        out_channels = filters * 5
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.3, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class OutputBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def init_weights(module: nn.Module) -> None:
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
