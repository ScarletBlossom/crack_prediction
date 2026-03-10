import torch
import torch.nn as nn

from .blocks import DownSample, init_weights


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_layers = nn.ModuleList([
            DownSample(6, 64, 4, 2, use_bn=False),
            DownSample(64, 128, 4, 2, use_bn=True),
            DownSample(128, 256, 4, 2, use_bn=True),
        ])
        self.pad1 = nn.ZeroPad2d(1)
        self.conv1 = nn.Conv2d(256, 512, kernel_size=4, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.activation = nn.LeakyReLU(0.3, inplace=True)
        self.pad2 = nn.ZeroPad2d(1)
        self.final = nn.Conv2d(512, 1, kernel_size=4, stride=1)
        init_weights(self)

    def forward(self, inp: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
        x = torch.cat([inp, tar], dim=1)
        for layer in self.down_layers:
            x = layer(x)
        x = self.pad1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.pad2(x)
        x = self.final(x)
        return x
