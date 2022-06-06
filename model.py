import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchsummary import summary


class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(conv_bn_relu, self).__init__()
        layers = [nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False)]
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class StrokeMaskPredictionModule(nn.Module):
    def __init__(self):
        super(StrokeMaskPredictionModule, self).__init__()

        self.down1 = nn.Sequential(
            conv_bn_relu(32, 3),
            conv_bn_relu(32, 32),
            nn.Conv2d(32, 64, 3, 2, padding=1)
        )
        self.down2 = nn.Sequential(
            conv_bn_relu(64, 64),
            conv_bn_relu(64, 64),
            nn.Conv2d(64, 128, 3, 2, padding=1)
        )

    def forward(self, x):
        pass
