from torch import nn
from layers.anti import Downsample


class AntiMaxPool(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=None):
        super(AntiMaxPool, self).__init__()
        self.pool = nn.Sequential(
            nn.MaxPool2d(kernel_size, stride=1),
            Downsample(channels=in_channels, stride=stride)
        )

    def forward(self, x):
        return self.pool(x)


if __name__ == '__main__':
    import torch

    x = torch.rand(1, 2, 8, 8)
    pool = AntiMaxPool(2, 2, 2)
    y = pool(x)
    print(y)
