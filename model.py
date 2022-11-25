import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.prior = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                      kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.downsample = downsample

    def forward(self, x):
        shortcut = x
        # 层间连接为虚线连接，借助下采样实现规格统一
        if self.downsample is not None:
            shortcut = self.downsample(x)
        out = self.prior(x)
        out += shortcut
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.prior = nn.Sequential(
            # First
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            # Second
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                      kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            # Third
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                      kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel * self.expansion)
        )
        self.downsample = downsample

    def forward(self, x):
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)
        out = self.prior(x)
        out += shortcut
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.pre = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.in_channel,
                      kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, blocks_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, blocks_num):
            layers.append(block(self.in_channel, channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x


def resnet18(num_classes, include_top=True):
    return ResNet(ResidualBlock, [2, 2, 2, 2],
                  num_classes=num_classes, include_top=include_top)


def resnet34(num_classes, include_top=True):
    return ResNet(ResidualBlock, [3, 4, 6, 3],
                  num_classes=num_classes, include_top=include_top)


def resnet50(num_classes, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes, include_top=include_top)


def resnet101(num_classes, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes, include_top=include_top)
