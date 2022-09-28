import torch
from torch import nn


class BriefNet(nn.Module):
    def __init__(self):
        super(BriefNet, self).__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, data):
        data = self.sequence(data)
        return data


if __name__ == '__main__':
    bNet = BriefNet()
    test_input = torch.ones((64, 3, 32, 32))
    test_output = bNet(test_input)
    print(test_output.shape)
