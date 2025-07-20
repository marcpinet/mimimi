import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(in_channels, in_channels * 4)
        self.fc2 = nn.Linear(in_channels * 4, in_channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        conv_out = self.conv(x)
        pooled = self.avg_pool(conv_out)
        squeezed = pooled.view(batch_size, -1)

        attention = self.fc1(squeezed)
        attention = F.relu(attention)
        attention = self.fc2(attention)
        attention = self.softmax(attention)

        attention = attention.view(batch_size, channels, 1, 1)

        attended = x * attention
        output = attended + x

        return output