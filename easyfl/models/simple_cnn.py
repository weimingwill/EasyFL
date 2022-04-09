import torch
import torch.nn.functional as F
from torch import nn

from easyfl.models import BaseModel


class Model(BaseModel):
    def __init__(self, channels=32):
        super(Model, self).__init__()
        self.num_channels = channels
        self.conv1 = nn.Conv2d(3, self.num_channels, 3, stride=1)
        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels * 2, 3, stride=1)
        self.conv3 = nn.Conv2d(self.num_channels * 2, self.num_channels * 2, 3, stride=1)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(4 * 4 * self.num_channels * 2, self.num_channels * 2)
        self.fc2 = nn.Linear(self.num_channels * 2, 10)

    def forward(self, s):
        s = self.conv1(s)  # batch_size x num_channels x 32 x 32

        s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels x 16 x 16

        s = self.conv2(s)  # batch_size x num_channels*2 x 16 x 16

        s = F.relu(F.max_pool2d(s, 2))  # batch_size x num_channels*2 x 8 x 8

        s = self.conv3(s)  # batch_size x num_channels*2 x 8 x 8

        # s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*2 x 4 x 4

        # flatten the output for each image
        s = s.view(-1, 4 * 4 * self.num_channels * 2)  # batch_size x 4*4*num_channels*4

        # apply 2 fully connected layers with dropout
        s = F.relu(self.fc1(s))
        s = self.fc2(s)  # batch_size x 10

        return s
