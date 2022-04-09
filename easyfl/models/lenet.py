from torch import nn
import torch.nn.functional as F

from easyfl.models.model import BaseModel


class Model(BaseModel):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=(2, 2))
        self.conv2 = nn.Conv2d(32, 64, 5, padding=(2, 2))
        self.fc1 = nn.Linear(7 * 7 * 64, 2048)
        self.fc2 = nn.Linear(2048, 62)
        self.init_weights()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 7 * 7 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def init_weights(self):
        init_range = 0.1
        self.conv1.weight.data.uniform_(-init_range, init_range)
        self.conv1.bias.data.zero_()
        self.conv2.weight.data.uniform_(-init_range, init_range)
        self.conv2.bias.data.zero_()
        self.fc1.weight.data.uniform_(-init_range, init_range)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.uniform_(-init_range, init_range)
        self.fc2.bias.data.zero_()
