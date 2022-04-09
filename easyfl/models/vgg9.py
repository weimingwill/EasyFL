import torch
import torch.nn as nn
import math
from easyfl.models import BaseModel

cfg = {
    'VGG9': [32, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
}


def make_layers(cfg, batch_norm):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]

            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class Model(BaseModel):
    def __init__(self, features=make_layers(cfg['VGG9'], batch_norm=False), num_classes=10):
        super(Model, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.reset_parameters()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def VGG9(batch_norm=False, **kwargs):
    model = Model(make_layers(cfg['VGG9'], batch_norm), **kwargs)
    return model
