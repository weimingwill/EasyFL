from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init

from easyfl.models.model import BaseModel
from .resnet import *

__all__ = ["BUCModel"]


class AvgPooling(nn.Module):
    def __init__(self, input_feature_size, embedding_feature_size=2048, dropout=0.5):
        super(self.__class__, self).__init__()

        # embedding
        self.embedding_feature_size = embedding_feature_size
        self.embedding = nn.Linear(input_feature_size, embedding_feature_size)
        self.embedding_bn = nn.BatchNorm1d(embedding_feature_size)
        init.kaiming_normal_(self.embedding.weight, mode='fan_out')
        init.constant_(self.embedding.bias, 0)
        init.constant_(self.embedding_bn.weight, 1)
        init.constant_(self.embedding_bn.bias, 0)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs):
        net = inputs.mean(dim=1)
        eval_features = F.normalize(net, p=2, dim=1)
        net = self.embedding(net)
        net = self.embedding_bn(net)
        net = F.normalize(net, p=2, dim=1)
        net = self.drop(net)
        return net, eval_features


class BUCModel(BaseModel):
    def __init__(self, dropout=0.5, embedding_feature_size=2048):
        super(self.__class__, self).__init__()
        self.CNN = resnet50(dropout=dropout)
        self.avg_pooling = AvgPooling(input_feature_size=2048,
                                      embedding_feature_size=embedding_feature_size,
                                      dropout=dropout)

    def forward(self, x):
        # resnet encoding
        resnet_feature = self.CNN(x)
        shape = resnet_feature.shape

        # reshape back into (batch, samples, ...)
        # samples of video frames, we only use images, so always 1.
        resnet_feature = resnet_feature.view(shape[0], 1, -1)

        # avg pooling
        output = self.avg_pooling(resnet_feature)
        return output
