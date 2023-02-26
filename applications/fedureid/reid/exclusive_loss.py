from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch import nn, autograd


class Exclusive(autograd.Function):
    # def __init__(ctx, V):
    #     super(Exclusive, ctx).__init__()
    #     ctx.V = V

    @staticmethod
    def forward(ctx, inputs, targets, V):
        ctx.V = V
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.V.t())
        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = grad_outputs.mm(ctx.V) if ctx.needs_input_grad[0] else None
        for x, y in zip(inputs, targets):
            ctx.V[y] = F.normalize( (ctx.V[y] + x) / 2, p=2, dim=0)
        return grad_inputs, None, None


class ExLoss(nn.Module):
    def __init__(self, num_features, num_classes, t=1.0, weight=None):
        super(ExLoss, self).__init__()
        self.num_features = num_features
        self.t = t
        self.weight = weight
        self.register_buffer('V', torch.zeros(num_classes, num_features))

    def forward(self, inputs, targets):
        outputs = Exclusive.apply(inputs, targets, self.V) * self.t
        # outputs = Exclusive(self.V)(inputs, targets) * self.t
        loss = F.cross_entropy(outputs, targets, weight=self.weight)
        return loss, outputs
