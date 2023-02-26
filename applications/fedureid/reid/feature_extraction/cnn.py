from __future__ import absolute_import
from collections import OrderedDict

from torch.autograd import Variable

from ..utils import to_torch
import torch


def extract_cnn_feature(model, inputs, modules=None):
    with torch.no_grad():
        model.eval()
        inputs = to_torch(inputs)
        inputs = Variable(inputs)
        if modules is None:
            fcs, pool5s = model(inputs)
            fcs = fcs.data.cpu()
            pool5s = pool5s.data.cpu()
            return fcs, pool5s
        # Register forward hook for each module
        outputs = OrderedDict()
        handles = []
        for m in modules:
            outputs[id(m)] = None
            def func(m, i, o): outputs[id(m)] = o.data.cpu()
            handles.append(m.register_forward_hook(func))
        model(inputs)
        for h in handles:
            h.remove()
    return list(outputs.values())
