from __future__ import print_function, absolute_import

import logging
import time

from torch.autograd import Variable

from .evaluation_metrics import accuracy
from .utils.meters import AverageMeter

logger = logging.getLogger(__name__)


class BaseTrainer(object):
    def __init__(self, model, criterion, device, fixed_layer=False):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.fixed_layer = fixed_layer
        self.device = device

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        stop_local_training = False
        precision_avg = []

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            loss, prec1 = self._forward(inputs, targets)

            losses.update(loss.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                logger.info('Epoch: [{}][{}/{}]\t'
                            'Time {:.3f} ({:.3f})\t'
                            'Data {:.3f} ({:.3f})\t'
                            'Loss {:.3f} ({:.3f})\t'
                            'Prec {:.2%} ({:.2%})\t'
                            .format(epoch, i + 1, len(data_loader),
                                    batch_time.val, batch_time.avg,
                                    data_time.val, data_time.avg,
                                    losses.val, losses.avg,
                                    precisions.val, precisions.avg))
            precision_avg.append(precisions.avg)
            if precisions.val == 1 or precisions.avg > 0.95:
                stop_local_training = True
        return stop_local_training

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        x, y = inputs
        inputs = Variable(x.to(self.device), requires_grad=False)
        targets = Variable(y.to(self.device))
        return inputs, targets

    def _forward(self, inputs, targets):
        outputs, _ = self.model(inputs)
        outputs = outputs.to(self.device)
        loss, outputs = self.criterion(outputs, targets)
        prec, = accuracy(outputs.data, targets.data)
        prec = prec[0]
        return loss, prec
