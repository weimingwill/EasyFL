import gc
import logging

import torch
import torch._utils

from losses import get_losses
from trainer import Trainer, LR_POLY
from easyfl.client.base import BaseClient
from easyfl.distributed.distributed import CPU

logger = logging.getLogger(__name__)


class MASClient(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, sleep_time=0):
        super(MASClient, self).__init__(cid, conf, train_data, test_data, device, sleep_time)
        self._local_model = None
        criteria = self.load_loss_fn(conf)
        train_loader = self.load_loader(conf)
        self.trainer = Trainer(self.cid, conf, train_loader, self.model, optimizer=None, criteria=criteria, device=device)

    def decompression(self):
        if self.model is None:
            # Initialization at beginning of the task
            self.model = self.compressed_model

    def train(self, conf, device=CPU):
        self.model.to(device)
        optimizer = self.load_optimizer(conf)
        self.trainer.update(self.model, optimizer, device)
        transference = self.trainer.train()
        if conf.lookahead == 'y':
            logger.info(f"Round {conf.round_id} - Client {self.cid} transference: {transference}")

    def load_loss_fn(self, conf):
        criteria = get_losses(conf.task_str, conf.rotate_loss, conf.task_weights)
        return criteria

    def load_loader(self, conf):
        train_loader = self.train_data.loader(conf.batch_size,
                                              self.cid,
                                              shuffle=True,
                                              num_workers=conf.num_workers,
                                              seed=conf.seed)
        return train_loader

    def load_optimizer(self, conf, lr=None):
        if conf.optimizer.lr_type == LR_POLY:
            lr = conf.optimizer.lr * pow(1 - (conf.round_id / conf.rounds), 0.9)
        else:
            if self.trainer.lr:
                lr = self.trainer.lr
            else:
                lr = conf.optimizer.lr

        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=lr,
                                    momentum=conf.optimizer.momentum,
                                    weight_decay=conf.optimizer.weight_decay)
        return optimizer

    def post_upload(self):
        del self.model
        del self.compressed_model
        self.model = None
        self.compressed_model = None
        assert self.model is None
        assert self.compressed_model is None
        gc.collect()
