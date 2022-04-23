import logging
import os
import time

import numpy as np
import torch
import torch._utils
import torch.nn as nn
import torch.optim as optim

from easyfl.client.base import BaseClient
from easyfl.distributed.distributed import CPU
from easyfl.pb import common_pb2 as common_pb
from easyfl.pb import server_service_pb2 as server_pb
from easyfl.protocol import codec
from easyfl.tracking import metric
from evaluate import test_evaluate, extract_feature
from model import get_classifier

logger = logging.getLogger(__name__)


class FedReIDClient(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, sleep_time=0):
        super(FedReIDClient, self).__init__(cid, conf, train_data, test_data, device, sleep_time)
        self.classifier = get_classifier(len(self.train_data.classes[cid])).to(device)
        self.gallery_cam = None
        self.gallery_label = None
        self.query_cam = None
        self.query_label = None
        self.test_gallery_loader = None
        self.test_query_loader = None

    def train(self, conf, device=CPU):
        self.model.classifier.classifier = self.classifier.to(device)
        start_time = time.time()
        loss_fn, optimizer = self.pretrain_setup(conf, device)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        epoch_loss = []
        for i in range(conf.local_epoch):
            batch_loss = []
            for batched_x, batched_y in self.train_loader:
                x, y = batched_x.to(device), batched_y.to(device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            scheduler.step()
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(float(current_epoch_loss))
            logger.info("Client {}, local epoch: {}, loss: {}".format(self.cid, i, current_epoch_loss))
        self.current_round_time = time.time() - start_time
        self.track(metric.TRAIN_TIME, self.current_round_time)
        self.track(metric.TRAIN_LOSS, epoch_loss)
        self.classifier = self.model.classifier.classifier
        self.model.classifier.classifier = nn.Sequential()

    def load_optimizer(self, conf):
        ignored_params = list(map(id, self.model.classifier.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, self.model.parameters())
        optimizer_ft = optim.SGD([
            {'params': base_params, 'lr': 0.1 * conf.optimizer.lr},
            {'params': self.model.classifier.parameters(), 'lr': conf.optimizer.lr}
        ], weight_decay=5e-4, momentum=conf.optimizer.momentum, nesterov=True)
        return optimizer_ft

    def test(self, conf, device=CPU):
        self.model = self.model.eval()
        self.model = self.model.to(device)
        gallery_id = '{}_{}'.format(self.cid, 'gallery')
        query_id = '{}_{}'.format(self.cid, 'query')
        if self.test_gallery_loader is None or self.test_query_loader is None:
            self.test_gallery_loader = self.test_data.loader(batch_size=128,
                                                             client_id=gallery_id,
                                                             shuffle=False,
                                                             seed=conf.seed)
            self.test_query_loader = self.test_data.loader(batch_size=128,
                                                           client_id=query_id,
                                                           shuffle=False,
                                                           seed=conf.seed)
            gallery_path = [(self.test_data.data[gallery_id]['x'][i],
                             self.test_data.data[gallery_id]['y'][i])
                            for i in range(len(self.test_data.data[gallery_id]['y']))]
            query_path = [(self.test_data.data[query_id]['x'][i],
                           self.test_data.data[query_id]['y'][i])
                          for i in range(len(self.test_data.data[query_id]['y']))]
            gallery_cam, gallery_label = self._get_id(gallery_path)
            self.gallery_cam = gallery_cam
            self.gallery_label = gallery_label
            query_cam, query_label = self._get_id(query_path)
            self.query_cam = query_cam
            self.query_label = query_label
        with torch.no_grad():
            gallery_feature = extract_feature(self.model,
                                              self.test_gallery_loader,
                                              device)
            query_feature = extract_feature(self.model,
                                            self.test_query_loader,
                                            device)

        result = {
            'gallery_f': gallery_feature.numpy(),
            'gallery_label': np.array([self.gallery_label]),
            'gallery_cam': np.array([self.gallery_cam]),
            'query_f': query_feature.numpy(),
            'query_label': np.array([self.query_label]),
            'query_cam': np.array([self.query_cam]),
        }

        logger.info("Evaluating {}".format(self.cid))
        rank1, rank5, rank10, mAP = test_evaluate(result, device)
        logger.info("Dataset: {} Rank@1:{:.2%} Rank@5:{:.2%} Rank@10:{:.2%} mAP:{:.2%}".format(
            self.cid, rank1, rank5, rank10, mAP))
        self._upload_holder = server_pb.UploadContent(
            data=codec.marshal(server_pb.Performance(accuracy=rank1, loss=0)),  # loss not applicable
            type=common_pb.DATA_TYPE_PERFORMANCE,
            data_size=len(self.query_label),
        )

    def _get_id(self, img_path):
        camera_id = []
        labels = []
        for p, v in img_path:
            filename = os.path.basename(p)
            if filename[:3] != 'cam':
                label = filename[0:4]
                camera = filename.split('c')[1]
                camera = camera.split('s')[0]
            else:
                label = filename.split('_')[2]
                camera = filename.split('_')[1]
            if label[0:2] == '-1':
                labels.append(-1)
            else:
                labels.append(int(label))
            camera_id.append(int(camera[0]))
        return camera_id, labels
