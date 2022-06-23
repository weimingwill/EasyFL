import copy
import gc
import logging
import time
from collections import Counter

import numpy as np
import torch
import torch._utils
import torch.nn as nn
import torch.nn.functional as F

import model
import utils
from communication import ONLINE, TARGET, BOTH, LOCAL, GLOBAL, DAPU, NONE, EMA, DYNAMIC_DAPU, DYNAMIC_EMA_ONLINE, SELECTIVE_EMA
from easyfl.client.base import BaseClient
from easyfl.distributed.distributed import CPU

logger = logging.getLogger(__name__)

L2 = "l2"


class FedSSLClient(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, sleep_time=0):
        super(FedSSLClient, self).__init__(cid, conf, train_data, test_data, device, sleep_time)
        self._local_model = None
        self.DAPU_predictor = LOCAL
        self.encoder_distance = 1
        self.encoder_distances = []
        self.previous_trained_round = -1
        self.weight_scaler = None

    def decompression(self):
        if self.model is None:
            # Initialization at beginning of the task
            self.model = self.compressed_model

        self.update_model()

    def update_model(self):
        if self.conf.model in [model.MoCo, model.MoCoV2]:
            self.model.encoder_q = self.compressed_model.encoder_q
            # self.model.encoder_k = copy.deepcopy(self._local_model.encoder_k)
        elif self.conf.model == model.SimCLR:
            self.model.online_encoder = self.compressed_model.online_encoder
        elif self.conf.model in [model.SimSiam, model.SimSiamNoSG]:
            if self._local_model is None:
                self.model.online_encoder = self.compressed_model.online_encoder
                self.model.online_predictor = self.compressed_model.online_predictor
                return

            if self.conf.update_encoder == ONLINE:
                online_encoder = self.compressed_model.online_encoder
            else:
                raise ValueError(f"Encoder: aggregate {self.conf.aggregate_encoder}, "
                                 f"update {self.conf.update_encoder} is not supported")

            if self.conf.update_predictor == GLOBAL:
                predictor = self.compressed_model.online_predictor
            else:
                raise ValueError(f"Predictor: {self.conf.update_predictor} is not supported")

            self.model.online_encoder = copy.deepcopy(online_encoder)
            self.model.online_predictor = copy.deepcopy(predictor)

        elif self.conf.model in [model.Symmetric, model.SymmetricNoSG]:
            self.model.online_encoder = self.compressed_model.online_encoder

        elif self.conf.model in [model.BYOL, model.BYOLNoSG, model.BYOLNoPredictor]:

            if self._local_model is None:
                logger.info("Use aggregated encoder and predictor")
                self.model.online_encoder = self.compressed_model.online_encoder
                self.model.target_encoder = self.compressed_model.online_encoder
                self.model.online_predictor = self.compressed_model.online_predictor
                return

            def ema_online():
                self._calculate_weight_scaler()
                logger.info(f"Encoder: update online with EMA of global encoder @ round {self.conf.round_id}")
                weight = self.encoder_distance
                weight = min(1, self.weight_scaler * weight)
                weight = 1 - weight
                self.compressed_model = self.compressed_model.cpu()
                online_encoder = self.compressed_model.online_encoder
                target_encoder = self._local_model.target_encoder
                ema_updater = model.EMA(weight)
                model.update_moving_average(ema_updater, online_encoder, self._local_model.online_encoder)
                return online_encoder, target_encoder

            def ema_predictor():
                logger.info(f"Predictor: use dynamic DAPU")
                distance = self.encoder_distance
                distance = min(1, distance * self.weight_scaler)
                if distance > 0.5:
                    weight = distance
                    ema_updater = model.EMA(weight)
                    predictor = self._local_model.online_predictor
                    model.update_moving_average(ema_updater, predictor, self.compressed_model.online_predictor)
                else:
                    weight = 1 - distance
                    ema_updater = model.EMA(weight)
                    predictor = self.compressed_model.online_predictor
                    model.update_moving_average(ema_updater, predictor, self._local_model.online_predictor)
                return predictor

            if self.conf.aggregate_encoder == ONLINE and self.conf.update_encoder == ONLINE:
                logger.info("Encoder: aggregate online, update online")
                online_encoder = self.compressed_model.online_encoder
                target_encoder = self._local_model.target_encoder
            elif self.conf.aggregate_encoder == TARGET and self.conf.update_encoder == ONLINE:
                logger.info("Encoder: aggregate target, update online")
                online_encoder = self.compressed_model.target_encoder
                target_encoder = self._local_model.target_encoder
            elif self.conf.aggregate_encoder == TARGET and self.conf.update_encoder == TARGET:
                logger.info("Encoder: aggregate target, update target")
                online_encoder = self._local_model.online_encoder
                target_encoder = self.compressed_model.target_encoder
            elif self.conf.aggregate_encoder == ONLINE and self.conf.update_encoder == TARGET:
                logger.info("Encoder: aggregate online, update target")
                online_encoder = self._local_model.online_encoder
                target_encoder = self.compressed_model.online_encoder
            elif self.conf.aggregate_encoder == ONLINE and self.conf.update_encoder == BOTH:
                logger.info("Encoder: aggregate online, update both")
                online_encoder = self.compressed_model.online_encoder
                target_encoder = self.compressed_model.online_encoder
            elif self.conf.aggregate_encoder == TARGET and self.conf.update_encoder == BOTH:
                logger.info("Encoder: aggregate target, update both")
                online_encoder = self.compressed_model.target_encoder
                target_encoder = self.compressed_model.target_encoder
            elif self.conf.update_encoder == NONE:
                logger.info("Encoder: use local online and target encoders")
                online_encoder = self._local_model.online_encoder
                target_encoder = self._local_model.target_encoder
            elif self.conf.update_encoder == EMA:
                logger.info(f"Encoder: use EMA, weight {self.conf.encoder_weight}")
                online_encoder = self._local_model.online_encoder
                ema_updater = model.EMA(self.conf.encoder_weight)
                model.update_moving_average(ema_updater, online_encoder, self.compressed_model.online_encoder)
                target_encoder = self._local_model.target_encoder
            elif self.conf.update_encoder == DYNAMIC_EMA_ONLINE:
                # Use FedEMA to update online encoder
                online_encoder, target_encoder = ema_online()
            elif self.conf.update_encoder == SELECTIVE_EMA:
                # Use FedEMA to update online encoder
                # For random selection, only update with EMA when the client is selected in previous round.
                if self.previous_trained_round + 1 == self.conf.round_id:
                    online_encoder, target_encoder = ema_online()
                else:
                    logger.info(f"Encoder: update online and target @ round {self.conf.round_id}")
                    online_encoder = self.compressed_model.online_encoder
                    target_encoder = self.compressed_model.online_encoder
            else:
                raise ValueError(f"Encoder: aggregate {self.conf.aggregate_encoder}, "
                                 f"update {self.conf.update_encoder} is not supported")

            if self.conf.update_predictor == GLOBAL:
                logger.info("Predictor: use global predictor")
                predictor = self.compressed_model.online_predictor
            elif self.conf.update_predictor == LOCAL:
                logger.info("Predictor: use local predictor")
                predictor = self._local_model.online_predictor
            elif self.conf.update_predictor == DAPU:
                # Divergence-aware predictor update (DAPU)
                logger.info(f"Predictor: use DAPU, mu {self.conf.dapu_threshold}")
                if self.DAPU_predictor == GLOBAL:
                    predictor = self.compressed_model.online_predictor
                elif self.DAPU_predictor == LOCAL:
                    predictor = self._local_model.online_predictor
                else:
                    raise ValueError(f"Predictor: DAPU predictor can either use local or global predictor")
            elif self.conf.update_predictor == DYNAMIC_DAPU:
                # Use FedEMA to update predictor
                predictor = ema_predictor()
            elif self.conf.update_predictor == SELECTIVE_EMA:
                # For random selection, only update with EMA when the client is selected in previous round.
                if self.previous_trained_round + 1 == self.conf.round_id:
                    predictor = ema_predictor()
                else:
                    logger.info("Predictor: use global predictor")
                    predictor = self.compressed_model.online_predictor
            elif self.conf.update_predictor == EMA:
                logger.info(f"Predictor: use EMA, weight {self.conf.predictor_weight}")
                predictor = self._local_model.online_predictor
                ema_updater = model.EMA(self.conf.predictor_weight)
                model.update_moving_average(ema_updater, predictor, self.compressed_model.online_predictor)
            else:
                raise ValueError(f"Predictor: {self.conf.update_predictor} is not supported")

            self.model.online_encoder = copy.deepcopy(online_encoder)
            self.model.target_encoder = copy.deepcopy(target_encoder)
            self.model.online_predictor = copy.deepcopy(predictor)

    def train(self, conf, device=CPU):
        start_time = time.time()
        loss_fn, optimizer = self.pretrain_setup(conf, device)
        if conf.model in [model.MoCo, model.MoCoV2]:
            self.model.reset_key_encoder()
        self.train_loss = []
        self.model.to(device)
        old_model = copy.deepcopy(nn.Sequential(*list(self.model.children())[:-1])).cpu()
        for i in range(conf.local_epoch):
            batch_loss = []
            for (batched_x1, batched_x2), _ in self.train_loader:
                x1, x2 = batched_x1.to(device), batched_x2.to(device)
                optimizer.zero_grad()

                if conf.model in [model.MoCo, model.MoCoV2]:
                    loss = self.model(x1, x2, device)
                elif conf.model == model.SimCLR:
                    images = torch.cat((x1, x2), dim=0)
                    features = self.model(images)
                    logits, labels = self.info_nce_loss(features)
                    loss = loss_fn(logits, labels)
                else:
                    loss = self.model(x1, x2)

                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

                if conf.model in [model.BYOL, model.BYOLNoSG, model.BYOLNoPredictor] and conf.momentum_update:
                    self.model.update_moving_average()

            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
        self.train_time = time.time() - start_time

        # store trained model locally
        self._local_model = copy.deepcopy(self.model).cpu()
        self.previous_trained_round = conf.round_id
        if conf.update_predictor in [DAPU, DYNAMIC_DAPU, SELECTIVE_EMA] or conf.update_encoder in [DYNAMIC_EMA_ONLINE, SELECTIVE_EMA]:
            new_model = copy.deepcopy(nn.Sequential(*list(self.model.children())[:-1])).cpu()
            self.encoder_distance = self._calculate_divergence(old_model, new_model)
            self.encoder_distances.append(self.encoder_distance.item())
            self.DAPU_predictor = self._DAPU_predictor_usage(self.encoder_distance)
            if self.conf.auto_scaler == 'y' and self.conf.random_selection:
                self._calculate_weight_scaler()
            if (conf.round_id + 1) % 100 == 0:
                logger.info(f"Client {self.cid}, encoder distances: {self.encoder_distances}")

    def _DAPU_predictor_usage(self, distance):
        if distance < self.conf.dapu_threshold:
            return GLOBAL
        else:
            return LOCAL

    def _calculate_divergence(self, old_model, new_model, typ=L2):
        size = 0
        total_distance = 0
        old_dict = old_model.state_dict()
        new_dict = new_model.state_dict()
        for name, param in old_model.named_parameters():
            if 'conv' in name and 'weight' in name:
                total_distance += self._calculate_distance(old_dict[name].detach().clone().view(1, -1),
                                                           new_dict[name].detach().clone().view(1, -1),
                                                           typ)
                size += 1
        distance = total_distance / size
        logger.info(f"Model distance: {distance} = {total_distance}/{size}")
        return distance

    def _calculate_distance(self, m1, m2, typ=L2):
        if typ == L2:
            return torch.dist(m1, m2, 2)

    def _calculate_weight_scaler(self):
        if not self.weight_scaler:
            if self.conf.auto_scaler == 'y':
                self.weight_scaler = self.conf.auto_scaler_target / self.encoder_distance
            else:
                self.weight_scaler = self.conf.weight_scaler
            logger.info(f"Client {self.cid}: weight scaler {self.weight_scaler}")

    def load_loader(self, conf):
        drop_last = conf.drop_last
        train_loader = self.train_data.loader(conf.batch_size,
                                              self.cid,
                                              shuffle=True,
                                              drop_last=drop_last,
                                              seed=conf.seed,
                                              transform=self._load_transform(conf))
        _print_label_count(self.cid, self.train_data.data[self.cid]['y'])
        return train_loader

    def load_optimizer(self, conf):
        lr = conf.optimizer.lr
        if conf.optimizer.lr_type == "cosine":
            lr = compute_lr(conf.round_id, conf.rounds, 0, conf.optimizer.lr)

        # movo_v1 should use the default learning rate
        if conf.model == model.MoCo:
            lr = conf.optimizer.lr

        params = self.model.parameters()
        if conf.model in [model.BYOL, model.BYOLNoSG, model.BYOLNoPredictor]:
            params = [
                {'params': self.model.online_encoder.parameters()},
                {'params': self.model.online_predictor.parameters()}
            ]

        if conf.optimizer.type == "Adam":
            optimizer = torch.optim.Adam(params, lr=lr)
        else:
            optimizer = torch.optim.SGD(params,
                                        lr=lr,
                                        momentum=conf.optimizer.momentum,
                                        weight_decay=conf.optimizer.weight_decay)
        return optimizer

    def _load_transform(self, conf):
        transformation = utils.get_transformation(conf.model)
        return transformation(conf.image_size, conf.gaussian)

    def post_upload(self):
        if self.conf.model in [model.BYOL, model.BYOLNoSG, model.BYOLNoPredictor]:
            del self.model
            del self.compressed_model
            self.model = None
            self.compressed_model = None
            assert self.model is None
            assert self.compressed_model is None
            gc.collect()
            torch.cuda.empty_cache()

    def info_nce_loss(self, features, n_views=2, temperature=0.07):
        labels = torch.cat([torch.arange(self.conf.batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     n_views * self.conf.batch_size, n_views * self.conf.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / temperature
        return logits, labels


def compute_lr(current_round, rounds=800, eta_min=0, eta_max=0.3):
    """Compute learning rate as cosine decay"""
    pi = np.pi
    eta_t = eta_min + 0.5 * (eta_max - eta_min) * (np.cos(pi * current_round / rounds) + 1)
    return eta_t


def _print_label_count(cid, labels):
    logger.info(f"client {cid}: {Counter(labels)}")
