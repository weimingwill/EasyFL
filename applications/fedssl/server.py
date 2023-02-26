import copy
import logging
import os

import torch
import torch.distributed as dist
from torchvision import datasets

import model
import utils
from communication import TARGET
from easyfl.datasets.data import CIFAR100
from easyfl.distributed import reduce_models
from easyfl.distributed.distributed import CPU
from easyfl.server import strategies
from easyfl.server.base import BaseServer, MODEL, DATA_SIZE
from easyfl.tracking import metric
from knn_monitor import knn_monitor

logger = logging.getLogger(__name__)


class FedSSLServer(BaseServer):
    def __init__(self, conf, test_data=None, val_data=None, is_remote=False, local_port=22999):
        super(FedSSLServer, self).__init__(conf, test_data, val_data, is_remote, local_port)
        self.train_loader = None
        self.test_loader = None

    def aggregation(self):
        if self.conf.client.auto_scaler == 'y' and self.conf.server.random_selection:
            self._retain_weight_scaler()

        uploaded_content = self.get_client_uploads()
        models = list(uploaded_content[MODEL].values())
        weights = list(uploaded_content[DATA_SIZE].values())

        # Aggregate networks gradually with different components.
        if self.conf.model in [model.Symmetric, model.SymmetricNoSG, model.SimSiam, model.SimSiamNoSG, model.BYOL,
                               model.BYOLNoSG, model.BYOLNoPredictor, model.SimCLR]:
            online_encoders = [m.online_encoder for m in models]
            online_encoder = self._federated_averaging(online_encoders, weights)
            self._model.online_encoder.load_state_dict(online_encoder.state_dict())

        if self.conf.model in [model.SimSiam, model.SimSiamNoSG, model.BYOL, model.BYOLNoSG, model.BYOLNoPredictor]:
            predictors = [m.online_predictor for m in models]
            predictor = self._federated_averaging(predictors, weights)
            self._model.online_predictor.load_state_dict(predictor.state_dict())

        if self.conf.model in [model.BYOL, model.BYOLNoSG, model.BYOLNoPredictor]:
            target_encoders = [m.target_encoder for m in models]
            target_encoder = self._federated_averaging(target_encoders, weights)
            self._model.target_encoder = copy.deepcopy(target_encoder)

        if self.conf.model in [model.MoCo, model.MoCoV2]:
            encoder_qs = [m.encoder_q for m in models]
            encoder_q = self._federated_averaging(encoder_qs, weights)
            self._model.encoder_q.load_state_dict(encoder_q.state_dict())

            encoder_ks = [m.encoder_k for m in models]
            encoder_k = self._federated_averaging(encoder_ks, weights)
            self._model.encoder_k.load_state_dict(encoder_k.state_dict())

    def _retain_weight_scaler(self):
        self.client_id_to_index = {c.cid: i for i, c in enumerate(self._clients)}

        client_index = self.client_id_to_index[self.grouped_clients[0].cid]
        weight_scaler = self.grouped_clients[0].weight_scaler if self.grouped_clients[0].weight_scaler else 0
        scaler = torch.tensor((client_index, weight_scaler)).to(self.conf.device)
        scalers = [torch.zeros_like(scaler) for _ in self.selected_clients]
        dist.barrier()
        dist.all_gather(scalers, scaler)

        logger.info(f"Synced scaler {scalers}")
        for i, client in enumerate(self._clients):
            for scaler in scalers:
                scaler = scaler.cpu().numpy()
                if self.client_id_to_index[client.cid] == int(scaler[0]) and not client.weight_scaler:
                    self._clients[i].weight_scaler = scaler[1]

    def _federated_averaging(self, models, weights):
        fn_average = strategies.federated_averaging
        fn_sum = strategies.weighted_sum
        fn_reduce = reduce_models

        if self.conf.is_distributed:
            dist.barrier()
            model_, sample_sum = fn_sum(models, weights)
            fn_reduce(model_, torch.tensor(sample_sum).to(self.conf.device))
        else:
            model_ = fn_average(models, weights)
        return model_

    def test_in_server(self, device=CPU):
        testing_model = self._get_testing_model()
        testing_model.eval()
        testing_model.to(device)

        self._get_test_data()

        with torch.no_grad():
            accuracy = knn_monitor(testing_model, self.train_loader, self.test_loader, device=device)

        test_results = {
            metric.TEST_ACCURACY: float(accuracy),
            metric.TEST_LOSS: 0,
        }
        return test_results

    def _get_test_data(self):
        transformation = self._load_transform()
        if self.train_loader is None or self.test_loader is None:
            if self.conf.data.dataset == CIFAR100:
                data_path = "./data/cifar100"
                train_dataset = datasets.CIFAR100(data_path, download=True, transform=transformation)
                test_dataset = datasets.CIFAR100(data_path, train=False, download=True, transform=transformation)
            else:
                data_path = "./data/cifar10"
                train_dataset = datasets.CIFAR10(data_path, download=True, transform=transformation)
                test_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=transformation)

            if self.train_loader is None:
                self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, num_workers=8)

            if self.test_loader is None:
                self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, num_workers=8)

    def _load_transform(self):
        transformation = utils.get_transformation(self.conf.model)
        return transformation().test_transform

    def _get_testing_model(self, net=False):
        if self.conf.model in [model.MoCo, model.MoCoV2]:
            testing_model = self._model.encoder_q
        elif self.conf.model in [model.SimSiam, model.SimSiamNoSG, model.Symmetric, model.SymmetricNoSG, model.SimCLR]:
            testing_model = self._model.online_encoder
        else:
            # BYOL
            if self.conf.client.aggregate_encoder == TARGET:
                self.print_("Use aggregated target encoder for testing")
                testing_model = self._model.target_encoder
            else:
                self.print_("Use aggregated online encoder for testing")
                testing_model = self._model.online_encoder
        return testing_model

    def save_model(self):
        if self._do_every(self.conf.server.save_model_every, self._current_round, self.conf.server.rounds) and self.is_primary_server():
            save_path = self.conf.server.save_model_path
            if save_path == "":
                save_path = os.path.join(os.getcwd(), "saved_models", self.conf.task_id)
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path,
                                     "{}_global_model_r_{}.pth".format(self.conf.task_id, self._current_round))

            torch.save(self._get_testing_model().cpu().state_dict(), save_path)
            self.print_("Encoder model saved at {}".format(save_path))

            if self.conf.server.save_predictor:
                if self.conf.model in [model.SimSiam, model.BYOL]:
                    save_path = save_path.replace("global_model", "predictor")
                    torch.save(self._model.online_predictor.cpu().state_dict(), save_path)
                    self.print_("Predictor model saved at {}".format(save_path))
