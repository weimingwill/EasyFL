import argparse
import concurrent.futures
import copy
import logging
import os
import threading
import time

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from easyfl.communication import grpc_wrapper
from easyfl.datasets import TEST_IN_SERVER
from easyfl.distributed import grouping, reduce_models, reduce_models_only_params, \
    reduce_value, reduce_values, reduce_weighted_values, gather_value
from easyfl.distributed.distributed import CPU, GREEDY_GROUPING
from easyfl.pb import client_service_pb2 as client_pb
from easyfl.pb import common_pb2 as common_pb
from easyfl.protocol import codec
from easyfl.registry.etcd_client import EtcdClient
from easyfl.server import strategies
from easyfl.server.service import ServerService
from easyfl.tracking import metric
from easyfl.tracking.client import init_tracking
from easyfl.utils.float import rounding

logger = logging.getLogger(__name__)

# train and test params
MODEL = "model"
DATA_SIZE = "data_size"
ACCURACY = "accuracy"
LOSS = "loss"
CLIENT_METRICS = "client_metrics"

FEDERATED_AVERAGE = "FedAvg"
EQUAL_AVERAGE = "equal"

AGGREGATION_CONTENT_ALL = "all"
AGGREGATION_CONTENT_PARAMS = "parameters"


def create_argument_parser():
    """Create argument parser with arguments/configurations for starting server service.

    Returns:
        argparse.ArgumentParser: The parser with server service arguments.
    """
    parser = argparse.ArgumentParser(description='Federated Server')
    parser.add_argument('--local-port',
                        type=int,
                        default=22999,
                        help='Listen port of the client')
    parser.add_argument('--tracker-addr',
                        type=str,
                        default="localhost:12666",
                        help='Address of tracking service in [IP]:[PORT] format')
    parser.add_argument('--is-remote',
                        type=bool,
                        default=False,
                        help='Whether start as a remote server.')

    return parser


class BaseServer(object):
    """Default implementation of federated learning server.

    Args:
        conf (omegaconf.dictconfig.DictConfig): Configurations of EasyFL.
        test_data (:obj:`FederatedDataset`): Test dataset for centralized testing in server, optional.
        val_data (:obj:`FederatedDataset`): Validation dataset for centralized validation in server, optional.
        is_remote (bool): A flag to indicate whether start remote training.
        local_port (int): The port of remote server service.

    Override the class and functions to implement customized server.

    Example:
        >>> from easyfl.server import BaseServer
        >>> class CustomizedServer(BaseServer):
        >>>     def __init__(self, conf, test_data=None, val_data=None, is_remote=False, local_port=22999):
        >>>         super(CustomizedServer, self).__init__(conf, test_data, val_data, is_remote, local_port)
        >>>         pass  # more initialization of attributes.
        >>>
        >>>     def aggregation(self):
        >>>         # Implement customized aggregation method, which overwrites the default aggregation method.
        >>>         pass
    """

    def __init__(self,
                 conf,
                 test_data=None,
                 val_data=None,
                 is_remote=False,
                 local_port=22999):
        self.conf = conf
        self.test_data = test_data
        self.val_data = val_data
        self.is_remote = is_remote
        self.local_port = local_port
        self._is_training = False
        self._should_stop = False

        self._current_round = -1
        self._client_uploads = {}
        self._model = None
        self._compressed_model = None
        self._clients = None
        self._etcd = None
        self.selected_clients = []
        self.grouped_clients = []

        self._server_metric = None
        self._round_time = None
        self._begin_train_time = None  # training begin time for a round
        self._start_time = None  # training start time for a task
        self.client_stubs = {}

        if self.conf.is_distributed:
            self.default_time = self.conf.resource_heterogeneous.initial_default_time

        self._cumulative_times = []  # cumulative training after each test
        self._accuracies = []

        self._condition = threading.Condition()

        self._tracker = None
        self.init_tracker()

    def start(self, model, clients):
        """Start federated learning process, including training and testing.

        Args:
            model (nn.Module): The model to train.
            clients (list[:obj:`BaseClient`]|list[str]): Available clients.
                Clients are actually client grpc addresses when in remote training.
        """
        # Setup
        self._start_time = time.time()
        self._reset()
        self.set_model(model)
        self.set_clients(clients)

        if self._should_track():
            self._tracker.create_task(self.conf.task_id, OmegaConf.to_container(self.conf))

        # Get initial testing accuracies
        if self.conf.server.test_all:
            if self._should_track():
                self._tracker.set_round(self._current_round)
            self.test()
            self.save_tracker()

        while not self.should_stop():
            self._round_time = time.time()

            self._current_round += 1
            self.print_("\n-------- round {} --------".format(self._current_round))

            # Train
            self.pre_train()
            self.train()
            self.post_train()

            # Test
            if self._do_every(self.conf.server.test_every, self._current_round, self.conf.server.rounds):
                self.pre_test()
                self.test()
                self.post_test()

            # Save Model
            self.save_model()

            self.track(metric.ROUND_TIME, time.time() - self._round_time)
            self.save_tracker()

        self.print_("Accuracies: {}".format(rounding(self._accuracies, 4)))
        self.print_("Cumulative training time: {}".format(rounding(self._cumulative_times, 2)))

    def stop(self):
        """Set the flag to indicate training should stop."""
        self._should_stop = True

    def pre_train(self):
        """Preprocessing before training."""
        pass

    def train(self):
        """Training process of federated learning."""
        self.print_("--- start training ---")

        self.selection(self._clients, self.conf.server.clients_per_round)
        self.grouping_for_distributed()
        self.compression()

        begin_train_time = time.time()

        self.distribution_to_train()
        self.aggregation()

        train_time = time.time() - begin_train_time
        self.print_("Server train time: {}".format(train_time))
        self.track(metric.TRAIN_TIME, train_time)

    def post_train(self):
        """Postprocessing after training."""
        pass

    def pre_test(self):
        """Preprocessing before testing."""
        pass

    def test(self):
        """Testing process of federated learning."""
        self.print_("--- start testing ---")

        test_begin_time = time.time()

        test_results = {metric.TEST_ACCURACY: 0, metric.TEST_LOSS: 0, metric.TEST_TIME: 0}
        if self.conf.test_mode == TEST_IN_SERVER:
            if self.is_primary_server():
                test_results = self.test_in_server(self.conf.device)
        else:
            test_results = self.test_in_client()

        test_results[metric.TEST_TIME] = time.time() - test_begin_time
        self.track_test_results(test_results)

    def post_test(self):
        """Postprocessing after testing."""
        pass

    def should_stop(self):
        """Check whether should stop training. Stops the training under two conditions:
        1. Reach max number of training rounds
        2. TODO: Accuracy higher than certain amount.

        Returns:
            bool: A flag to indicate whether should stop training.
        """
        if self._should_stop or (self.conf.server.rounds and self._current_round + 1 >= self.conf.server.rounds):
            self._is_training = False
            return True
        return False

    def test_in_client(self):
        """Conduct testing in clients.
        Currently, it supports testing on the selected clients for training.
        TODO: Add optionals to select clients for testing.

        Returns:
            dict: Test metrics, {"test_loss": value, "test_accuracy": value, "test_time": value}.
        """
        self.compression()
        self.distribution_to_test()
        return self.aggregation_test()

    def test_in_server(self, device=CPU):
        """Conduct testing in the server.

        Args:
            device (str): The hardware device to conduct testing, either cpu or cuda devices.

        Returns:
            dict: Test metrics, {"test_loss": value, "test_accuracy": value, "test_time": value}.
        """
        self._model.eval()
        self._model.to(device)
        test_loss = 0
        correct = 0
        loss_fn = torch.nn.CrossEntropyLoss().to(device)
        with torch.no_grad():
            for batched_x, batched_y in self.test_data.loader(self.conf.server.batch_size, seed=self.conf.seed):
                x = batched_x.to(device)
                y = batched_y.to(device)
                log_probs = self._model(x)
                loss = loss_fn(log_probs, y)
                _, y_pred = torch.max(log_probs, -1)
                correct += y_pred.eq(y.data.view_as(y_pred)).long().cpu().sum()
                test_loss += loss.item()
            test_data_size = self.test_data.size()
            test_loss /= test_data_size
            accuracy = 100.00 * correct / test_data_size

            test_results = {
                metric.TEST_ACCURACY: float(accuracy),
                metric.TEST_LOSS: float(test_loss)
            }
            return test_results

    def selection(self, clients, clients_per_round):
        """Select a fraction of total clients for training.
        Two selection strategies are implemented: 1. random selection; 2. select the first K clients.

        Args:
            clients (list[:obj:`BaseClient`]|list[str]): Available clients.
            clients_per_round (int): Number of clients to participate in training each round.

        Returns:
            (list[:obj:`BaseClient`]|list[str]): The selected clients.
        """
        if clients_per_round > len(clients):
            logger.warning("Available clients for selection are smaller than required clients for each round")

        clients_per_round = min(clients_per_round, len(clients))
        if self.conf.server.random_selection:
            np.random.seed(self._current_round)
            self.selected_clients = np.random.choice(clients, clients_per_round, replace=False)
        else:
            self.selected_clients = clients[:clients_per_round]

        return self.selected_clients

    def grouping_for_distributed(self):
        """Divide the selected clients into groups for distributed training.
        Each group of clients is assigned to conduct training in one GPU. The number of groups = the number of gpus.

        Not in distributed training, selected clients are in the same group.
        In distributed, selected clients are grouped with different strategies: greedy and random.
        """
        if self.conf.is_distributed:
            groups = grouping(self.selected_clients,
                              self.conf.distributed.world_size,
                              self.default_time,
                              self.conf.resource_heterogeneous.grouping_strategy,
                              self._current_round)
            # assign a group for each rank to train with current device.
            self.grouped_clients = groups[self.conf.distributed.rank]
            grouping_info = [(c.cid, c.round_time) for c in self.grouped_clients]
            logger.info("Grouping Result for rank {}: {}".format(self.conf.distributed.rank, grouping_info))
        else:
            self.grouped_clients = self.selected_clients

        rank = 0 if len(self.grouped_clients) == len(self.selected_clients) else self.conf.distributed.rank

    def compression(self):
        """Model compression to reduce communication cost."""
        self._compressed_model = self._model

    def distribution_to_train(self):
        """Distribute model and configurations to selected clients to train."""
        if self.is_remote:
            self.distribution_to_train_remotely()
        else:
            self.distribution_to_train_locally()

            # Adaptively update the training time of clients for greedy grouping.
            if self.conf.is_distributed and self.conf.resource_heterogeneous.grouping_strategy == GREEDY_GROUPING:
                self.profile_training_speed()
                self.update_default_time()

    def distribution_to_train_locally(self):
        """Conduct training sequentially for selected clients in the group."""
        uploaded_models = {}
        uploaded_weights = {}
        uploaded_metrics = []
        for client in self.grouped_clients:
            # Update client config before training
            self.conf.client.task_id = self.conf.task_id
            self.conf.client.round_id = self._current_round

            uploaded_request = client.run_train(self._compressed_model, self.conf.client)
            uploaded_content = uploaded_request.content

            model = self.decompression(codec.unmarshal(uploaded_content.data))
            uploaded_models[client.cid] = model
            uploaded_weights[client.cid] = uploaded_content.data_size
            uploaded_metrics.append(metric.ClientMetric.from_proto(uploaded_content.metric))

        self.set_client_uploads_train(uploaded_models, uploaded_weights, uploaded_metrics)

    def distribution_to_train_remotely(self):
        """Distribute training requests to remote clients through multiple threads.
        The main thread waits for signal to proceed. The signal can be triggered via notification, as below example.

        Example to trigger signal:
            >>> with self.condition():
            >>>     self.notify_all()
        """
        start_time = time.time()
        should_track = self._tracker is not None and self.conf.client.track
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for client in self.grouped_clients:
                request = client_pb.OperateRequest(
                    type=client_pb.OP_TYPE_TRAIN,
                    model=codec.marshal(self._compressed_model),
                    data_index=client.index,
                    config=client_pb.OperateConfig(
                        batch_size=self.conf.client.batch_size,
                        local_epoch=self.conf.client.local_epoch,
                        seed=self.conf.seed,
                        local_test=self.conf.client.local_test,
                        optimizer=client_pb.Optimizer(
                            type=self.conf.client.optimizer.type,
                            lr=self.conf.client.optimizer.lr,
                            momentum=self.conf.client.optimizer.momentum,
                        ),
                        task_id=self.conf.task_id,
                        round_id=self._current_round,
                        track=should_track,
                    ),
                )
                executor.submit(self._distribution_remotely, client.client_id, request)

            distribute_time = time.time() - start_time
            self.track(metric.TRAIN_DISTRIBUTE_TIME, distribute_time)
            logger.info("Distribute to clients, time: {}".format(distribute_time))
        with self._condition:
            self._condition.wait()

    def distribution_to_test(self):
        """Distribute to conduct testing on clients."""
        if self.is_remote:
            self.distribution_to_test_remotely()
        else:
            self.distribution_to_test_locally()

    def distribution_to_test_locally(self):
        """Conduct testing sequentially for selected testing clients."""
        uploaded_accuracies = []
        uploaded_losses = []
        uploaded_data_sizes = []
        uploaded_metrics = []

        test_clients = self.get_test_clients()
        for client in test_clients:
            # Update client config before testing
            self.conf.client.task_id = self.conf.task_id
            self.conf.client.round_id = self._current_round

            uploaded_request = client.run_test(self._compressed_model, self.conf.client)
            uploaded_content = uploaded_request.content
            performance = codec.unmarshal(uploaded_content.data)

            uploaded_accuracies.append(performance.accuracy)
            uploaded_losses.append(performance.loss)
            uploaded_data_sizes.append(uploaded_content.data_size)
            uploaded_metrics.append(metric.ClientMetric.from_proto(uploaded_content.metric))

        self.set_client_uploads_test(uploaded_accuracies, uploaded_losses, uploaded_data_sizes, uploaded_metrics)

    def distribution_to_test_remotely(self):
        """Distribute testing requests to remote clients through multiple threads.
        The main thread waits for signal to proceed. The signal can be triggered via notification, as below example.

        Example to trigger signal:
            >>> with self.condition():
            >>>     self.notify_all()
        """
        start_time = time.time()
        should_track = self._tracker is not None and self.conf.client.track
        test_clients = self.get_test_clients()
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for client in test_clients:
                request = client_pb.OperateRequest(
                    type=client_pb.OP_TYPE_TEST,
                    model=codec.marshal(self._compressed_model),
                    data_index=client.index,
                    config=client_pb.OperateConfig(
                        batch_size=self.conf.client.batch_size,
                        test_batch_size=self.conf.client.test_batch_size,
                        seed=self.conf.seed,
                        task_id=self.conf.task_id,
                        round_id=self._current_round,
                        track=should_track,
                    )
                )
                executor.submit(self._distribution_remotely, client.client_id, request)

            distribute_time = time.time() - start_time
            self.track(metric.TEST_DISTRIBUTE_TIME, distribute_time)
            logger.info("Distribute to test clients, time: {}".format(distribute_time))
        with self._condition:
            self._condition.wait()

    def get_test_clients(self):
        """Get clients to run testing.

        Returns:
            (list[:obj:`BaseClient`]|list[str]): Clients to test.
        """
        if self.conf.server.test_all:
            if self.conf.is_distributed:
                # Group and assign clients to different hardware devices to test.
                test_clients = grouping(self._clients,
                                        self.conf.distributed.world_size,
                                        default_time=self.default_time,
                                        strategy=self.conf.resource_heterogeneous.grouping_strategy)
                test_clients = test_clients[self.conf.distributed.rank]
            else:
                test_clients = self._clients
        else:
            # For the initial testing, if no clients are selected, test all clients
            test_clients = self.grouped_clients if self.grouped_clients is not None else self._clients
        return test_clients

    def _distribution_remotely(self, cid, request):
        """Distribute request to the assigned client to conduct operations.

        Args:
            cid (str): Client id.
            request (:obj:`OperateRequest`): gRPC request of specific operations.
        """
        resp = self.client_stubs[cid].Operate(request)
        if resp.status.code != common_pb.SC_OK:
            logger.error("Failed to train/test in client {}, error: {}".format(cid, resp.status.message))
        else:
            logger.info("Distribute to train/test remotely successfully, client: {}".format(cid))

    def aggregation_test(self):
        """Aggregate testing results from clients.

        Returns:
            dict: Test metrics, format in {"test_loss": value, "test_accuracy": value}
        """
        accuracies = self._client_uploads[ACCURACY]
        losses = self._client_uploads[LOSS]
        test_sizes = self._client_uploads[DATA_SIZE]

        if self.conf.test_method == "average":
            loss = self._mean_value(losses)
            accuracy = self._mean_value(accuracies)
        elif self.conf.test_method == "weighted":
            loss = self._weighted_value(losses, test_sizes)
            accuracy = self._weighted_value(accuracies, test_sizes)
        else:
            raise ValueError("test_method not supported, please use average or weighted")

        test_results = {
            metric.TEST_ACCURACY: float(accuracy),
            metric.TEST_LOSS: float(loss)
        }
        return test_results

    def _mean_value(self, values):
        if self.conf.is_distributed:
            return reduce_values(values, self.conf.device)
        else:
            return np.mean(values)

    def _weighted_value(self, values, weights):
        if self.conf.is_distributed:
            return reduce_weighted_values(values, weights, self.conf.device)
        else:
            return np.average(values, weights=weights)

    def decompression(self, model):
        """Decompression the models from clients"""
        return model

    def aggregation(self):
        """Aggregate training updates from clients.
        Server aggregates trained models from clients via federated averaging.
        """
        uploaded_content = self.get_client_uploads()
        models = list(uploaded_content[MODEL].values())
        weights = list(uploaded_content[DATA_SIZE].values())

        model = self.aggregate(models, weights)
        self.set_model(model, load_dict=True)

    def aggregate(self, models, weights):
        """Aggregate models uploaded from clients via federated averaging.

        Args:
            models (list[nn.Module]): List of models.
            weights (list[float]): List of weights, corresponding to each model.
                Weights are dataset size of clients by default.
        Returns
            nn.Module: Aggregated model.
        """
        if self.conf.server.aggregation_strategy == EQUAL_AVERAGE:
            weights = [1 for _ in range(len(models))]

        fn_average = strategies.federated_averaging
        fn_sum = strategies.weighted_sum
        fn_reduce = reduce_models
        if self.conf.server.aggregation_content == AGGREGATION_CONTENT_PARAMS:
            fn_average = strategies.federated_averaging_only_params
            fn_sum = strategies.weighted_sum_only_params
            fn_reduce = reduce_models_only_params

        if self.conf.is_distributed:
            dist.barrier()
            model, sample_sum = fn_sum(models, weights)
            fn_reduce(model, torch.tensor(sample_sum).to(self.conf.device))
        else:
            model = fn_average(models, weights)
        return model

    def _reset(self):
        self._current_round = -1
        self._should_stop = False
        self._is_training = True

    def is_training(self):
        """Check whether the server is in training or has stopped training.

        Returns:
            bool: A flag to indicate whether server is in training.
        """
        return self._is_training

    def set_model(self, model, load_dict=False):
        """Update the universal model in the server.

        Args:
            model (nn.Module): New model.
            load_dict (bool): A flag to indicate whether load state dict or copy the model.
        """
        if load_dict:
            self._model.load_state_dict(model.state_dict())
        else:
            self._model = copy.deepcopy(model)

    def set_clients(self, clients):
        self._clients = clients

    def num_of_clients(self):
        return len(self._clients)

    def save_model(self):
        """Save the model in the server."""
        if self._do_every(self.conf.server.save_model_every, self._current_round, self.conf.server.rounds) and \
                self.is_primary_server():
            save_path = self.conf.server.save_model_path
            if save_path == "":
                save_path = os.path.join(os.getcwd(), "saved_models")
            os.makedirs(save_path, exist_ok=True)
            save_path = os.path.join(save_path,
                                     "{}_global_model_r_{}.pth".format(self.conf.task_id, self._current_round))
            torch.save(self._model.cpu().state_dict(), save_path)
            self.print_("Model saved at {}".format(save_path))

    def set_client_uploads_train(self, models, weights, metrics=None):
        """Set training updates uploaded from clients.

        Args:
            models (dict): A collection of models.
            weights (dict): A collection of weights.
            metrics (dict): Client training metrics.
        """
        self.set_client_uploads(MODEL, models)
        self.set_client_uploads(DATA_SIZE, weights)
        if self._should_gather_metrics():
            metrics = self.gather_client_train_metrics()
        self.set_client_uploads(CLIENT_METRICS, metrics)

    def set_client_uploads_test(self, accuracies, losses, test_sizes, metrics=None):
        """Set testing results uploaded from clients.

        Args:
            accuracies (list[float]): Testing accuracies of clients.
            losses (list[float]): Testing losses of clients.
            test_sizes (list[float]): Test dataset sizes of clients.
            metrics (dict): Client testing metrics.
        """
        self.set_client_uploads(ACCURACY, accuracies)
        self.set_client_uploads(LOSS, losses)
        self.set_client_uploads(DATA_SIZE, test_sizes)
        if self._should_gather_metrics() and CLIENT_METRICS in self._client_uploads:
            train_metrics = self.get_client_uploads()[CLIENT_METRICS]
            metrics = metric.ClientMetric.merge_train_to_test_metrics(train_metrics, metrics)
        self.set_client_uploads(CLIENT_METRICS, metrics)

    def set_client_uploads(self, key, value):
        """A general function to set uploaded content from clients.

        Args:
            key (str): Dictionary key.
            value (*): Uploaded content.
        """
        self._client_uploads[key] = value

    def get_client_uploads(self):
        """Get client uploaded contents.

        Returns:
            dict: A dictionary that contains client uploaded contents.
        """
        return self._client_uploads

    def _do_every(self, every, current_round, rounds):
        return (current_round + 1) % every == 0 or (current_round + 1) == rounds

    def print_(self, content):
        """print only the server is primary server.

        Args:
            content (str): The content to log.
        """
        if self.is_primary_server():
            logger.info(content)

    def is_primary_server(self):
        """Check whether the current process is the primary server.
        In standalone or remote training, the server is primary.
        In distributed training, the server on rank0 is primary.

        Returns:
            bool: A flag to indicate whether current process is the primary server.
        """
        return not self.conf.is_distributed or self.conf.distributed.rank == 0

    # Functions for remote training

    def start_service(self):
        """Start federated learning server GRPC service."""
        if self.is_remote:
            grpc_wrapper.start_service(grpc_wrapper.TYPE_SERVER, ServerService(self), self.local_port)
            logger.info("GRPC server started at :{}".format(self.local_port))

    def connect_remote_clients(self, clients):
        # TODO: This client should be consistent with client started separately.
        for client in clients:
            if client.client_id not in self.client_stubs:
                self.client_stubs[client.client_id] = grpc_wrapper.init_stub(grpc_wrapper.TYPE_CLIENT, client.address)
                logger.info("Successfully connected to gRPC client {}".format(client.address))

    def init_etcd(self, addresses):
        """Initialize etcd as the registry for client registration.

        Args:
            addresses (str): The etcd addresses split by ","
        """
        self._etcd = EtcdClient("server", addresses, "backends")

    def start_remote_training(self, model, clients):
        """Start federated learning in the remote training mode.
        Server establishes gPRC connection with clients that are not connected first before training.

        Args:
            model (nn.Module): The model to train.
            clients (list[str]): Client addresses.
        """
        self.connect_remote_clients(clients)
        self.start(model, clients)

    # Functions for tracking

    def init_tracker(self):
        """Initialize tracking"""
        if self.conf.server.track:
            self._tracker = init_tracking(self.conf.tracking.database, self.conf.tracker_addr)

    def track(self, metric_name, value):
        """Track a metric.

        Args:
            metric_name (str): Name of the metric of a round.
            value (str|int|float|bool|dict|list): Value of the metric.
        """
        if not self._should_track():
            return
        self._tracker.track_round(metric_name, value)

    def track_test_results(self, results):
        """Track test results collected from clients.

        Args:
            results (dict): Test metrics, format in {"test_loss": value, "test_accuracy": value, "test_time": value}
        """
        self._cumulative_times.append(time.time() - self._start_time)
        self._accuracies.append(results[metric.TEST_ACCURACY])

        for metric_name in results:
            self.track(metric_name, results[metric_name])

        self.print_('Test time {:.2f}s, Test loss: {:.2f}, Test accuracy: {:.2f}%'.format(
            results[metric.TEST_TIME], results[metric.TEST_LOSS], results[metric.TEST_ACCURACY]))

    def save_tracker(self):
        """Save metrics in the tracker to database."""
        if self._tracker:
            self.track_communication_cost()
            if self.is_primary_server():
                self._tracker.save_round()
            # In distributed training, each server saves their clients separately.
            self._tracker.save_clients(self._client_uploads[CLIENT_METRICS])

    def track_communication_cost(self):
        """Track communication cost among server and clients.
        Communication cost occurs in `training` and `testing` with downlink and uplink costs.
        """
        train_upload_size = 0
        train_download_size = 0
        test_upload_size = 0
        test_download_size = 0
        for client_metric in self._client_uploads[CLIENT_METRICS]:
            if client_metric.round_id == self._current_round and client_metric.task_id == self.conf.task_id:
                train_upload_size += client_metric.train_upload_size
                train_download_size += client_metric.train_download_size
                test_upload_size += client_metric.test_upload_size
                test_download_size += client_metric.test_download_size
        if self.conf.is_distributed:
            train_upload_size = reduce_value(train_upload_size, self.conf.device).item()
            train_download_size = reduce_value(train_download_size, self.conf.device).item()
            test_upload_size = reduce_value(test_upload_size, self.conf.device).item()
            test_download_size = reduce_value(test_download_size, self.conf.device).item()
        self._tracker.track_round(metric.TRAIN_UPLOAD_SIZE, train_upload_size)
        self._tracker.track_round(metric.TRAIN_DOWNLOAD_SIZE, train_download_size)
        self._tracker.track_round(metric.TEST_UPLOAD_SIZE, test_upload_size)
        self._tracker.track_round(metric.TEST_DOWNLOAD_SIZE, test_download_size)

    def _should_track(self):
        """Check whether server should track metrics.
        Server tracks metrics only when tracking is enabled and it is the primary server.

        Returns:
            bool: A flag indicate whether server should track metrics.
        """
        return self._tracker is not None and self.is_primary_server()

    def _should_gather_metrics(self):
        """Check whether the server should gather metrics from GPUs.
        Gather metrics only when testing all in `distributed` training.

        Testing all resets clients' training metrics, thus,
        server needs to gather train metrics to construct full client metrics.

        Returns:
            bool: A flag indicate whether server should gather metrics.
        """
        return self.conf.is_distributed and self.conf.server.test_all and self._tracker

    def gather_client_train_metrics(self):
        """Gather client train metrics from other ranks for distributed training, when testing all clients (test_all).
        When testing all clients, the trained metrics may be override by the test metrics
        because clients may be placed in different GPUs in training and testing, leading to losses of train metrics.
        So we gather train metrics and set them in test metrics.
        TODO: gather is not progressing. Need fix.
        """
        world_size = self.conf.distributed.world_size
        device = self.conf.device
        uploads = self.get_client_uploads()
        client_id_list = []
        train_accuracy_list = []
        train_loss_list = []
        train_time_list = []
        train_upload_time_list = []
        train_upload_size_list = []
        train_download_size_list = []

        for m in uploads[CLIENT_METRICS]:
            # client_id_list += gather_value(m.client_id, world_size, device).tolist()
            train_accuracy_list += gather_value(m.train_accuracy, world_size, device)
            train_loss_list += gather_value(m.train_loss, world_size, device)
            train_time_list += gather_value(m.train_time, world_size, device)
            train_upload_time_list += gather_value(m.train_upload_time, world_size, device)
            train_upload_size_list += gather_value(m.train_upload_size, world_size, device)
            train_download_size_list += gather_value(m.train_download_size, world_size, device)
        metrics = []
        # Note: Client id may not match with its training stats because all_gather string is not supported.
        client_id_list = [c.cid for c in self.selected_clients]
        for i, client_id in enumerate(client_id_list):
            m = metric.ClientMetric(self.conf.task_id, self._current_round, client_id)
            m.add(metric.TRAIN_ACCURACY, train_accuracy_list[i])
            m.add(metric.TRAIN_LOSS, train_loss_list[i])
            m.add(metric.TRAIN_TIME, train_time_list[i])
            m.add(metric.TRAIN_UPLOAD_TIME, train_upload_time_list[i])
            m.add(metric.TRAIN_UPLOAD_SIZE, train_upload_size_list[i])
            m.add(metric.TRAIN_DOWNLOAD_SIZE, train_download_size_list[i])
            metrics.append(m)
        return metrics

    # Functions for remote training.

    def condition(self):
        return self._condition

    def notify_all(self):
        self._condition.notify_all()

    # Functions for distributed training optimization.

    def profile_training_speed(self):
        """Manage profiling of client training speeds for distributed training optimization."""
        profile_required = []
        for client in self.selected_clients:
            if not client.profiled:
                profile_required.append(client)
        if len(profile_required) > 0:
            original = torch.FloatTensor([c.round_time for c in profile_required]).to(self.conf.device)
            time_update = torch.FloatTensor([c.train_time for c in profile_required]).to(self.conf.device)
            dist.barrier()
            dist.all_reduce(time_update)
            for i in range(len(profile_required)):
                old_round_time = original[i]
                current_round_time = time_update[i]
                if old_round_time == 0 or self._should_update_round_time(old_round_time, current_round_time):
                    profile_required[i].round_time = float(current_round_time)
                    profile_required[i].train_time = 0
                else:
                    profile_required[i].profiled = True

    def update_default_time(self):
        """Update the estimated default training time of clients using actual training time from profiled clients."""
        default_momentum = self.conf.resource_heterogeneous.default_time_momentum
        current_round_average = np.mean([float(c.round_time) for c in self.selected_clients])
        self.default_time = default_momentum * current_round_average + self.default_time * (1 - default_momentum)

    def _should_update_round_time(self, old_round_time, new_round_time, threshold=0.3):
        """Check whether assign a new estimated round time to client or set it to profiled.

        Args:
            old_round_time (float): previous estimated round time.
            new_round_time (float): Currently profiled round time.
            threshold (float): Tolerance threshold of difference between old and new times.
        Returns:
            bool: A flag to indicate whether should update round time.
        """
        if new_round_time < old_round_time:
            return ((old_round_time - new_round_time) / new_round_time) >= threshold
        else:
            return ((new_round_time - old_round_time) / old_round_time) >= threshold
