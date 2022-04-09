import argparse
import copy
import logging
import time

import torch

from easyfl.client.service import ClientService
from easyfl.communication import grpc_wrapper
from easyfl.distributed.distributed import CPU
from easyfl.pb import common_pb2 as common_pb
from easyfl.pb import server_service_pb2 as server_pb
from easyfl.protocol import codec
from easyfl.tracking import metric
from easyfl.tracking.client import init_tracking
from easyfl.tracking.evaluation import model_size

logger = logging.getLogger(__name__)


def create_argument_parser():
    """Create argument parser with arguments/configurations for starting remote client service.

    Returns:
        argparse.ArgumentParser: Parser with client service arguments.
    """
    parser = argparse.ArgumentParser(description='Federated Client')
    parser.add_argument('--local-port',
                        type=int,
                        default=23000,
                        help='Listen port of the client')
    parser.add_argument('--server-addr',
                        type=str,
                        default="localhost:22999",
                        help='Address of server in [IP]:[PORT] format')
    parser.add_argument('--tracker-addr',
                        type=str,
                        default="localhost:12666",
                        help='Address of tracking service in [IP]:[PORT] format')
    parser.add_argument('--is-remote',
                        type=bool,
                        default=False,
                        help='Whether start as a remote client.')
    return parser


class BaseClient(object):
    """Default implementation of federated learning client.

    Args:
        cid (str): Client id.
        conf (omegaconf.dictconfig.DictConfig): Client configurations.
        train_data (:obj:`FederatedDataset`): Training dataset.
        test_data (:obj:`FederatedDataset`): Test dataset.
        device (str): Hardware device for training, cpu or cuda devices.
        sleep_time (float): Duration of on hold after training to simulate stragglers.
        is_remote (bool): Whether start remote training.
        local_port (int): Port of remote client service.
        server_addr (str): Remote server service grpc address.
        tracker_addr (str): Remote tracking service grpc address.


    Override the class and functions to implement customized client.

    Example:
        >>> from easyfl.client import BaseClient
        >>> class CustomizedClient(BaseClient):
        >>>     def __init__(self, cid, conf, train_data, test_data, device, **kwargs):
        >>>         super(CustomizedClient, self).__init__(cid, conf, train_data, test_data, device, **kwargs)
        >>>         pass  # more initialization of attributes.
        >>>
        >>>     def train(self, conf, device=CPU):
        >>>         # Implement customized client training method, which overwrites the default training method.
        >>>         pass
    """
    def __init__(self,
                 cid,
                 conf,
                 train_data,
                 test_data,
                 device,
                 sleep_time=0,
                 is_remote=False,
                 local_port=23000,
                 server_addr="localhost:22999",
                 tracker_addr="localhost:12666"):
        self.cid = cid
        self.conf = conf
        self.train_data = train_data
        self.train_loader = None
        self.test_data = test_data
        self.test_loader = None
        self.device = device

        self.round_time = 0
        self.train_time = 0
        self.test_time = 0

        self.train_accuracy = []
        self.train_loss = []
        self.test_accuracy = 0
        self.test_loss = 0

        self.profiled = False
        self._sleep_time = sleep_time

        self.compressed_model = None
        self.model = None
        self._upload_holder = server_pb.UploadContent()

        self.is_remote = is_remote
        self.local_port = local_port
        self._server_addr = server_addr
        self._tracker_addr = tracker_addr
        self._server_stub = None
        self._tracker = None
        self._is_train = True

        if conf.track:
            self._tracker = init_tracking(init_store=False)

    def run_train(self, model, conf):
        """Conduct training on clients.

        Args:
            model (nn.Module): Model to train.
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
        Returns:
            :obj:`UploadRequest`: Training contents. Unify the interface for both local and remote operations.
        """
        self.conf = conf
        if conf.track:
            self._tracker.set_client_context(conf.task_id, conf.round_id, self.cid)

        self._is_train = True

        self.download(model)
        self.track(metric.TRAIN_DOWNLOAD_SIZE, model_size(model))

        self.decompression()

        self.pre_train()
        self.train(conf, self.device)
        self.post_train()

        self.track(metric.TRAIN_ACCURACY, self.train_accuracy)
        self.track(metric.TRAIN_LOSS, self.train_loss)
        self.track(metric.TRAIN_TIME, self.train_time)

        if conf.local_test:
            self.test_local()

        self.compression()

        self.track(metric.TRAIN_UPLOAD_SIZE, model_size(self.compressed_model))

        self.encryption()

        return self.upload()

    def run_test(self, model, conf):
        """Conduct testing on clients.

        Args:
            model (nn.Module): Model to test.
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
        Returns:
            :obj:`UploadRequest`: Testing contents. Unify the interface for both local and remote operations.
        """
        self.conf = conf
        if conf.track:
            reset = not self._is_train
            self._tracker.set_client_context(conf.task_id, conf.round_id, self.cid, reset_client=reset)

        self._is_train = False

        self.download(model)
        self.track(metric.TEST_DOWNLOAD_SIZE, model_size(model))

        self.decompression()

        self.pre_test()
        self.test(conf, self.device)
        self.post_test()

        self.track(metric.TEST_ACCURACY, float(self.test_accuracy))
        self.track(metric.TEST_LOSS, float(self.test_loss))
        self.track(metric.TEST_TIME, self.test_time)

        return self.upload()

    def download(self, model):
        """Download model from the server.

        Args:
            model (nn.Module): Global model distributed from the server.
        """
        if self.compressed_model:
            self.compressed_model.load_state_dict(model.state_dict())
        else:
            self.compressed_model = copy.deepcopy(model)

    def decompression(self):
        """Decompressed model. It can be further implemented when the model is compressed in the server."""
        self.model = self.compressed_model

    def pre_train(self):
        """Preprocessing before training."""
        pass

    def train(self, conf, device=CPU):
        """Execute client training.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            device (str): Hardware device for training, cpu or cuda devices.
        """
        start_time = time.time()
        loss_fn, optimizer = self.pretrain_setup(conf, device)
        self.train_loss = []
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
            current_epoch_loss = sum(batch_loss) / len(batch_loss)
            self.train_loss.append(float(current_epoch_loss))
            logger.debug("Client {}, local epoch: {}, loss: {}".format(self.cid, i, current_epoch_loss))
        self.train_time = time.time() - start_time
        logger.debug("Client {}, Train Time: {}".format(self.cid, self.train_time))

    def post_train(self):
        """Postprocessing after training."""
        pass

    def pretrain_setup(self, conf, device):
        """Setup loss function and optimizer before training."""
        self.simulate_straggler()
        self.model.train()
        self.model.to(device)
        loss_fn = self.load_loss_fn(conf)
        optimizer = self.load_optimizer(conf)
        if self.train_loader is None:
            self.train_loader = self.load_loader(conf)
        return loss_fn, optimizer

    def load_loss_fn(self, conf):
        return torch.nn.CrossEntropyLoss()

    def load_optimizer(self, conf):
        """Load training optimizer. Implemented Adam and SGD."""
        if conf.optimizer.type == "Adam":
            optimizer = torch.optim.Adam(self.model.parameters(), lr=conf.optimizer.lr)
        else:
            # default using optimizer SGD
            optimizer = torch.optim.SGD(self.model.parameters(),
                                        lr=conf.optimizer.lr,
                                        momentum=conf.optimizer.momentum,
                                        weight_decay=conf.optimizer.weight_decay)
        return optimizer

    def load_loader(self, conf):
        """Load the training data loader.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
        Returns:
            torch.utils.data.DataLoader: Data loader.
        """
        return self.train_data.loader(conf.batch_size, self.cid, shuffle=True, seed=conf.seed)

    def test_local(self):
        """Test client local model after training."""
        pass

    def pre_test(self):
        """Preprocessing before testing."""
        pass

    def test(self, conf, device=CPU):
        """Execute client testing.

        Args:
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            device (str): Hardware device for training, cpu or cuda devices.
        """
        begin_test_time = time.time()
        self.model.eval()
        self.model.to(device)
        loss_fn = self.load_loss_fn(conf)
        if self.test_loader is None:
            self.test_loader = self.test_data.loader(conf.test_batch_size, self.cid, shuffle=False, seed=conf.seed)
        # TODO: make evaluation metrics a separate package and apply it here.
        self.test_loss = 0
        correct = 0
        with torch.no_grad():
            for batched_x, batched_y in self.test_loader:
                x = batched_x.to(device)
                y = batched_y.to(device)
                log_probs = self.model(x)
                loss = loss_fn(log_probs, y)
                _, y_pred = torch.max(log_probs, -1)
                correct += y_pred.eq(y.data.view_as(y_pred)).long().cpu().sum()
                self.test_loss += loss.item()
            test_size = self.test_data.size(self.cid)
            self.test_loss /= test_size
            self.test_accuracy = 100.0 * float(correct) / test_size

        logger.debug('Client {}, testing -- Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            self.cid, self.test_loss, correct, test_size, self.test_accuracy))

        self.test_time = time.time() - begin_test_time
        self.model = self.model.cpu()

    def post_test(self):
        """Postprocessing after testing."""
        pass

    def encryption(self):
        """Encrypt the client local model."""
        # TODO: encryption of model, remember to track encrypted model instead of compressed one after implementation.
        pass

    def compression(self):
        """Compress the client local model after training and before uploading to the server."""
        self.compressed_model = self.model

    def upload(self):
        """Upload the messages from client to the server.

        Returns:
            :obj:`UploadRequest`: The upload request defined in protobuf to unify local and remote operations.
                Only applicable for local training as remote training upload through a gRPC request.
        """
        request = self.construct_upload_request()
        if not self.is_remote:
            self.post_upload()
            return request

        self.upload_remotely(request)
        self.post_upload()

    def post_upload(self):
        """Postprocessing after uploading training/testing results."""
        pass

    def construct_upload_request(self):
        """Construct client upload request for training updates and testing results.

        Returns:
            :obj:`UploadRequest`: The upload request defined in protobuf to unify local and remote operations.
        """
        data = codec.marshal(server_pb.Performance(accuracy=self.test_accuracy, loss=self.test_loss))
        typ = common_pb.DATA_TYPE_PERFORMANCE
        try:
            if self._is_train:
                data = codec.marshal(copy.deepcopy(self.compressed_model))
                typ = common_pb.DATA_TYPE_PARAMS
                data_size = self.train_data.size(self.cid)
            else:
                data_size = 1 if not self.test_data else self.test_data.size(self.cid)
        except KeyError:
            # When the datasize cannot be get from dataset, default to use equal aggregate
            data_size = 1

        m = self._tracker.get_client_metric().to_proto() if self._tracker else common_pb.ClientMetric()
        return server_pb.UploadRequest(
            task_id=self.conf.task_id,
            round_id=self.conf.round_id,
            client_id=self.cid,
            content=server_pb.UploadContent(
                data=data,
                type=typ,
                data_size=data_size,
                metric=m,
            ),
        )

    def upload_remotely(self, request):
        """Send upload request to remote server via gRPC.

        Args:
            request (:obj:`UploadRequest`): Upload request.
        """
        start_time = time.time()

        self.connect_to_server()
        resp = self._server_stub.Upload(request)

        upload_time = time.time() - start_time
        m = metric.TRAIN_UPLOAD_TIME if self._is_train else metric.TEST_UPLOAD_TIME
        self.track(m, upload_time)

        logger.info("client upload time: {}s".format(upload_time))
        if resp.status.code == common_pb.SC_OK:
            logger.info("Uploaded remotely to the server successfully\n")
        else:
            logger.error("Failed to upload, code: {}, message: {}\n".format(resp.status.code, resp.status.message))

    # Functions for remote services.

    def start_service(self):
        """Start client service."""
        if self.is_remote:
            grpc_wrapper.start_service(grpc_wrapper.TYPE_CLIENT, ClientService(self), self.local_port)

    def connect_to_server(self):
        """Establish connection between the client and the server."""
        if self.is_remote and self._server_stub is None:
            self._server_stub = grpc_wrapper.init_stub(grpc_wrapper.TYPE_SERVER, self._server_addr)
            logger.info("Successfully connected to gRPC server {}".format(self._server_addr))

    def operate(self, model, conf, index, is_train=True):
        """A wrapper over operations (training/testing) on clients.

        Args:
            model (nn.Module): Model for operations.
            conf (omegaconf.dictconfig.DictConfig): Client configurations.
            index (int): Client index in the client list, for retrieving data. TODO: improvement.
            is_train (bool): The flag to indicate whether the operation is training, otherwise testing.
        """
        try:
            # Load the data index depending on server request
            self.cid = self.train_data.users[index]
        except IndexError:
            logger.error("Data index exceed the available data, abort training")
            return

        if self.conf.track and self._tracker is None:
            self._tracker = init_tracking(init_store=False)

        if is_train:
            logger.info("Train on data index {}, client: {}".format(index, self.cid))
            self.run_train(model, conf)
        else:
            logger.info("Test on data index {}, client: {}".format(index, self.cid))
            self.run_test(model, conf)

    # Functions for tracking.

    def track(self, metric_name, value):
        """Track a metric.

        Args:
            metric_name (str): The name of the metric.
            value (str|int|float|bool|dict|list): The value of the metric.
        """
        if not self.conf.track or self._tracker is None:
            logger.debug("Tracker not available, Tracking not supported")
            return
        self._tracker.track_client(metric_name, value)

    def save_metrics(self):
        """Save client metrics to database."""
        # TODO: not tested
        if self._tracker is None:
            logger.debug("Tracker not available, no saving")
            return
        self._tracker.save_client()

    # Functions for simulation.

    def simulate_straggler(self):
        """Simulate straggler effect of system heterogeneity."""
        if self._sleep_time > 0:
            time.sleep(self._sleep_time)
