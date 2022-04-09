import logging
import threading

from easyfl.pb import server_service_pb2_grpc as server_grpc, server_service_pb2 as server_pb, common_pb2 as common_pb
from easyfl.protocol import codec
from easyfl.tracking import metric

logger = logging.getLogger(__name__)


class ServerService(server_grpc.ServerServiceServicer):
    """"Remote gRPC server service.

    Args:
        server (:obj:`BaseServer`): Federated learning server instance.
    """
    def __init__(self, server):
        self._base = server

        self._clients_per_round = 0

        self._train_client_count = 0
        self._uploaded_models = {}
        self._uploaded_weights = {}
        self._uploaded_metrics = []

        self._test_client_count = 0
        self._accuracies = []
        self._losses = []
        self._test_sizes = []

    def Run(self, request, context):
        """Trigger federated learning process."""
        response = server_pb.RunResponse(
            status=common_pb.Status(code=common_pb.SC_OK),
        )

        if self._base.is_training():
            response = server_pb.RunResponse(
                status=common_pb.Status(
                    code=common_pb.SC_ALREADY_EXISTS,
                    message="Training in progress, please stop current training or wait for completion",
                ),
            )
        else:
            model = codec.unmarshal(request.model)
            self._base.start_remote_training(model, request.clients)

        return response

    def Stop(self, request, context):
        """Stop federated learning process."""
        response = server_pb.StopResponse(
            status=common_pb.Status(code=common_pb.SC_OK),
        )

        if self._base.is_training():
            self._base.stop()
        else:
            response = server_pb.RunResponse(
                status=common_pb.Status(
                    code=common_pb.SC_NOT_FOUND,
                    message="No existing training",
                ),
            )
        return response

    def Upload(self, request, context):
        """Handle upload from clients."""
        # TODO: put train and test logic in a separate thread and add thread lock to ensure atomicity.
        t = threading.Thread(target=self._handle_upload, args=[request, context])
        t.start()
        response = server_pb.UploadResponse(
            status=common_pb.Status(code=common_pb.SC_OK),
        )
        return response

    def _handle_upload(self, request, context):
        # if not self._base.upload_event.is_set():
        data = codec.unmarshal(request.content.data)
        data_size = request.content.data_size
        client_metric = metric.ClientMetric.from_proto(request.content.metric)

        clients_per_round = self._base.conf.server.clients_per_round
        num_of_clients = self._base.num_of_clients()
        if num_of_clients < clients_per_round:
            # TODO: use a more appropriate way to handle this situation
            logger.warning(
                "Available number of clients {} is smaller than clients per round {}".format(num_of_clients,
                                                                                             clients_per_round))
            self._clients_per_round = num_of_clients
        else:
            self._clients_per_round = clients_per_round

        if request.content.type == common_pb.DATA_TYPE_PARAMS:
            self._handle_upload_train(request.client_id, data, data_size, client_metric)
        elif request.content.type == common_pb.DATA_TYPE_PERFORMANCE:
            self._handle_upload_test(data, data_size, client_metric)

    def _handle_upload_train(self, client_id, data, data_size, client_metric):
        model = self._base.decompression(data)
        self._uploaded_models[client_id] = model
        self._uploaded_weights[client_id] = data_size
        self._uploaded_metrics.append(client_metric)
        self._train_client_count += 1
        self._trigger_aggregate_train()

    def _handle_upload_test(self, data, data_size, client_metric):
        self._accuracies.append(data.accuracy)
        self._losses.append(data.loss)
        self._test_sizes.append(data_size)
        self._uploaded_metrics.append(client_metric)
        self._test_client_count += 1
        self._trigger_aggregate_test()

    def _trigger_aggregate_train(self):
        logger.info("train_client_count: {}/{}".format(self._train_client_count, self._clients_per_round))
        if self._train_client_count == self._clients_per_round:
            self._base.set_client_uploads_train(self._uploaded_models, self._uploaded_weights, self._uploaded_metrics)
            self._train_client_count = 0
            self._reset_train_cache()
            with self._base.condition():
                self._base.notify_all()

    def _trigger_aggregate_test(self):
        # TODO: determine the testing clients not only by the selected number of clients
        if self._test_client_count == self._clients_per_round:
            self._base.set_client_uploads_test(self._accuracies, self._losses, self._test_sizes, self._uploaded_metrics)
            self._test_client_count = 0
            self._reset_test_cache()
            with self._base.condition():
                self._base.notify_all()

    def _reset_train_cache(self):
        self._uploaded_models = {}
        self._uploaded_weights = {}
        self._uploaded_metrics = []

    def _reset_test_cache(self):
        self._accuracies = []
        self._losses = []
        self._test_sizes = []
        self._uploaded_metrics = []
