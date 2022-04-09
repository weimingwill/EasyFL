import logging
import threading

from easyfl.pb import client_service_pb2_grpc as client_grpc, client_service_pb2 as client_pb, common_pb2 as common_pb
from easyfl.protocol import codec

logger = logging.getLogger(__name__)


class ClientService(client_grpc.ClientServiceServicer):
    """"Remote gRPC client service.

    Args:
        client (:obj:`BaseClient`): Federated learning client instance.
    """
    def __init__(self, client):
        self._base = client

    def Operate(self, request, context):
        """Perform training/testing operations."""
        # TODO: add request validation.
        model = codec.unmarshal(request.model)
        is_train = request.type == client_pb.OP_TYPE_TRAIN
        # Threading is necessary to respond to server quickly
        t = threading.Thread(target=self._base.operate, args=[model, request.config, request.data_index, is_train])
        t.start()
        response = client_pb.OperateResponse(
            status=common_pb.Status(code=common_pb.SC_OK),
        )
        return response
