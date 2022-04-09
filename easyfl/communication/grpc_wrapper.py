from concurrent import futures

import grpc

from easyfl.pb import client_service_pb2_grpc as client_grpc
from easyfl.pb import server_service_pb2_grpc as server_grpc
from easyfl.pb import tracking_service_pb2_grpc as tracking_grpc

MAX_MESSAGE_LENGTH = 524288000  # 500MB

TYPE_CLIENT = "client"
TYPE_SERVER = "server"
TYPE_TRACKING = "tracking"


def init_stub(typ, address):
    """Initialize gRPC stub.

    Args:
        typ (str): Type of service, option: client, server, tracking
        address (str): Address of the gRPC service.
    Returns:
        (:obj:`ClientServiceStub`|:obj:`ServerServiceStub`|:obj:`TrackingServiceStub`): stub of the gRPC service.
    """

    channel = grpc.insecure_channel(
        address,
        options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ],
    )
    if typ == TYPE_CLIENT:
        stub = client_grpc.ClientServiceStub(channel)
    elif typ == TYPE_TRACKING:
        stub = tracking_grpc.TrackingServiceStub(channel)
    else:
        stub = server_grpc.ServerServiceStub(channel)

    return stub


def start_service(typ, service, port):
    """Start gRPC service.
    Args:
        typ (str): Type of service, option: client, server, tracking.
        service (:obj:`ClientService`|:obj:`ServerService`|:obj:`TrackingService`): gRPC service to start.
        port (int): The port of the service.
    """
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ],
    )
    if typ == TYPE_CLIENT:
        client_grpc.add_ClientServiceServicer_to_server(service, server)
    elif typ == TYPE_TRACKING:
        tracking_grpc.add_TrackingServiceServicer_to_server(service, server)
    else:
        server_grpc.add_ServerServiceServicer_to_server(service, server)
    server.add_insecure_port('[::]:{}'.format(port))
    server.start()
    server.wait_for_termination()


def endpoint(host, port):
    """Format endpoint.

    Args:
        host (str): Host address.
        port (int): Port number.
    Returns:
        str: Address in `host:port` format.
    """
    return "{}:{}".format(host, port)
