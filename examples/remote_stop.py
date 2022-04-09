import argparse

from easyfl.communication import grpc_wrapper
from easyfl.pb import common_pb2 as common_pb
from easyfl.pb import server_service_pb2 as server_pb

parser = argparse.ArgumentParser(description='Federated Server')
parser.add_argument('--server-addr',
                    type=str,
                    default="172.18.0.1:23501",
                    help='Server address')

args = parser.parse_args()


def send_stop_request():
    stub = grpc_wrapper.init_stub(grpc_wrapper.TYPE_SERVER, args.server_addr)
    response = stub.Stop(server_pb.StopRequest())
    if response.status.code == common_pb.SC_OK:
        print("Success")
    else:
        print(response)


if __name__ == '__main__':
    send_stop_request()
