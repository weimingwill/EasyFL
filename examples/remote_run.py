import argparse

import easyfl
from easyfl.pb import common_pb2 as common_pb
from easyfl.pb import server_service_pb2 as server_pb
from easyfl.protocol import codec
from easyfl.communication import grpc_wrapper
from easyfl.registry import get_clients, SOURCES


parser = argparse.ArgumentParser(description='Federated Server')
parser.add_argument('--server-addr',
                    type=str,
                    default="172.18.0.1:23501",
                    help='Server address')
parser.add_argument('--etcd-addrs',
                    type=str,
                    default="172.17.0.1:2379",
                    help='Etcd address, or list of etcd addrs separated by ","')
parser.add_argument('--source',
                    type=str,
                    default="manual",
                    choices=SOURCES,
                    help='Source to get the clients')
args = parser.parse_args()


def send_run_request():
    config = {
        "data": {"dataset": "femnist"},
        "model": "lenet",
        "test_mode": "test_in_client"
    }

    print("Server address: {}".format(args.server_addr))
    print("Etcd address: {}".format(args.etcd_addrs))

    easyfl.init(config)
    model = easyfl.init_model()
    stub = grpc_wrapper.init_stub(grpc_wrapper.TYPE_SERVER, args.server_addr)

    request = server_pb.RunRequest(
        model=codec.marshal(model),
    )

    clients = get_clients(args.source, args.etcd_addrs)
    for c in clients:
        request.clients.append(server_pb.Client(client_id=c.id, index=c.index, address=c.address))

    response = stub.Run(request)
    if response.status.code == common_pb.SC_OK:
        print("Success")
    else:
        print(response)


if __name__ == '__main__':
    send_run_request()
