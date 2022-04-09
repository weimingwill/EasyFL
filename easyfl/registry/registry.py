from easyfl.registry import etcd_client
from easyfl.registry import k8s
from easyfl.registry.vclient import VirtualClient

SOURCE_MANUAL = "manual"
SOURCE_ETCD = "etcd"
SOURCE_K8S = "kubernetes"
SOURCES = [SOURCE_MANUAL, SOURCE_ETCD, SOURCE_K8S]

CLIENT_DOCKER_IMAGE = "easyfl-client"


def get_clients(source, etcd_addresses=None):
    """Get clients from registry.

    Args:
        source (str): Registry source, options: manual, etcd, kubernetes.
        etcd_addresses (str, optional): The addresses of etcd service.
    Returns:
        list[:obj:`VirtualClient`]: A list of clients with addresses.
    """

    if source == SOURCE_MANUAL:
        return [VirtualClient("1", "localhost:23400", 0), VirtualClient("2", "localhost:23401", 1)]
    elif source == SOURCE_ETCD:
        etcd = etcd_client.EtcdClient("server", etcd_addresses, "backends")
        return etcd.get_clients(CLIENT_DOCKER_IMAGE)
    elif source == SOURCE_K8S:
        return k8s.get_clients()
    else:
        raise ValueError("Not supported source type")
