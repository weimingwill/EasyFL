from kubernetes import client, config
from easyfl.registry.vclient import VirtualClient


def get_clients():
    """Get clients in kubernetes based on client field selector.

    Returns:
        list[:obj:`VirtualClient`]: A list of clients.
    """
    config.load_kube_config()

    v1 = client.CoreV1Api()

    ret = v1.list_namespaced_endpoints('easyfl', watch=False, field_selector="metadata.name=easyfl-client-svc")
    
    clients = []
    for record in ret.items:
        for subset in record.subsets:
            port = subset.ports[0].port            
            for index, address in enumerate(subset.addresses):
                addr = "{}:{}".format(address.ip, port)
                c = VirtualClient(address.target_ref.name, addr, index)
                clients.append(c)
    return clients
    

if __name__ == '__main__':
    get_clients()
