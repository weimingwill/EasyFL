import logging
import os
import re
import socket

logger = logging.getLogger(__name__)


def setup(port=23344):
    """Setup distributed settings of slurm.

    Args:
        port (int, optional): The port of the primary server.
            It respectively auto-increments by 1 when the port is in-use.

    Returns:
        int: The rank of current process.
        int: The local rank of current process.
        int: Total number of processes.
        str: The address of the distributed init method.
    """
    try:
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        host = get_ip(os.environ['SLURM_STEP_NODELIST'])
        while is_port_in_use(host, port):
            port += 1
        host_addr = 'tcp://' + host + ':' + str(port)
    except KeyError:
        return 0, 0, 0, ""
    return rank, local_rank, world_size, host_addr


def get_ip(node_list):
    """Get the ip address of nodes.

    Args:
        node_list (str): Name of the nodes.

    Returns:
        str: The first node in the nodes.
    """
    if "[" not in node_list:
        return node_list
    r = re.search(r'([\w-]*)\[(\d*)[-+,+\d]*\]', node_list)
    if not r:
        return
    base, node = r.groups()
    return base + node


def is_port_in_use(host, port):
    """Check whether the port is in use.

    Args:
        host (str): Host address.
        port (int): Port to use.

    Returns:
        bool: A flag to indicate whether the port is in use in the host.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0
