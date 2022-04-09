import logging

import numpy as np

from easyfl.simulation.mobile_ratio import MOBILE_RATIO
from easyfl.datasets.simulation import equal_division

logger = logging.getLogger(__name__)

SIMULATE_ISO = "iso"  # isometric sleep time distribution among selected clients
SIMULATE_DIR = "dir"  # use symmetric dirichlet process to sample sleep time heterogenous
SIMULATE_REAL = "real"  # use real speed ratio of main stream smartphones to simulate sleep time heterogenous


def assign_value_to_group(groups, values):
    assert len(groups) == len(values)
    result = []
    for i in range(len(groups)):
        result.extend([values[i]] * len(groups[i]))
    return result


def sample_real_ratio(num_values):
    value_pool = list(MOBILE_RATIO.values())
    idxs = np.random.randint(0, len(value_pool), size=num_values)
    return np.array([value_pool[i] for i in idxs]).astype(float)


def resource_hetero_simulation(fraction, hetero_type, sleep_group_num, level, total_time, num_clients):
    """Simulated resource heterogeneous by add sleeping time to clients.

    Args:
        fraction (float): The fraction of clients attending heterogeneous simulation.
        hetero_type (str): The type of heterogeneous simulation, options: iso, dir or real.
        sleep_group_num (int): The number of groups with different sleep time.
        level (int): The level of heterogeneous (0-5), 0 means no heterogeneous among clients.
        total_time (float): The total sleep time of all clients.
        num_clients (int): The total number of clients.

    Returns:
        list[float]: A list of sleep time with distribution according to heterogeneous type.
    """
    sleep_clients = int(fraction * num_clients)
    unsleep_clients = [0] * (num_clients - sleep_clients)
    sleep_group_num = sleep_group_num
    if sleep_group_num > sleep_clients:
        logger.warning("sleep_group_num {} is more than sleep_clients number {}, \
        so we set sleep_group_num to sleep_clients".format(sleep_group_num, sleep_clients))
        sleep_group_num = sleep_clients
    groups, _ = equal_division(sleep_group_num, np.arange(sleep_clients))
    if level == 0:
        distribution = np.array([1] * sleep_clients)
    elif hetero_type == SIMULATE_DIR:
        alpha = 1 / (level * level)
        values = np.random.dirichlet(np.repeat(alpha, sleep_group_num))
        distribution = assign_value_to_group(groups, values)
    elif hetero_type == SIMULATE_ISO:
        if level > 5:
            raise ValueError("level cannot be more than 5")
        begin = 0.5 - level * 0.1
        end = 0.5 + level * 0.1
        values = np.arange(begin, end, (end - begin) / sleep_group_num)
        distribution = assign_value_to_group(groups, values)
    elif hetero_type == SIMULATE_REAL:
        values = sample_real_ratio(sleep_group_num)
        distribution = assign_value_to_group(groups, values)
    else:
        raise ValueError("sleep type not supported, please use either dir or iso")
    distribution += unsleep_clients
    np.random.shuffle(distribution)
    return distribution / sum(distribution) * total_time
