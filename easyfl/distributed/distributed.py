import logging

import numpy as np
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

CPU = "cpu"

RANDOMIZE_GROUPING = "random"
GREEDY_GROUPING = "greedy"
SLOWEST_GROUPING = "slowest"


def reduce_models(model, sample_sum):
    """Aggregate models across devices and update the model with the new aggregated model parameters.

    Args:
        model (nn.Module): The model in a device to aggregate.
        sample_sum (int): Sum of the total dataset sizes of clients in a device.
    """
    dist.all_reduce(sample_sum, op=dist.ReduceOp.SUM)
    state = model.state_dict()
    for k in state.keys():
        dist.all_reduce(state[k], op=dist.ReduceOp.SUM)
        state[k] = torch.div(state[k], sample_sum)
    model.load_state_dict(state)


def reduce_models_only_params(model, sample_sum):
    """Aggregate models across devices and update the model with the new aggregated model parameters,
    excluding the persistent buffers like BN stats.

    Args:
        model (nn.Module): The model in a device to aggregate.
        sample_sum (torch.Tensor): Sum of the total dataset sizes of clients in a device.
    """
    dist.all_reduce(sample_sum, op=dist.ReduceOp.SUM)
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data = torch.div(param.data, sample_sum)


def reduce_value(value, device):
    """Calculate the sum of the value across devices.

    Args:
        value (float/int): Value to sum.
        device (str): The device where the value is on, either cpu or cuda devices.
    Returns:
         torch.Tensor: Sum of the values.
    """
    v = torch.tensor(value).to(device)
    dist.all_reduce(v, op=dist.ReduceOp.SUM)
    return v


def reduce_values(values, device):
    """Calculate the average of values across devices.

    Args:
        values (list[float|int]): Values to average.
        device (str): The device where the value is on, either cpu or cuda devices.
    Returns:
         torch.Tensor: The average of the values across devices.
    """
    length = torch.tensor(len(values)).to(device)
    total = torch.tensor(sum(values)).to(device)
    dist.all_reduce(length, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return torch.div(total, length)


def reduce_weighted_values(values, weights, device):
    """Calculate the weighted average of values across devices.

    Args:
        values (list[float|int]): Values to average.
        weights (list[float|int]): The weights to calculate weighted average.
        device (str): The device where the value is on, either cpu or cuda devices.
    Returns:
         torch.Tensor: The average of values across devices.
    """
    values = torch.tensor(values).to(device)
    weights = torch.tensor(weights).to(device)
    total_weights = torch.sum(weights).to(device)
    weighted_sum = torch.sum(values * weights).to(device)
    dist.all_reduce(total_weights, op=dist.ReduceOp.SUM)
    dist.all_reduce(weighted_sum, op=dist.ReduceOp.SUM)
    return torch.div(weighted_sum, total_weights)


def gather_value(value, world_size, device):
    """Gather the value from devices to a list.

    Args:
        value (float|int): The value to gather.
        world_size (int): The number of processes.
        device (str): The device where the value is on, either cpu or cuda devices.
    Returns:
         list[torch.Tensor]: A list of gathered values.
    """
    v = torch.tensor(value).to(device)
    target = [v.clone() for _ in range(world_size)]
    dist.all_gather(target, v)
    return target


def grouping(clients, world_size, default_time=10, strategy=RANDOMIZE_GROUPING, seed=1):
    """Divide clients into groups with different strategies.

    Args:
        clients (list[:obj:`BaseClient`]): A list of clients.
        world_size (int): The number of processes, it represent the number of groups here.
        default_time (float, optional): The default training time for not profiled clients.
        strategy (str, optional): Strategy of grouping, options: random, greedy, worst.
            When no strategy is applied, each client is a group.
        seed (int, optional): Random seed.

    Returns:
        list[list[:obj:`BaseClient`]]: Groups of clients, each group is a sub-list.
    """
    np.random.seed(seed)
    if strategy == RANDOMIZE_GROUPING:
        return randomize_grouping(clients, world_size)
    elif strategy == GREEDY_GROUPING:
        return greedy_grouping(clients, world_size, default_time)
    elif strategy == SLOWEST_GROUPING:
        return slowest_grouping(clients, world_size)
    else:
        # default, no strategy applied
        return [[client] for client in clients]


def randomize_grouping(clients, world_size):
    """"Randomly divide clients into groups.

    Args:
        clients (list[:obj:`BaseClient`]): A list of clients.
        world_size (int): The number of processes, it represent the number of groups here.

    Returns:
        list[list[:obj:`BaseClient`]]: Groups of clients, each group is a sub-list.
    """
    num_of_clients = len(clients)
    np.random.shuffle(clients)
    data_per_client = num_of_clients // world_size
    large_group_num = num_of_clients - world_size * data_per_client
    small_group_num = world_size - large_group_num
    grouped_clients = []
    for i in range(small_group_num):
        base_index = data_per_client * i
        grouped_clients.append(clients[base_index: base_index + data_per_client])
    small_size = data_per_client * small_group_num
    data_per_client += 1
    for i in range(large_group_num):
        base_index = small_size + data_per_client * i
        grouped_clients.append(clients[base_index: base_index + data_per_client])
    return grouped_clients


def greedy_grouping(clients, world_size, default_time):
    """"Greedily allocate the clients with longest training time to the most available device.


    Args:
        clients (list[:obj:`BaseClient`]): A list of clients.
        world_size (int): The number of processes, it represent the number of groups here.
        default_time (float, optional): The default training time for not profiled clients.

    Returns:
        list[list[:obj:`BaseClient`]]: Groups of clients, each group is a sub-list.
    """
    round_time_estimation = [[i, c.round_time] if c.round_time != 0
                             else [i, default_time] for i, c in enumerate(clients)]
    round_time_estimation = sorted(round_time_estimation, reverse=True, key=lambda tup: (tup[1], tup[0]))
    top_world_size = round_time_estimation[:world_size]
    groups = [[clients[index]] for (index, time) in top_world_size]
    time_sum = [time for (index, time) in top_world_size]
    for i in round_time_estimation[world_size:]:
        min_index = np.argmin(time_sum)
        groups[min_index].append(clients[i[0]])
        time_sum[min_index] += i[1]
    return groups


def slowest_grouping(clients, world_size):
    """"Allocate the clients with longest training time to the most busy device.
    Only for experiment, not practical in use.


    Args:
        clients (list[:obj:`BaseClient`]): A list of clients.
        world_size (int): The number of processes, it represent the number of groups here.

    Returns:
        list[list[:obj:`BaseClient`]]: Groups of clients, each group is a sub-list.
    """
    num_of_clients = len(clients)
    clients = sorted(clients, key=lambda tup: (tup.round_time, tup.cid))
    data_per_client = num_of_clients // world_size
    large_group_num = num_of_clients - world_size * data_per_client
    small_group_num = world_size - large_group_num
    grouped_clients = []
    for i in range(small_group_num):
        base_index = data_per_client * i
        grouped_clients.append(clients[base_index: base_index + data_per_client])
    small_size = data_per_client * small_group_num
    data_per_client += 1
    for i in range(large_group_num):
        base_index = small_size + data_per_client * i
        grouped_clients.append(clients[base_index: base_index + data_per_client])
    return grouped_clients


def dist_init(backend, init_method, world_size, rank, local_rank):
    """Initialize PyTorch distribute.

    Args:
        backend (str or Backend): Distributed backend to use, e.g., `nccl`, `gloo`.
        init_method (str, optional): URL specifying how to initialize the process group.
        world_size (int, optional): Number of processes participating in the job.
        rank (int, optional): Rank of the current process.
        local rank (int, optional): Local rank of the current process.

    Returns:
        int: Rank of current process.
        int: Total number of processes.
    """
    dist.init_process_group(backend, init_method=init_method, rank=rank, world_size=world_size)
    assert dist.is_initialized()
    return rank, world_size


def get_device(gpu, world_size, local_rank):
    """Obtain the device by checking the number of GPUs and distributed settings.

    Args:
        gpu (int): The number of requested gpu.
        world_size (int): The number of processes.
        local_rank (int): The local rank of the current process.

    Returns:
        str: Device to be used in PyTorch like `tensor.to(device)`.
    """
    if gpu > world_size:
        logger.error("Available gpu: {}, requested gpu: {}".format(world_size, gpu))
        raise ValueError("available number of gpu are less than requested")

    # TODO: think of a better way to handle this, maybe just use one config param instead of two.
    assert gpu == world_size

    n = torch.cuda.device_count()

    device_ids = list(range(n))
    return device_ids[local_rank]
