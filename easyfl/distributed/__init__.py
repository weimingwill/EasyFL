from easyfl.distributed.distributed import (
    dist_init,
    get_device,
    grouping,
    reduce_models,
    reduce_models_only_params,
    reduce_value,
    reduce_values,
    reduce_weighted_values,
    gather_value,
    CPU
)

from easyfl.distributed.slurm import setup, get_ip

__all__ = ['dist_init', 'get_device', 'grouping', 'gather_value', 'setup', 'get_ip',
           'reduce_models', 'reduce_models_only_params', 'reduce_value', 'reduce_values', 'reduce_weighted_values']

