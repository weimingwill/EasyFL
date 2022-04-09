from easyfl.server.base import BaseServer
from easyfl.server.service import ServerService
from easyfl.server.strategies import federated_averaging, federated_averaging_only_params, \
    weighted_sum, weighted_sum_only_params

__all__ = ['BaseServer', 'ServerService', 'federated_averaging', 'federated_averaging_only_params',
           'weighted_sum', 'weighted_sum_only_params']
