import argparse
import logging
import os

import easyfl
from client import FedReIDClient
from dataset import prepare_train_data, prepare_test_data
from easyfl.distributed import slurm
from model import Model

logger = logging.getLogger(__name__)


def run():
    parser = argparse.ArgumentParser(description='FedReID Application')
    parser.add_argument('--task_id', type=str, default="")
    parser.add_argument('--data_dir', type=str, metavar='PATH', default="datasets/fedreid")
    parser.add_argument("--datasets", nargs="+", default=None, help="list of datasets, e.g., ['ilids']")
    parser.add_argument('--test_every', type=int, default=10)
    parser.add_argument("--gpu", type=int, default=1, help="default number of GPU")
    args = parser.parse_args()
    logger.info("arguments: ", args)

    train_data = prepare_train_data(args.data_dir, args.datasets)
    test_data = prepare_test_data(args.data_dir, args.datasets)
    easyfl.register_dataset(train_data, test_data)
    easyfl.register_model(Model)
    easyfl.register_client(FedReIDClient)

    config = {
        "task_id": args.task_id,
        "gpu": args.gpu,
        "client": {
            "test_every": args.test_every,
        },
        "server": {
            "test_every": args.test_every
        }
    }
    if args.gpu > 1:
        rank, local_rank, world_size, host_addr = slurm.setup()
        distribute_config = {
            "gpu": world_size,
            "distributed": {
                "rank": rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "init_method": host_addr
            },
        }
        config.update(distribute_config)
    config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.yaml")
    config = easyfl.load_config(config_file, config)

    easyfl.init(config)
    easyfl.run()


if __name__ == '__main__':
    run()
