import argparse
import os

import torch

import easyfl
from easyfl.distributed import slurm

from client import MASClient
from server import MASServer
from dataset import get_dataset
from losses import parse_tasks
from models.model import get_model


STANDALONE_CLIENT_FOLDER = "standalone_clients"
DEFAULT_CLIENT_ID = "NA"

def construct_parser(parser):
    parser.add_argument("--task_id", type=str, default="")
    parser.add_argument('--tasks', default='s', help='which tasks to train, options: sdnkt')
    parser.add_argument('--task_groups', default='', help='e.g., groups of tasks separtely by comma, "sd,nkt"')
    parser.add_argument("--dataset", type=str, default='taskonomy', help='')
    parser.add_argument("--arch", type=str, default='xception', help='model architecture')
    parser.add_argument('--data_dir', type=str, help='directory to load data')
    parser.add_argument('--client_file', type=str, default='clients.txt', help='directory to load data')
    parser.add_argument('--client_id', type=str, default=DEFAULT_CLIENT_ID, help='client id for standalone training')

    parser.add_argument('--image_size', default=256, type=int, help='size of image side (images are square)')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--local_epoch', default=5, type=int)
    parser.add_argument('--rounds', default=100, type=int)
    parser.add_argument('--num_of_clients', default=32, type=int)
    parser.add_argument('--clients_per_round', default=5, type=int)
    parser.add_argument('--optimizer_type', default='SGD', type=str, help='optimizer type')
    parser.add_argument('--random_selection', action='store_true', help='whether randomly select clients')
    parser.add_argument('--lr', default=0.1, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_type', default="poly", type=str,
                        help='learning rate schedule type: poly or custom, custom lr requires stateful client.')
    parser.add_argument('--minimum_learning_rate', default=3e-5, type=float,
                        metavar='LR', help='End training when learning rate falls below this value.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')

    parser.add_argument('--test_every', default=10, type=int, help='test every x rounds')
    parser.add_argument('--save_model_every', default=10, type=int, help='save model every x rounds')
    parser.add_argument("--aggregation_content", type=str, default="all", help="aggregation content")
    parser.add_argument("--aggregation_strategy", type=str, default="FedAvg", help="aggregation strategy")
    
    parser.add_argument('--lookahead', default='y', type=str, help='whether use lookahead optimizer')
    parser.add_argument('--lookahead_step', default=10, type=int, help='lookahead every x step')
    parser.add_argument('--num_workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--rotate_loss', dest='rotate_loss', action='store_true', help='should loss rotation occur')
    parser.add_argument('--pretrained', default='n', help='use pretrained model')
    parser.add_argument('--pretrained_tasks', default='sdnkt', help='tasks for pretrained')
    parser.add_argument('--load_decoder', default='y', help='whether load pretrained decoder')
    parser.add_argument('--fp16', action='store_true', help='Run model fp16 mode.')
    parser.add_argument('--half', default='n', help='whether use half output')
    parser.add_argument('--half_sized_output', action='store_true', help='output 128x128 rather than 256x256.')
    parser.add_argument('--no_augment', action='store_true', help='Run model fp16 mode.')
    parser.add_argument('--model_limit', default=None, type=int,
                        help='Limit the number of training instances from a single 3d building model.')
    parser.add_argument('--task_weights', default=None, type=str,
                        help='a comma separated list of numbers one for each task to multiply the loss by.')
    parser.add_argument('-vb', '--virtual_batch_multiplier', default=1, type=int,
                        metavar='N', help='number of forward/backward passes per parameter update')
    parser.add_argument('--dist_port', default=23344, type=int)
    parser.add_argument('--run_count', default=0, type=int)

    # Not effective arguments, to be deleted
    parser.add_argument('--maximum_loss_tracking_window', default=2000000, type=int,
                        help='maximum loss tracking window (default: 2000000)')

    return parser


def run(args):
    rank, local_rank, world_size, host_addr = slurm.setup(args.dist_port)
    task_id = args.task_id
    if task_id == "":
        task_id = f"{args.arch}_{args.tasks}_{args.clients_per_round}c{args.num_of_clients}_run{args.run_count}"

    tasks = parse_tasks(args.tasks)

    config = {
        "task_id": task_id,
        "model": args.arch,
        "gpu": world_size,
        "distributed": {"rank": rank, "local_rank": local_rank, "world_size": world_size, "init_method": host_addr},
        "test_mode": "test_in_server",
        "server": {
            "batch_size": args.batch_size,
            "rounds": args.rounds,
            "test_every": args.test_every,
            "save_model_every": args.save_model_every,
            "clients_per_round": args.clients_per_round,
            "test_all": False,  # False means do not test clients in the start of training
            "random_selection": args.random_selection,
            "aggregation_content": args.aggregation_content,
            "aggregation_stragtegy": args.aggregation_strategy,
            "track": False,
        },
        "client": {
            "track": False,
            "drop_last": True,
            "batch_size": args.batch_size,
            "local_epoch": args.local_epoch,
            "rounds": args.rounds,

            "optimizer": {
                "type": args.optimizer_type,
                "lr_type": args.lr_type,
                "lr": args.lr,
                "momentum": args.momentum,
                "weight_decay": args.weight_decay,
            },
            "minimum_learning_rate": args.minimum_learning_rate,
            
            "tasks": tasks,
            "task_str": args.tasks,
            "task_weights": args.task_weights,
            "rotate_loss": args.rotate_loss,

            "lookahead": args.lookahead,
            "lookahead_step": args.lookahead_step,
            "num_workers": args.num_workers,
            "fp16": args.fp16,
            "virtual_batch_multiplier": args.virtual_batch_multiplier,
            "maximum_loss_tracking_window": args.maximum_loss_tracking_window,
        },
        "tracking": {"database": os.path.join(os.getcwd(), "tracker", task_id)},
    }

    model = get_model(args.arch, tasks)
    if args.pretrained != "n":
        pretrained_tasks = parse_tasks(args.pretrained_tasks)
        pretrained_model = get_model(args.arch, pretrained_tasks)
        pretrained_path = os.path.join(os.getcwd(), "saved_models", "mas", args.pretrained)

        checkpoint = torch.load(pretrained_path)
        pretrained_model.load_state_dict(checkpoint['state_dict'])

        model.encoder.load_state_dict(pretrained_model.encoder.state_dict())
        if not args.load_decoder == "n":
            print("load decoder")
            pretrained_decoder_keys = list(pretrained_model.task_to_decoder.keys())
            for i, task in enumerate(model.task_to_decoder.keys()):
                pi = pretrained_decoder_keys.index(task)
                model.decoders[i].load_state_dict(pretrained_model.decoders[pi].state_dict())

    augment = not args.no_augment
    client_file = args.client_file
    if args.client_id != DEFAULT_CLIENT_ID:
        client_file = f"{STANDALONE_CLIENT_FOLDER}/{args.client_id}.txt"
        with open(client_file, "w") as f:
            f.write(args.client_id)
    if args.half == 'y':
        args.half_sized_output = True
    print("train client file:", client_file)
    print("test client file:", args.client_file)
    train_data, val_data, test_data = get_dataset(args.data_dir,
                                                  train_client_file=client_file,
                                                  test_client_file=args.client_file,
                                                  tasks=tasks,
                                                  image_size=args.image_size,
                                                  model_limit=args.model_limit,
                                                  half_sized_output=args.half_sized_output,
                                                  augment=augment)
    easyfl.register_dataset(train_data, test_data, val_data)
    easyfl.register_model(model)
    easyfl.register_client(MASClient)
    easyfl.register_server(MASServer)
    easyfl.init(config, init_all=True)
    easyfl.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MAS')
    parser = construct_parser(parser)
    args = parser.parse_args()
    print("arguments: ", args)
    run(args)
