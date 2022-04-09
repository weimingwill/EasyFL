import os

import easyfl
from easyfl.client import base as client_base
from easyfl.server import base as server_base


def start_remote_client(conf=None, train_data=None, test_data=None, model=None, client=None):
    """Start a remote client.

    Args:
        conf (dict): Configurations. optional, Use the configuration loaded from file if not provided. It overwrites the
            configurations from file.
        train_data (:obj:`FederatedDataset`): Training dataset.
        test_data (:obj:`FederatedDataset`): Testing dataset.
        model (nn.Module): Model used in client training.
        client (:obj:`BaseClient`): Customized federated learning client class.
    """
    parser = client_base.create_argument_parser()
    parser.add_argument('--index', type=int, default=0, help='Client index for quick testing')
    parser.add_argument('--config', type=str, default="client_config.yaml", help='Client config file')
    args = parser.parse_args()

    if os.path.isfile(args.config):
        conf = easyfl.load_config(args.config, conf)

    if train_data and test_data:
        easyfl.register_dataset(train_data, test_data)
    elif train_data:
        easyfl.register_dataset(train_data, None)
    elif test_data:
        easyfl.register_dataset(None, test_data)

    if model:
        easyfl.register_model(model)

    if client:
        easyfl.register_client(client)

    easyfl.init(conf, init_all=False)
    easyfl.start_client(args)


def start_remote_server(conf=None, test_data=None, model=None, server=None):
    """Start a remote server.

    Args:
        conf (dict): Configurations. optional, Use the configuration loaded from file if not provided. It overwrites the
            configurations from file.
        test_data (:obj:`FederatedDataset`): Test dataset for centralized testing on server.
        model (nn.Module): Model used in client training.
        server (:obj:`BaseServer`): Customized federated learning server class.
    """
    parser = server_base.create_argument_parser()
    parser.add_argument('--config', type=str, default="server_config.yaml", help='Server config file')
    args = parser.parse_args()

    if os.path.isfile(args.config):
        conf = easyfl.load_config(args.config, conf)

    if test_data:
        easyfl.register_dataset(None, test_data)

    if model:
        easyfl.register_model(model)

    if server:
        easyfl.register_server(server)

    easyfl.init(conf, init_all=False)
    easyfl.start_server(args)
