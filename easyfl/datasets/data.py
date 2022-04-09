import importlib
import json
import logging
import os

from easyfl.datasets.dataset import FederatedTensorDataset
from easyfl.datasets.utils.base_dataset import BaseDataset, CIFAR10, CIFAR100
from easyfl.datasets.utils.util import load_dict

logger = logging.getLogger(__name__)


def read_dir(data_dir):
    clients = []
    groups = []
    data = {}

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(dataset_name, train_data_dir, test_data_dir):
    """Load datasets from data directories.

    Args:
        dataset_name (str): The name of the dataset.
        train_data_dir (str): The directory of training data.
        test_data_dir (str): The directory of testing data.

    Returns:
        list[str]: A list of client ids.
        list[str]: A list of group ids for dataset with hierarchies.
        dict: A dictionary of training data, e.g., {"id1": {"x": data, "y": label}, "id2": {"x": data, "y": label}}.
        dict: A dictionary of testing data. The format is same as training data for FEMNIST and Shakespeare datasets.
            For CIFAR datasets, the format is {"x": data, "y": label}, for centralized testing in the server.
    """
    if dataset_name == CIFAR10 or dataset_name == CIFAR100:
        train_data = load_dict(train_data_dir)
        test_data = load_dict(test_data_dir)
        return [], [], train_data, test_data

    # Data in the directories are `json` files with keys `users` and `user_data`.
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


def load_data(root,
              dataset_name,
              num_of_clients,
              split_type,
              min_size,
              class_per_client,
              data_amount,
              iid_fraction,
              user,
              train_test_split,
              quantity_weights,
              alpha):
    """Simulate and load federated datasets.

    Args:
        root (str): The root directory where datasets stored.
        dataset_name (str): The name of the dataset. It currently supports: femnist, shakespeare, cifar10, and cifar100.
            Among them, femnist and shakespeare are adopted from LEAF benchmark.
        num_of_clients (int): The targeted number of clients to construct.
        split_type (str): The type of statistical simulation, options: iid, dir, and class.
            `iid` means independent and identically distributed data.
            `niid` means non-independent and identically distributed data for Femnist and Shakespeare.
            `dir` means using Dirichlet process to simulate non-iid data, for CIFAR-10 and CIFAR-100 datasets.
            `class` means partitioning the dataset by label classes, for datasets like CIFAR-10, CIFAR-100.
        min_size (int): The minimal number of samples in each client.
            It is applicable for LEAF datasets and dir simulation of CIFAR-10 and CIFAR-100.
        class_per_client (int): The number of classes in each client. Only applicable when the split_type is 'class'.
        data_amount (float): The fraction of data sampled for LEAF datasets.
            e.g., 10% means that only 10% of total dataset size are used.
        iid_fraction (float): The fraction of the number of clients used when the split_type is 'iid'.
        user (bool): A flag to indicate whether partition users of the dataset into train-test groups.
            Only applicable to LEAF datasets.
            True means partitioning users of the dataset into train-test groups.
            False means partitioning each users' samples into train-test groups.
        train_test_split (float): The fraction of data for training; the rest are for testing.
            e.g., 0.9 means 90% of data are used for training and 10% are used for testing.
        quantity_weights (list[float]): The targeted distribution of quantities to simulate data quantity heterogeneity.
            The values should sum up to 1. e.g., [0.1, 0.2, 0.7].
            The `num_of_clients` should be divisible by `len(weights)`.
            None means clients are simulated with the same data quantity.
        alpha (float): The parameter for Dirichlet distribution simulation, applicable only when split_type is `dir`.

    Returns:
        dict: A dictionary of training data, e.g., {"id1": {"x": data, "y": label}, "id2": {"x": data, "y": label}}.
        dict: A dictionary of testing data.
        function: A function to preprocess training data.
        function: A function to preprocess testing data.
        torchvision.transforms.transforms.Compose: Training data transformation.
        torchvision.transforms.transforms.Compose: Testing data transformation.
    """
    user_str = "user" if user else "sample"
    setting = BaseDataset.get_setting_folder(dataset_name, split_type, num_of_clients, min_size, class_per_client,
                                             data_amount, iid_fraction, user_str, train_test_split, alpha,
                                             quantity_weights)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    dataset_file = os.path.join(dir_path, "data_process", "{}.py".format(dataset_name))
    if not os.path.exists(dataset_file):
        logger.error("Please specify a valid process file path for process_x and process_y functions.")
    dataset_path = "easyfl.datasets.data_process.{}".format(dataset_name)
    dataset_lib = importlib.import_module(dataset_path)
    process_x = getattr(dataset_lib, "process_x", None)
    process_y = getattr(dataset_lib, "process_y", None)
    transform_train = getattr(dataset_lib, "transform_train", None)
    transform_test = getattr(dataset_lib, "transform_test", None)

    data_dir = os.path.join(root, dataset_name)
    if not data_dir:
        os.makedirs(data_dir)
    train_data_dir = os.path.join(data_dir, setting, "train")
    test_data_dir = os.path.join(data_dir, setting, "test")

    if not os.path.exists(train_data_dir) or not os.path.exists(test_data_dir):
        dataset_class_path = "easyfl.datasets.{}.{}".format(dataset_name, dataset_name)
        dataset_class_lib = importlib.import_module(dataset_class_path)
        class_name = dataset_name.capitalize()
        dataset = getattr(dataset_class_lib, class_name)(root=data_dir,
                                                         fraction=data_amount,
                                                         split_type=split_type,
                                                         user=user,
                                                         iid_user_fraction=iid_fraction,
                                                         train_test_split=train_test_split,
                                                         minsample=min_size,
                                                         num_of_client=num_of_clients,
                                                         class_per_client=class_per_client,
                                                         setting_folder=setting,
                                                         alpha=alpha,
                                                         weights=quantity_weights)
        try:
            filename = f"{setting}.zip"
            dataset.download_packaged_dataset_and_extract(filename)
            logger.info(f"Downloaded packaged dataset {dataset_name}: {filename}")
        except Exception as e:
            logger.info(f"Failed to download packaged dataset: {e.args}")

        # CIFAR10 generate data in setup() stage, LEAF related datasets generate data in sampling()
        if not os.path.exists(train_data_dir):
            dataset.setup()
        if not os.path.exists(train_data_dir):
            dataset.sampling()

    users, train_groups, train_data, test_data = read_data(dataset_name, train_data_dir, test_data_dir)
    return train_data, test_data, process_x, process_y, transform_train, transform_test


def construct_datasets(root,
                       dataset_name,
                       num_of_clients,
                       split_type,
                       min_size,
                       class_per_client,
                       data_amount,
                       iid_fraction,
                       user,
                       train_test_split,
                       quantity_weights,
                       alpha):
    """Construct and load provided federated learning datasets.

    Args:
        root (str): The root directory where datasets stored.
        dataset_name (str): The name of the dataset. It currently supports: femnist, shakespeare, cifar10, and cifar100.
            Among them, femnist and shakespeare are adopted from LEAF benchmark.
        num_of_clients (int): The targeted number of clients to construct.
        split_type (str): The type of statistical simulation, options: iid, dir, and class.
            `iid` means independent and identically distributed data.
            `niid` means non-independent and identically distributed data for Femnist and Shakespeare.
            `dir` means using Dirichlet process to simulate non-iid data, for CIFAR-10 and CIFAR-100 datasets.
            `class` means partitioning the dataset by label classes, for datasets like CIFAR-10, CIFAR-100.
        min_size (int): The minimal number of samples in each client.
            It is applicable for LEAF datasets and dir simulation of CIFAR-10 and CIFAR-100.
        class_per_client (int): The number of classes in each client. Only applicable when the split_type is 'class'.
        data_amount (float): The fraction of data sampled for LEAF datasets.
            e.g., 10% means that only 10% of total dataset size are used.
        iid_fraction (float): The fraction of the number of clients used when the split_type is 'iid'.
        user (bool): A flag to indicate whether partition users of the dataset into train-test groups.
            Only applicable to LEAF datasets.
            True means partitioning users of the dataset into train-test groups.
            False means partitioning each users' samples into train-test groups.
        train_test_split (float): The fraction of data for training; the rest are for testing.
            e.g., 0.9 means 90% of data are used for training and 10% are used for testing.
        quantity_weights (list[float]): The targeted distribution of quantities to simulate data quantity heterogeneity.
            The values should sum up to 1. e.g., [0.1, 0.2, 0.7].
            The `num_of_clients` should be divisible by `len(weights)`.
            None means clients are simulated with the same data quantity.
        alpha (float): The parameter for Dirichlet distribution simulation, applicable only when split_type is `dir`.

    Returns:
        :obj:`FederatedDataset`: Training dataset.
        :obj:`FederatedDataset`: Testing dataset.
    """
    train_data, test_data, process_x, process_y, transform_train, transform_test = load_data(root,
                                                                                             dataset_name,
                                                                                             num_of_clients,
                                                                                             split_type,
                                                                                             min_size,
                                                                                             class_per_client,
                                                                                             data_amount,
                                                                                             iid_fraction,
                                                                                             user,
                                                                                             train_test_split,
                                                                                             quantity_weights,
                                                                                             alpha)

    # CIFAR datasets are simulated.
    test_simulated = True
    if dataset_name == CIFAR10 or dataset_name == CIFAR100:
        test_simulated = False

    train_data = FederatedTensorDataset(train_data,
                                        simulated=True,
                                        do_simulate=False,
                                        process_x=process_x,
                                        process_y=process_y,
                                        transform=transform_train)
    test_data = FederatedTensorDataset(test_data,
                                       simulated=test_simulated,
                                       do_simulate=False,
                                       process_x=process_x,
                                       process_y=process_y,
                                       transform=transform_test)
    return train_data, test_data
