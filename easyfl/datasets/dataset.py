import logging
import os
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets.folder import default_loader, make_dataset

from easyfl.datasets.dataset_util import TransformDataset, ImageDataset
from easyfl.datasets.simulation import data_simulation, SIMULATE_IID

logger = logging.getLogger(__name__)

TEST_IN_SERVER = "test_in_server"
TEST_IN_CLIENT = "test_in_client"

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

DEFAULT_MERGED_ID = "Merged"


def default_process_x(raw_x_batch):
    return torch.tensor(raw_x_batch)


def default_process_y(raw_y_batch):
    return torch.tensor(raw_y_batch)


class FederatedDataset(ABC):
    """The abstract class of federated dataset for EasyFL."""

    def __init__(self):
        pass

    @abstractmethod
    def loader(self, batch_size, shuffle=True):
        """Get data loader.

        Args:
            batch_size (int): The batch size of the data loader.
            shuffle (bool): Whether shuffle the data in the loader.
        """
        raise NotImplementedError("Data loader not implemented")

    @abstractmethod
    def size(self, cid):
        """Get dataset size.

        Args:
            cid (str): client id.
        """
        raise NotImplementedError("Size not implemented")

    @property
    def users(self):
        """Get client ids of the federated dataset."""
        raise NotImplementedError("Users not implemented")


class FederatedTensorDataset(FederatedDataset):
    """Federated tensor dataset, data of clients are in format of tensor or list.

    Args:
        data (dict): A dictionary of data, e.g., {"id1": {"x": [[], [], ...], "y": [...]]}}.
            If simulation is not done previously, it is in format of {'x':[[],[], ...], 'y': [...]}.
        transform (torchvision.transforms.transforms.Compose, optional): Transformation for data.
        target_transform (torchvision.transforms.transforms.Compose, optional): Transformation for data labels.
        process_x (function, optional): A function to preprocess training data.
        process_y (function, optional): A function to preprocess testing data.
        simulated (bool, optional): Whether the dataset is simulated to federated learning settings.
        do_simulate (bool, optional): Whether conduct simulation. It is only effective if it is not simulated.
        num_of_clients (int, optional): number of clients for simulation. Only need if doing simulation.
        simulation_method(optional): split method. Only need if doing simulation.
        weights (list[float], optional): The targeted distribution of quantities to simulate quantity heterogeneity.
            The values should sum up to 1. e.g., [0.1, 0.2, 0.7].
            The `num_of_clients` should be divisible by `len(weights)`.
            None means clients are simulated with the same data quantity.
        alpha (float, optional): The parameter for Dirichlet distribution simulation, only for dir simulation.
        min_size (int, optional): The minimal number of samples in each client, only for dir simulation.
        class_per_client (int, optional): The number of classes in each client, only for non-iid by class simulation.
    """

    def __init__(self,
                 data,
                 transform=None,
                 target_transform=None,
                 process_x=default_process_x,
                 process_y=default_process_x,
                 simulated=False,
                 do_simulate=True,
                 num_of_clients=10,
                 simulation_method=SIMULATE_IID,
                 weights=None,
                 alpha=0.5,
                 min_size=10,
                 class_per_client=1):
        super(FederatedTensorDataset, self).__init__()
        self.simulated = simulated
        self.data = data
        self._validate_data(self.data)
        self.process_x = process_x
        self.process_y = process_y
        self.transform = transform
        self.target_transform = target_transform
        if simulated:
            self._users = sorted(list(self.data.keys()))

        elif do_simulate:
            # For simulation method provided, we support testing in server for now
            # TODO: support simulation for test data => test in clients
            self.simulation(num_of_clients, simulation_method, weights, alpha, min_size, class_per_client)

    def simulation(self, num_of_clients, niid=SIMULATE_IID, weights=None, alpha=0.5, min_size=10, class_per_client=1):
        if self.simulated:
            logger.warning("The dataset is already simulated, the simulation would not proceed.")
            return
        self._users, self.data = data_simulation(
            self.data['x'],
            self.data['y'],
            num_of_clients,
            niid,
            weights,
            alpha,
            min_size,
            class_per_client)
        self.simulated = True

    def loader(self, batch_size, client_id=None, shuffle=True, seed=0, transform=None, drop_last=False):
        """Get dataset loader.

        Args:
            batch_size (int): The batch size.
            client_id (str, optional): The id of client.
            shuffle (bool, optional): Whether to shuffle before batching.
            seed (int, optional): The shuffle seed.
            transform (torchvision.transforms.transforms.Compose, optional): Data transformation.
            drop_last (bool, optional): Whether to drop the last batch if its size is smaller than batch size.

        Returns:
            torch.utils.data.DataLoader: The data loader to load data.
        """
        # Simulation need to be done before creating a data loader
        if client_id is None:
            data = self.data
        else:
            data = self.data[client_id]

        data_x = data['x']
        data_y = data['y']

        data_x = np.array(data_x)
        data_y = np.array(data_y)

        data_x = self._input_process(data_x)
        data_y = self._label_process(data_y)
        if shuffle:
            np.random.seed(seed)
            rng_state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(rng_state)
            np.random.shuffle(data_y)

        transform = self.transform if transform is None else transform
        if transform is not None:
            dataset = TransformDataset(data_x,
                                       data_y,
                                       transform_x=transform,
                                       transform_y=self.target_transform)
        else:
            dataset = TensorDataset(data_x, data_y)
        loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last)
        return loader

    @property
    def users(self):
        return self._users

    @users.setter
    def users(self, value):
        self._users = value

    def size(self, cid=None):
        if cid is not None:
            return len(self.data[cid]['y'])
        else:
            return len(self.data['y'])

    def total_size(self):
        if 'y' in self.data:
            return len(self.data['y'])
        else:
            return sum([len(self.data[i]['y']) for i in self.data])

    def _input_process(self, sample):
        if self.process_x is not None:
            sample = self.process_x(sample)
        return sample

    def _label_process(self, label):
        if self.process_y is not None:
            label = self.process_y(label)
        return label

    def _validate_data(self, data):
        if self.simulated:
            for i in data:
                assert len(data[i]['x']) == len(data[i]['y'])
        else:
            assert len(data['x']) == len(data['y'])


class FederatedImageDataset(FederatedDataset):
    """
    Federated image dataset, data of clients are in format of image folder.

    Args:
        root (str|list[str]): The root directory or directories of image data folder.
            If the dataset is simulated to multiple clients, the root is a list of directories.
            Otherwise, it is the directory of an image data folder.
        simulated (bool): Whether the dataset is simulated to federated learning settings.
        do_simulate (bool, optional): Whether conduct simulation. It is only effective if it is not simulated.
        extensions (list[str], optional): A list of allowed image extensions.
            Only one of `extensions` and `is_valid_file` can be specified.
        is_valid_file (function, optional): A function that takes path of an Image file and check if it is valid.
            Only one of `extensions` and `is_valid_file` can be specified.
        transform (torchvision.transforms.transforms.Compose, optional): Transformation for data.
        target_transform (torchvision.transforms.transforms.Compose, optional): Transformation for data labels.
        num_of_clients (int, optional): number of clients for simulation. Only need if doing simulation.
        simulation_method(optional): split method. Only need if doing simulation.
        weights (list[float], optional): The targeted distribution of quantities to simulate quantity heterogeneity.
            The values should sum up to 1. e.g., [0.1, 0.2, 0.7].
            The `num_of_clients` should be divisible by `len(weights)`.
            None means clients are simulated with the same data quantity.
        alpha (float, optional): The parameter for Dirichlet distribution simulation, only for dir simulation.
        min_size (int, optional): The minimal number of samples in each client, only for dir simulation.
        class_per_client (int, optional): The number of classes in each client, only for non-iid by class simulation.
        client_ids (list[str], optional): A list of client ids.
            Each client id matches with an element in roots.
            The client ids are ["f0000001", "f00000002", ...] if not specified.
    """

    def __init__(self,
                 root,
                 simulated,
                 do_simulate=True,
                 extensions=IMG_EXTENSIONS,
                 is_valid_file=None,
                 transform=None,
                 target_transform=None,
                 client_ids="default",
                 num_of_clients=10,
                 simulation_method=SIMULATE_IID,
                 weights=None,
                 alpha=0.5,
                 min_size=10,
                 class_per_client=1):
        super(FederatedImageDataset, self).__init__()
        self.simulated = simulated
        self.transform = transform
        self.target_transform = target_transform

        if self.simulated:
            self.data = {}
            self.classes = {}
            self.class_to_idx = {}
            self.roots = root
            self.num_of_clients = len(self.roots)
            if client_ids == "default":
                self.users = ["f%07.0f" % (i) for i in range(len(self.roots))]
            else:
                self.users = client_ids
            for i in range(self.num_of_clients):
                current_client_id = self.users[i]
                classes, class_to_idx = self._find_classes(self.roots[i])
                samples = make_dataset(self.roots[i], class_to_idx, extensions, is_valid_file)
                if len(samples) == 0:
                    msg = "Found 0 files in subfolders of: {}\n".format(self.root)
                    if extensions is not None:
                        msg += "Supported extensions are: {}".format(",".join(extensions))
                    raise RuntimeError(msg)

                self.classes[current_client_id] = classes
                self.class_to_idx[current_client_id] = class_to_idx
                temp_client = {'x': [i[0] for i in samples], 'y': [i[1] for i in samples]}
                self.data[current_client_id] = temp_client
        elif do_simulate:
            self.root = root
            classes, class_to_idx = self._find_classes(self.root)
            samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
            if len(samples) == 0:
                msg = "Found 0 files in subfolders of: {}\n".format(self.root)
                if extensions is not None:
                    msg += "Supported extensions are: {}".format(",".join(extensions))
                raise RuntimeError(msg)
            self.extensions = extensions
            self.classes = classes
            self.class_to_idx = class_to_idx
            self.samples = samples
            self.inputs = [i[0] for i in self.samples]
            self.labels = [i[1] for i in self.samples]
            self.simulation(num_of_clients, simulation_method, weights, alpha, min_size, class_per_client)

    def simulation(self, num_of_clients, niid="iid", weights=[1], alpha=0.5, min_size=10, class_per_client=1):
        if self.simulated:
            logger.warning("The dataset is already simulated, the simulation would not proceed.")
            return
        self.users, self.data = data_simulation(self.inputs,
                                                self.labels,
                                                num_of_clients,
                                                niid,
                                                weights,
                                                alpha,
                                                min_size,
                                                class_per_client)
        self.simulated = True

    def loader(self, batch_size, client_id=None, shuffle=True, seed=0, num_workers=2, transform=None):
        """Get dataset loader.

        Args:
            batch_size (int): The batch size.
            client_id (str, optional): The id of client.
            shuffle (bool, optional): Whether to shuffle before batching.
            seed (int, optional): The shuffle seed.
            transform (torchvision.transforms.transforms.Compose, optional): Data transformation.
            num_workers (int, optional): The number of workers for dataset loader.

        Returns:
            torch.utils.data.DataLoader: The data loader to load data.
        """
        assert self.simulated is True
        if client_id is None:
            data = self.data
        else:
            data = self.data[client_id]
        data_x = data['x'][:]
        data_y = data['y'][:]

        # randomly shuffle data
        if shuffle:
            np.random.seed(seed)
            rng_state = np.random.get_state()
            np.random.shuffle(data_x)
            np.random.set_state(rng_state)
            np.random.shuffle(data_y)

        transform = self.transform if transform is None else transform
        dataset = ImageDataset(data_x, data_y, transform, self.target_transform)
        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers,
                                             pin_memory=False)
        return loader

    @property
    def users(self):
        return self._users

    @users.setter
    def users(self, value):
        self._users = value

    def size(self, cid=None):
        if cid is not None:
            return len(self.data[cid]['y'])
        else:
            return len(self.data['y'])

    def _find_classes(self, dir):
        """Get the classes of the dataset.

        Args:
            dir (str): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to directory and class_to_idx is a dictionary.

        Note:
            Need to ensure that no class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class FederatedTorchDataset(FederatedDataset):
    """Wrapper over PyTorch dataset.

    Args:
        data (dict): A dictionary of client datasets, format {"client_id": loader1, "client_id2": loader2}.
    """

    def __init__(self, data, users):
        super(FederatedTorchDataset, self).__init__()
        self.data = data
        self._users = users

    def loader(self, batch_size, client_id=None, shuffle=True, seed=0, num_workers=2, transform=None):
        if client_id is None:
            data = self.data
        else:
            data = self.data[client_id]

        loader = torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
        return loader

    @property
    def users(self):
        return self._users

    @users.setter
    def users(self, value):
        self._users = value

    def size(self, cid=None):
        if cid is not None:
            return len(self.data[cid])
        else:
            return len(self.data)
