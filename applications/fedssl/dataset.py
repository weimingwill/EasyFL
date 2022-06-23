import logging
import os

import torchvision
import torchvision.transforms as transforms

from easyfl.datasets import FederatedTensorDataset
from easyfl.datasets.data import CIFAR100
from easyfl.datasets.simulation import data_simulation
from easyfl.datasets.utils.util import save_dict, load_dict
from utils import get_transformation

logger = logging.getLogger(__name__)


def semi_supervised_preprocess(dataset, num_of_client, split_type, weights, alpha, min_size, class_per_client,
                               label_ratio=0.01):
    setting = f"{dataset}_{split_type}_{num_of_client}_{min_size}_{class_per_client}_{alpha}_{0}_{label_ratio}"
    data_path = f"./data/{dataset}"
    data_folder = os.path.join(data_path, setting)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    train_path = os.path.join(data_folder, "train")
    test_path = os.path.join(data_folder, "test")
    labeled_path = os.path.join(data_folder, "labeled")

    if os.path.exists(train_path):
        print("Load existing data")
        return load_dict(train_path), load_dict(test_path), load_dict(labeled_path)

    if dataset == CIFAR100:
        train_set = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True)
        test_set = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True)
    else:
        train_set = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True)
        test_set = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True)
    train_size = len(train_set.data)
    label_size = int(train_size * label_ratio)
    labeled_data = {
        'x': train_set.data[:label_size],
        'y': train_set.targets[:label_size],
    }
    train_data = {
        'x': train_set.data[label_size:],
        'y': train_set.targets[label_size:],
    }
    test_data = {
        'x': test_set.data,
        'y': test_set.targets,
    }
    print(f"{dataset} data simulation begins")
    _, train_data = data_simulation(train_data['x'],
                                    train_data['y'],
                                    num_of_client,
                                    split_type,
                                    weights,
                                    alpha,
                                    min_size,
                                    class_per_client)
    print(f"{dataset} data simulation is done")

    save_dict(train_data, train_path)
    save_dict(test_data, test_path)
    save_dict(labeled_data, labeled_path)

    return train_data, test_data, labeled_data


def get_semi_supervised_dataset(dataset, num_of_client, split_type, class_per_client, label_ratio=0.01, image_size=32,
                                gaussian=False):
    train_data, test_data, labeled_data = semi_supervised_preprocess(dataset, num_of_client, split_type, None, 0.5, 10,
                                                                     class_per_client, label_ratio)

    fine_tune_transform = transforms.Compose([
        torchvision.transforms.ToPILImage(mode='RGB'),
        torchvision.transforms.Resize(size=image_size),
        torchvision.transforms.ToTensor(),
    ])

    train_data = FederatedTensorDataset(train_data,
                                        simulated=True,
                                        do_simulate=False,
                                        process_x=None,
                                        process_y=None,
                                        transform=get_transformation("byol")(image_size, gaussian))
    test_data = FederatedTensorDataset(test_data,
                                       simulated=False,
                                       do_simulate=False,
                                       process_x=None,
                                       process_y=None,
                                       transform=get_transformation("byol")(image_size, gaussian).test_transform)
    labeled_data = FederatedTensorDataset(labeled_data,
                                          simulated=False,
                                          do_simulate=False,
                                          process_x=None,
                                          process_y=None,
                                          transform=fine_tune_transform)
    return train_data, test_data, labeled_data
