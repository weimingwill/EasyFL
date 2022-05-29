import torch
from torchvision import datasets

from dataset import get_semi_supervised_dataset
from easyfl.datasets.data import CIFAR100
from transform import SimCLRTransform


def get_data_loaders(dataset, image_size=32, batch_size=512, num_workers=8):
    transformation = SimCLRTransform(size=image_size, gaussian=False).test_transform

    if dataset == CIFAR100:
        data_path = "./data/cifar100"
        train_dataset = datasets.CIFAR100(data_path, download=True, transform=transformation)
        test_dataset = datasets.CIFAR100(data_path, train=False, download=True, transform=transformation)
    else:
        data_path = "./data/cifar10"
        train_dataset = datasets.CIFAR10(data_path, download=True, transform=transformation)
        test_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=transformation)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_loader, test_loader


def get_semi_supervised_data_loaders(dataset, data_distribution, class_per_client, label_ratio, batch_size=512, num_workers=8, image_size=32):
    transformation = SimCLRTransform(size=image_size, gaussian=False).test_transform
    if dataset == CIFAR100:
        data_path = "./data/cifar100"
        test_dataset = datasets.CIFAR100(data_path, train=False, download=True, transform=transformation)
    else:
        data_path = "./data/cifar10"
        test_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=transformation)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    _, _, labeled_data = get_semi_supervised_dataset(dataset, 5, data_distribution, class_per_client, label_ratio)
    return labeled_data.loader(batch_size), test_loader
