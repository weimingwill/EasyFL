import logging
import os
from abc import abstractmethod

from easyfl.datasets.utils.remove_users import remove
from easyfl.datasets.utils.sample import sample, extreme
from easyfl.datasets.utils.split_data import split_train_test

logger = logging.getLogger(__name__)

CIFAR10 = "cifar10"
CIFAR100 = "cifar100"


class BaseDataset(object):
    """The internal base dataset implementation.

    Args:
        root (str): The root directory where datasets stored.
        dataset_name (str): The name of the dataset.
        fraction (float): The fraction of the data chosen from the raw data to use.
        num_of_clients (int): The targeted number of clients to construct.
        split_type (str): The type of statistical simulation, options: iid, dir, and class.
            `iid` means independent and identically distributed data.
            `niid` means non-independent and identically distributed data for Femnist and Shakespeare.
            `dir` means using Dirichlet process to simulate non-iid data, for CIFAR-10 and CIFAR-100 datasets.
            `class` means partitioning the dataset by label classes, for datasets like CIFAR-10, CIFAR-100.
        minsample (int): The minimal number of samples in each client.
            It is applicable for LEAF datasets and dir simulation of CIFAR-10 and CIFAR-100.
        class_per_client (int): The number of classes in each client. Only applicable when the split_type is 'class'.
        iid_user_fraction (float): The fraction of the number of clients used when the split_type is 'iid'.
        user (bool): A flag to indicate whether partition users of the dataset into train-test groups.
            Only applicable to LEAF datasets.
            True means partitioning users of the dataset into train-test groups.
            False means partitioning each users' samples into train-test groups.
        train_test_split (float): The fraction of data for training; the rest are for testing.
            e.g., 0.9 means 90% of data are used for training and 10% are used for testing.
        num_class: The number of classes in this dataset.
        seed: Random seed.
    """

    def __init__(self,
                 root,
                 dataset_name,
                 fraction,
                 split_type,
                 user,
                 iid_user_fraction,
                 train_test_split,
                 minsample,
                 num_class,
                 num_of_client,
                 class_per_client,
                 setting_folder,
                 seed=-1,
                 **kwargs):
        # file_path = os.path.dirname(os.path.realpath(__file__))
        # self.base_folder = os.path.join(os.path.dirname(file_path), "data", dataset_name)
        self.base_folder = root
        self.dataset_name = dataset_name
        self.fraction = fraction
        self.split_type = split_type  # iid, niid, class
        self.user = user
        self.iid_user_fraction = iid_user_fraction
        self.train_test_split = train_test_split
        self.minsample = minsample
        self.num_class = num_class
        self.num_of_client = num_of_client
        self.class_per_client = class_per_client
        self.seed = seed
        if split_type == "iid":
            assert self.user == False
            self.iid = True
        elif split_type == "niid":
            # if niid, user can be either True or False
            self.iid = False

        self.setting_folder = setting_folder
        self.data_folder = os.path.join(self.base_folder, self.setting_folder)

    @abstractmethod
    def download_packaged_dataset_and_extract(self, filename):
        raise NotImplementedError("download_packaged_dataset_and_extract not implemented")

    @abstractmethod
    def download_raw_file_and_extract(self):
        raise NotImplementedError("download_raw_file_and_extract not implemented")

    @abstractmethod
    def preprocess(self):
        raise NotImplementedError("preprocess not implemented")

    @abstractmethod
    def convert_data_to_json(self):
        raise NotImplementedError("convert_data_to_json not implemented")

    @staticmethod
    def get_setting_folder(dataset, split_type, num_of_client, min_size, class_per_client,
                           fraction, iid_fraction, user_str, train_test_split, alpha=None, weights=None):
        if dataset == CIFAR10 or dataset == CIFAR100:
            return "{}_{}_{}_{}_{}_{}_{}".format(dataset, split_type, num_of_client, min_size, class_per_client, alpha,
                                                 1 if weights else 0)
        else:
            return "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(dataset, split_type, num_of_client, min_size, class_per_client,
                                                       fraction, iid_fraction, user_str, train_test_split)

    def setup(self):
        self.download_raw_file_and_extract()
        self.preprocess()
        self.convert_data_to_json()

    def sample_customized(self):
        meta_folder = os.path.join(self.base_folder, "meta")
        if not os.path.exists(meta_folder):
            os.makedirs(meta_folder)
        sample_folder = os.path.join(self.data_folder, "sampled_data")
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)
        if not os.listdir(sample_folder):
            sample(self.base_folder, self.data_folder, meta_folder, self.fraction, self.iid, self.iid_user_fraction, self.seed)

    def sample_extreme(self):
        meta_folder = os.path.join(self.base_folder, "meta")
        if not os.path.exists(meta_folder):
            os.makedirs(meta_folder)
        sample_folder = os.path.join(self.data_folder, "sampled_data")
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder)
        if not os.listdir(sample_folder):
            extreme(self.base_folder, self.data_folder, meta_folder, self.fraction, self.num_class, self.num_of_client, self.class_per_client, self.seed)

    def remove_unqualified_user(self):
        rm_folder = os.path.join(self.data_folder, "rem_user_data")
        if not os.path.exists(rm_folder):
            os.makedirs(rm_folder)
        if not os.listdir(rm_folder):
            remove(self.data_folder, self.dataset_name, self.minsample)

    def split_train_test_set(self):
        meta_folder = os.path.join(self.base_folder, "meta")
        train = os.path.join(self.data_folder, "train")
        if not os.path.exists(train):
            os.makedirs(train)
        test = os.path.join(self.data_folder, "test")
        if not os.path.exists(test):
            os.makedirs(test)
        if not os.listdir(train) and not os.listdir(test):
            split_train_test(self.data_folder, meta_folder, self.dataset_name, self.user, self.train_test_split, self.seed)

    def sampling(self):
        if self.split_type == "iid":
            self.sample_customized()
        elif self.split_type == "niid":
            self.sample_customized()
        elif self.split_type == "class":
            self.sample_extreme()
        self.remove_unqualified_user()
        self.split_train_test_set()
