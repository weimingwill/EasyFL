# Tutorial 3: Datasets

In this note, we present how to use the out-of-the-box datasets to simulate different federated learning (FL) scenarios.
Besides, we introduce how to use the customized dataset in EasyFL.

We currently provide four out-of-the-box datasets: FEMNIST, Shakespeare, CIFAR-10, and CIFAR-100. FEMNIST and
Shakespeare are adopted from [LEAF benchmark](https://leaf.cmu.edu/). We plan to integrate and provide more
out-of-the-box datasets in the future.

## Out-of-the-box Datasets

The simulation of different FL scenarios is configured in the configurations. You can refer to the
other [tutorial](config.md) to learn more about how to modify configs. In this note, we focus on how to config the
datasets with different simulations.

The following are dataset configurations.

```yaml
data:
  # The root directory where datasets are stored.
  root: "./data/"
  # The name of the dataset, support: femnist, shakespeare, cifar10, and cifar100.
  dataset: femnist
    # The data distribution of each client, support: iid, niid (for femnist and shakespeare), and dir and class (for cifar datasets).
    # `iid` means independent and identically distributed data.
    # `niid` means non-independent and identically distributed data for FEMNIST and Shakespeare.
    # `dir` means using Dirichlet process to simulate non-iid data, for CIFAR-10 and CIFAR-100 datasets.
  # `class` means partitioning the dataset by label classes, for datasets like CIFAR-10, CIFAR-100.
  split_type: "iid"

  # The minimal number of samples in each client. It is applicable for LEAF datasets and dir simulation of CIFAR-10 and CIFAR-100.
  min_size: 10
  # The fraction of data sampled for LEAF datasets. e.g., 10% means that only 10% of the total dataset size is used.
  data_amount: 0.05
  # The fraction of the number of clients used when the split_type is 'iid'.
  iid_fraction: 0.1
    # Whether partition users of the dataset into train-test groups. Only applicable to femnist and shakespeare datasets.
    # True means partitioning users of the dataset into train-test groups.
  # False means partitioning each users' samples into train-test groups.
  user: False
  # The fraction of data for training; the rest are for testing.
  train_test_split: 0.9

  # The number of classes in each client. Only applicable when the split_type is 'class'.  
  class_per_client: 1
  # The targeted number of clients to construct.used in non-leaf dataset, number of clients split into. for leaf dataset, only used when split type class.
  num_of_clients: 100
  # The parameter for Dirichlet distribution simulation, applicable only when split_type is `dir` for CIFAR datasets.
  alpha: 0.5

    # The targeted distribution of quantities to simulate data quantity heterogeneity.
    # The values should sum up to 1. e.g., [0.1, 0.2, 0.7].
    # The `num_of_clients` should be divisible by `len(weights)`.
  # None means clients are simulated with the same data quantity.
  weights: NULL
```

Among them, `root` is applicable to all datasets. It specifies the directory to store datasets.

EasyFL automatically downloads a dataset if it is not exist in the root directory.

Next, we introduce the simulation and configuration for specific datasets.

### FEMNIST and Shakespeare Datasets

The following are basic stats of these two datasets.

FEMNIST

* Overview: Image Dataset
* Details: 3500 users, 62 different classes (10 digits, 26 lowercase, 26 uppercase), images are 28 by 28 pixels (with
  option to make them all 128 by 128 pixels)
* Task: Image Classification

Shakespeare

* Overview: Text Dataset of Shakespeare Dialogues
* Details: 1129 users (reduced to 660 with our choice of sequence length.)
* Task: Next-Character Prediction

The datasets are non-IID (independent and identically distributed) in nature.

`split_type`: There are two options for these two datasets: `iid` and `niid`, representing IID data simulation and
non-IID data simulation.

Five hyper-parameters determine the simulated dataset: `min_size`, `data_amount`, `iid_fraction`, `tran_test_split`,
and `user`.

`user` is a boolean that determines whether to partition the dataset to train test group by user or samples.
`user: True` means partitioning users of the dataset into train-test groups, i.e. some users are for training, some
users are for testing.
`user: False` means partitioning each users' samples into train-test groups, i.e. data in each client is partitioned
into training set and testing set.

Note: we normally use `test_mode: test_in_clients` for these two datasets.

#### IID Simulation

In IID simulation, data are randomly partitioned into multiple clients.

The number of clients is determined by `data_amount` and `iid_fraction`.

#### Non-IID Simulation

Since FEMNIST and Shakespeare are non-IID in nature, each user of the dataset is regarded as a client.

`data_amount` determine the number of clients participate in training.

### CIFAR-10 and CIFAR-100 Datasets

> The **CIFAR-10** dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

> The **CIFAR-100** dataset consists of 60000 32x32 colour images in 100 classes, with 600 images per class. There are 50000 training images and 10000 test images.

`split_type`: There are three options for CIFAR datasets: `iid`, `dir`, and `class`.

Three hyper-parameters determine the simulated dataset: `num_of_clients`, `class_per_client`, and `alpha`.

#### IID Simulation

In IID simulation, the training images of the datasets are randomly partitioned into `num_of_clients` clients.

#### Non-IID Simulation

We can simulate non-IID CIFAR datasets by Dirichlet process (`dir`) or by label class (`class`).

`alpha` controls the level of heterogeneity for `dir` simulation.

`class_per_client` determines the number of classes in each client.

## Customize Datasets

EasyFL also supports integrating with customized dataset to simulate federated learning.

You can use the following classes to integrate customized dataset: [FederatedImageDataset](../api.html#easyfl.datasets.FederatedImageDataset), [FederatedTensorDataset](../api.html#easyfl.datasets.FederatedTensorDataset), and [FederatedTorchDataset](../api.html#easyfl.datasets.FederatedTorchDataset).

The following is an example that integrates [nine person re-identification datasets](https://arxiv.org/abs/2008.11560), where each client contains one dataset.

```python
import easyfl
import os
from torchvision import transforms
from easyfl.datasets import FederatedImageDataset

TRANSFORM_TRAIN_LIST = transforms.Compose([
    transforms.Resize((256, 128), interpolation=3),
    transforms.Pad(10),
    transforms.RandomCrop((256, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
TRANSFORM_VAL_LIST = transforms.Compose([
    transforms.Resize(size=(256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

DATASETS = ["MSMT17", "Duke", "Market", "cuhk03", "prid", "cuhk01", "viper", "3dpes", "ilids"]

# Prepare customized training data
def prepare_train_data(data_dir):
    client_ids = []
    roots = []
    for db in DATASETS:
        client_ids.append(db)
        data_path = os.path.join(data_dir, db, "pytorch")
        roots.append(os.path.join(data_path, "train_all"))
    data = FederatedImageDataset(root=roots,
                                 simulated=True,
                                 do_simulate=False,
                                 transform=TRANSFORM_TRAIN_LIST,
                                 client_ids=client_ids)
    return data


# Prepare customized testing data
def prepare_test_data(data_dir):
    roots = []
    client_ids = []
    for db in DATASETS:
        test_gallery = os.path.join(data_dir, db, 'pytorch', 'gallery')
        test_query = os.path.join(data_dir, db, 'pytorch', 'query')
        roots.extend([test_gallery, test_query])
        client_ids.extend([f"{db}_gallery", f"{db}_query"])
    data = FederatedImageDataset(root=roots,
                                 simulated=True,
                                 do_simulate=False,
                                 transform=TRANSFORM_VAL_LIST,
                                 client_ids=client_ids)
    return data


if __name__ == '__main__':
    config = {...}
    data_dir = "datasets/"
    train_data, test_data = prepare_train_data(data_dir), prepare_test_data(data_dir)
    easyfl.register_dataset(train_data, test_data)
    easyfl.init(config)
    easyfl.run()
```

The folder structure of these datasets are as followed:
```
|-- MSMT17
|   |-- pytorch
|   |  	  |-- gallery
|   |     |-- query
|   |     |-- train
|   |     |-- train_all
|   |     `-- val
|-- cuhk01
|   |-- pytorch
|   |  	  |-- gallery
|   |     |-- query
|   |     |-- train
|   |     |-- train_all
| ...
```

Please [email us](mailto:weiming001@e.ntu.edu.sg) if you want to access these datasets with:
1. A short self-introduction.
2. The purposes of using these datasets.

*⚠️ Further distribution of the datasets are prohibited.*

### Create Your Own Federated Dataset

In case that the provided federated dataset class is not enough, 
you can implement your own federated dataset by inherit and implement [FederatedDataset](../api.html#easyfl.datasets.FederatedDataset).

You can refer to [FederatedImageDataset](../api.html#easyfl.datasets.FederatedImageDataset), [FederatedTensorDataset](../api.html#easyfl.datasets.FederatedTensorDataset), and [FederatedTorchDataset](../api.html#easyfl.datasets.FederatedTorchDataset) on how to implement.  
