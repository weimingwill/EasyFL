# Tutorial 2: Configurations

Configurations in EasyFL are to control and config the federated learning (FL) training behavior. It instructs data simulation, the model for training, training hyperparameters, distributed training, etc. 

We provide [default configs](#default-configurations) in EasyFL, while there are two ways you can modify the configs of EasyFL: using Python and using a yaml file.

## Modify Config

EasyFL provides two ways to modify the configurations: using Python dictionary and using a yaml file. Either way, if the new configs exist in the default configuration, they overwrite those specific fields. If the new configs do not exist, it adds them to the EasyFL configuration. Thus, you can either modify the default configurations or add new configurations based on your application needs.

### 1. Modify Using Python Dictionary

You can create a new Python dictionary to specify configurations. These configs take effect when you initialize EasyFL with them by calling `easyfl.init(config)`.   
 
The examples provided in the previous [tutorial](high-level_apis.md) demonstrate how to modify config via a Python dictionary.  
```python
import easyfl

# Define customized configurations.
config = {
    "data": {
        "dataset": "cifar10", 
        "num_of_clients": 1000
    },
    "server": {
        "rounds": 5, 
        "clients_per_round": 2
    },
    "client": {"local_epoch": 5},
    "model": "resnet18",
    "test_mode": "test_in_server",
}
# Initialize EasyFL with the new config.
easyfl.init(config)
# Execute federated learning training.
easyfl.run()
```

### 2. Modify Using A Yaml File

You can create a new yaml file named `config.yaml` for configuration and load them into EasyFL.

```python
import easyfl
# Define customized configurations in a yaml file.
config_file = "config.yaml"
# Load the yaml file as config.
config = easyfl.load_config(config_file)
# Initialize EasyFL with the new config.
easyfl.init(config)
# Execute federated learning training.
easyfl.run()
```

You can also combine these two methods of modifying configs.

```python
import easyfl

# Define part of customized configs.
config = {
    "data": {
        "dataset": "cifar10", 
        "num_of_clients": 1000
    },
    "server": {
        "rounds": 5, 
        "clients_per_round": 2
    },
    "client": {"local_epoch": 5},
    "model": "resnet18",
    "test_mode": "test_in_server",
}

# Define part of configs in a yaml file.
config_file = "config.yaml"
# Load and combine these two configs.
config = easyfl.load_config(config_file, config)
# Initialize EasyFL with the new config.
easyfl.init(config)
# Execute federated learning training.
easyfl.run()
```

## A Common Practice to Modify Configuration

Since some configurations are directly related to training, we may need to set them dynamically with different values. 

For example, we may want to experiment with the effect of batch size and local epoch on federated learning. Instead of changing the value manually each time in configuration, you can pass the value in as command-line arguments and set the value with different commands.

```python
import easyfl
import argparse

# Define command line arguments.
parser = argparse.ArgumentParser(description='Example')
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--local_epoch", type=int, default=5)
args = parser.parse_args()
print("args", args)

# Define customized configurations using the arguments.
config = {
    "client": {
        "batch_size": args.batch_size,
        "local_epoch": args.local_epoch,
    }
}
# Initialize EasyFL with the new config.
easyfl.init(config)
# Execute federated learning training.
easyfl.run()
```


## Default Configurations

The followings are the default configurations in EasyFL. 
They are copied from `easyfl/config.yaml` on April, 2022.

We provide more details on how to simulate different FL scenarios with the out-of-the-box datasets in [another note](dataset.md).  

```yaml
# The unique identifier for each federated learning task
task_id: ""

# Provide dataset and federated learning simulation related configuration.
data:
  # The root directory where datasets are stored.
  root: "./data/"
  # The name of the dataset, support: femnist, shakespeare, cifar10, and cifar100.
  dataset: femnist
  # The data distribution of each client, support: iid, niid (for femnist and shakespeare), and dir and class (for cifar datasets).
    # `iid` means independent and identically distributed data.
    # `niid` means non-independent and identically distributed data for Femnist and Shakespeare.
    # `dir` means using Dirichlet process to simulate non-iid data, for CIFAR-10 and CIFAR-100 datasets.
    # `class` means partitioning the dataset by label classes, for datasets like CIFAR-10, CIFAR-100.
  split_type: "iid"
  
  # The minimal number of samples in each client. It is applicable for LEAF datasets and dir simulation of CIFAR-10 and CIFAR-100.
  min_size: 10
  # The fraction of data sampled for LEAF datasets. e.g., 10% means that only 10% of total dataset size are used.
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

# The name of the model for training, support: lenet, rnn, resnet, resnet18, resnet50, vgg9.
model: lenet
# How to conduct testing, options: test_in_client or test_in_server.
  # `test_in_client` means that each client has a test set to run testing.
  # `test_in_server` means that server has a test set to run testing for the global model. Use this mode for cifar datasets.
test_mode: "test_in_client"
# The way to measure testing performance (accuracy) when test mode is `test_in_client`, support: average or weighted (means weighted average).
test_method: "average"

server:
  track: False  # Whether track server metrics using the tracking service.
  rounds: 10  # Total training round.
  clients_per_round: 5  # The number of clients to train in each round.
  test_every: 1  # The frequency of testing: conduct testing every N round.
  save_model_every: 10  # The frequency of saving model: save model every N round.
  save_model_path: ""  # The path to save model. Default path is root directory of the library.
  batch_size: 32  # The batch size of test_in_server.
  test_all: True  # Whether test all clients or only selected clients.
  random_selection: True  # Whether select clients to train randomly.
  # The strategy to aggregate client uploaded models, options: FedAvg, equal.
    # FedAvg aggregates models using weighted average, where the weights are data size of clients.
    # equal aggregates model by simple averaging.
  aggregation_strategy: "FedAvg"
  # The content of aggregation, options: all, parameters.
    # all means aggregating models using state_dict, including both model parameters and persistent buffers like BatchNorm stats.
    # parameters means aggregating only model parameters.
  aggregation_content: "all"

client:
  track: False  # Whether track server metrics using the tracking service.
  batch_size: 32  # The batch size of training in client.
  test_batch_size: 5  # The batch size of testing in client.
  local_epoch: 10  # The number of epochs to train in each round.
  optimizer:
    type: "Adam"  # The name of the optimizer, options: Adam, SGD.
    lr: 0.001
    momentum: 0.9
    weight_decay: 0
  seed: 0
  local_test: False  # Whether test the trained models in clients before uploading them to the server.

gpu: 0  # The total number of GPUs used in training. 0 means CPU.
distributed:  # The distributed training configurations. It is only applicable when gpu > 1.
  backend: "nccl"  # The distributed backend.
  init_method: ""
  world_size: 0
  rank: 0
  local_rank: 0

tracking:  # The configurations for logging and tracking.
  database: ""  # The path of local dataset, sqlite3.
  log_file: ""
  log_level: "INFO"  # The level of logging.
  metric_file: ""
  save_every: 1

# The configuration for system heterogeneity simulation.
resource_heterogeneous:
  simulate: False  # Whether simulate system heterogeneity in federated learning.
  # The type of heterogeneity to simulate, support iso, dir, real.
    # iso means that
  hetero_type: "real"
  level: 3  # The level of heterogeneous (0-5), 0 means no heterogeneous among clients.
  sleep_group_num: 1000  # The number of groups with different sleep time. 1 means all clients are the same.
  total_time: 1000  # The total sleep time of all clients, unit: second.
  fraction: 1  # The fraction of clients attending heterogeneous simulation.
  grouping_strategy: "greedy"  # The grouping strategy to handle system heterogeneity, support: random, greedy, slowest.
  initial_default_time: 5  # The estimated default training time for each training round, unit: second.
  default_time_momentum: 0.2  # The default momentum for default time update.

seed: 0  # The random seed.
```

### Default Config without Comments

```yaml
task_id: ""
data:
  root: "./data/"
  dataset: femnist
  split_type: "iid"
  
  min_size: 10
  data_amount: 0.05
  iid_fraction: 0.1
  user: False
  
  class_per_client: 1
  num_of_clients: 100
  train_test_split: 0.9  
  alpha: 0.5
  
  weights: NULL
  
model: lenet
test_mode: "test_in_client"
test_method: "average"

server:
  track: False
  rounds: 10
  clients_per_round: 5
  test_every: 1
  save_model_every: 10
  save_model_path: ""
  batch_size: 32
  test_all: True
  random_selection: True
  aggregation_strategy: "FedAvg"
  aggregation_content: "all"

client:
  track: False
  batch_size: 32
  test_batch_size: 5
  local_epoch: 10
  optimizer:
    type: "Adam"
    lr: 0.001
    momentum: 0.9
    weight_decay: 0
  seed: 0
  local_test: False

gpu: 0
distributed:
  backend: "nccl"
  init_method: ""
  world_size: 0
  rank: 0
  local_rank: 0

tracking:
  database: ""
  log_file: ""
  log_level: "INFO"
  metric_file: ""
  save_every: 1

resource_heterogeneous:
  simulate: False
  hetero_type: "real"
  level: 3
  sleep_group_num: 1000
  total_time: 1000
  fraction: 1
  grouping_strategy: "greedy"
  initial_default_time: 5
  default_time_momentum: 0.2

seed: 0  
```
