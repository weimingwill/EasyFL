## High-level Introduction

EasyFL provides numerous existing models and datasets. Models include LeNet, RNN, VGG9, and ResNet. Datasets include Femnist, Shakespeare, CIFAR-10, and CIFAR-100. 
This note will present how to start training with these existing models and standard datasets.

EasyFL provides three types of high-level APIs: **registration**, **initialization**, and **execution**.
Registration is for registering customized components, which we will introduce in the following notes.
In this note, we focus on **initialization** and **execution**.

## Simplest Run

We can run federated learning with only two lines of code (not counting the import statement).
It executes training with default configurations: simulating 100 clients with the FEMNIST dataset and randomly selecting 5 clients for training in each training round.
We explain more about the configurations in [another note](tutorials/config.md).

Note: we package default partitioning of Femnist data to avoid downloading the whole dataset.

```python
import easyfl

# Initialize federated learning with default configurations.
easyfl.init()
# Execute federated learning training.
easyfl.run()
```

## Run with Configurations

You can specify configurations to overwrite the default configurations.

```python
import easyfl

# Customized configuration.
config = {
    "data": {"dataset": "cifar10", "split_type": "class", "num_of_clients": 100},
    "server": {"rounds": 5, "clients_per_round": 2},
    "client": {"local_epoch": 5},
    "model": "resnet18",
    "test_mode": "test_in_server",
}
# Initialize federated learning with default configurations.
easyfl.init(config)
# Execute federated learning training.
easyfl.run()
```

In the example above, we run training with model ResNet-18 and CIFAR-10 dataset that is partitioned into 100 clients by label `class`.
It runs training with 2 clients per round for 5 rounds. In each round, each client trains 5 epochs.
