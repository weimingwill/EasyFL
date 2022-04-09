# Tutorial 1: High-level APIs

EasyFL provides three types of high-level APIs: **initialization**, **registration**, and **execution**.
The initialization API initializes EasyFL with configurations. 
Registration APIs register customized components into the platform. 
Execution APIs start federated learning process. 
These APIs are listed in the table below.

| API Name      | Description | Category 
| :---        |    :----:   | :--- |
| init(config) | Initialize EasyFL with configurations | Initialization | 
| register_dataset(train, test, val) | Register a customized dataset | Registration | 
| register_model(model) | Register a customized model | Registration | 
| register_server(server) | Register a customized server | Registration |
| register_client(client) | Register a customized client | Registration |
| run() | Start federated learning for standalone and distributed training | Execution |
| start_server() | Start server service for remote training | Execution |
| start_client() | Start client service for remote training | Execution |


`init(config):` Initialize EasyFL with provided configurations (`config`) or default configurations if not specified.  
These configurations determine the training hardware and hyperparameters.

`register_<module>:` Register customized modules to the system. 
EasyFL supports the registration of customized datasets, models, server, and client, replacing the default modules in FL training. In the experimental phase, users can register newly developed algorithms to understand their performance.

`run, start_<server/client>:` The APIs are commands to trigger execution. 
`run()` starts FL using standalone training or distributed training. 
 `start_server` and `start_client` start the server and client services to communicate remotely with `args` variables for configurations specific to remote training, such as the endpoint addresses.

Next, we introduce how to use these APIs with examples.

## Standalone Training Example

_**Standalone training**_ means that federated learning (FL) training is run on a single hardware device, such as your personal computer and a single GPU.
_**Distributed training**_ means conducting FL with multiple GPUs to speed up training.
Running distributed training is similar to standalone training, except that we need to configure the number of GPUs and the distributed settings. 
We explain more on distributed training in [another note](distributed_training.md) and focus on standalone training example here.  

To run any federated learning process, we need to first call the initialization API and then use the execution API. Registration is optional.

The simplest way is to run with the default setup. 
```python
import easyfl
# Initialize federated learning with default configurations.
easyfl.init()
# Execute federated learning training.
easyfl.run()
```

You can run it with specified configurations. 
```python
import easyfl

# Customized configuration.
config = {
    "data": {"dataset": "cifar10", "num_of_clients": 1000},
    "server": {"rounds": 5, "clients_per_round": 2, "test_all": False},
    "client": {"local_epoch": 5},
    "model": "resnet18",
    "test_mode": "test_in_server",
}
# Initialize federated learning with default configurations.
easyfl.init(config)
# Execute federated learning training.
easyfl.run()
```

You can also run federated learning with customized datasets, model, server and client implementations.

Note: `registration` must be done before `initialization`.

```python
import easyfl
from easyfl.client import BaseClient

# Inherit BaseClient to implement customized client operations.
class CustomizedClient(BaseClient):
    def __init__(self, cid, conf, train_data, test_data, device, **kwargs):
        super(CustomizedClient, self).__init__(cid, conf, train_data, test_data, device, **kwargs)
        pass  # more initialization of attributes.

    def train(self, conf, device):
        pass # Implement customized training method, overwriting the default one.

# Register customized client.
easyfl.register_client(CustomizedClient)
# Initialize federated learning with default configurations.
easyfl.init()
# Execute federated learning training.
easyfl.run()
```

## Remote Training Example

_**Remote training**_ is the scenario where the server and the clients are running on different devices.
We explain more on remote training in [another note](remote_training.md). 
Here we provide examples on how to start client and server services using the APIs.

Start remote server.
```python
import easyfl
# Configurations for the remote server.
conf = {"is_remote": True, "local_port": 22999}
# Initialize only the configuration.
easyfl.init(conf, init_all=False)
# Start remote server service.
# The remote server waits to be connected with the remote client.
easyfl.start_server()
```

Start remote client.
```python
import easyfl
# Configurations for the remote client.
conf = {"is_remote": True, "local_port": 23000}
# Initialize only the configuration.
easyfl.init(conf, init_all=False)
# Start remote client service.
# The remote client waits to be connected with the remote server.
easyfl.start_client()
```

We expose two additional APIs that wrap starting remote services with customized components.
They are `start_remote_server` and `start_remote_client`. 
You can explore more in the API documentation.
