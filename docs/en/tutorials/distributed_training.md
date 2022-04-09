# Tutorial 6: Distributed Training

EasyFL enables federated learning (FL) training over multiple GPUs. We define the following variables to further illustrate the idea:
* K: the number of clients who participated in training each round
* N: the number of available GPUs

When _K == N_, each selected client is allocated to a GPU to train.

When _K > N_, multiple clients are allocated to a GPU, then they execute training sequentially in the GPU.

When _K < N_, you can adjust to use fewer GPUs in training.

We make it easy to use distributed training. You just need to modify the configs, without changing the core implementations.
In particular, you need to set the number of GPUs in `gpu` and specific distributed settings in the `distributed` configs.

The following is an example of distributed training on a GPU cluster managed by _slurm_.

```python
import easyfl
from easyfl.distributed import slurm

# Get the distributed settings.
rank, local_rank, world_size, host_addr = slurm.setup()
# Set the distributed training settings.
config = {
    "gpu": world_size,
    "distributed": {
        "rank": rank, 
        "local_rank": local_rank, 
        "world_size": world_size, 
        "init_method": host_addr
    },
}
# Initialize EasyFL.
easyfl.init(config)
# Execute training with distributed training.
easyfl.run()
```

We will further provide scripts to set up distributed training using `multiprocess`. 
Pull requests are also welcomed.

