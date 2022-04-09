# Tutorial 4: Models

EasyFL supports numerous models and allows you to customize the model.

## Out-of-the-box Models

To use these models, you can set configurations `model: <model_name>`. 

We currently provide `lenet`, `resnet`, `resnet18`, `resnet50`,`vgg9`, and `rnn`.

## Customized Models

EasyFL allows training with a wide range of models by providing the flexibility to customize models. 
You can customize and register models in two ways: register as a class and register as an instance.
Either way, the basic is to **inherit and implement the `easyfl.models.BaseModel`**. 

### Register as a Class

In the example below, we implement and conduct FL training with a `CustomizedModel`. 

It is applicable when the model does not require extra arguments to initialize.

```python
from torch import nn
import torch.nn.functional as F
import easyfl
from easyfl.models import BaseModel

# Define a customized model class.
class CustomizedModel(BaseModel):
    def __init__(self):
        super(CustomizedModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 224, padding=(2, 2))
        self.conv2 = nn.Conv2d(32, 64, 5, padding=(2, 2))
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 42)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# Register the customized model class.
easyfl.register_model(CustomizedModel)
# Initialize EasyFL.
easyfl.init()
# Execute FL training.
easyfl.run()
```

### Register as an Instance

When the model requires arguments for initialization, we can implement and register a model instance. 

```python
from torch import nn
import torch.nn.functional as F
import easyfl
from easyfl.models import BaseModel

# Define a customized model class.
class CustomizedModel(BaseModel):
    def __init__(self, num_class):
        super(CustomizedModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 224, padding=(2, 2))
        self.conv2 = nn.Conv2d(32, 64, 5, padding=(2, 2))
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_class)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Register the customized model instance.
easyfl.register_model(CustomizedModel(num_class=10))
# Initialize EasyFL.
easyfl.init()
# Execute FL training.
easyfl.run()
```