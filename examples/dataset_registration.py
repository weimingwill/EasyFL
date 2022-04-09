import torch.nn.functional as F
from torch import nn
from torchvision import transforms

import easyfl
from easyfl.datasets import FederatedImageDataset
from easyfl.models import BaseModel


class TestModel(BaseModel):
    def __init__(self):
        super(TestModel, self).__init__()
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


default_transfrom = transforms.Compose([
    # images are of different size, reshape them to same size. Up to you to decide what size to use.
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # convert image to torch tensor
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])  # normalizing images helps improve convergence
root = "/Users/sg0014000170m/Downloads/shopee-product-detection-student/train/train/train"
train_data = FederatedImageDataset(root,
                                   simulated=False,
                                   do_simulate=True,
                                   transform=default_transfrom,
                                   num_of_clients=100)
test_data = FederatedImageDataset(root,
                                  simulated=False,
                                  do_simulate=False,
                                  transform=default_transfrom,
                                  num_of_clients=100)

easyfl.register_model(TestModel)
easyfl.register_dataset(train_data, test_data)
easyfl.init()
easyfl.run()
