import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from easyfl.datasets.data import CIFAR100
from eval_dataset import get_semi_supervised_data_loaders
from model import get_encoder_network


def test_whole(resnet, logreg, device, test_loader, model_path):
    print("### Calculating final testing performance ###")
    resnet.eval()
    logreg.eval()
    metrics = defaultdict(list)
    for step, (h, y) in enumerate(test_loader):
        h = h.to(device)
        y = y.to(device)
        with torch.no_grad():
            outputs = logreg(resnet(h))

        # calculate accuracy and save metrics
        accuracy = (outputs.argmax(1) == y).sum().item() / y.size(0)
        metrics["Accuracy/test"].append(accuracy)

    print(f"Final test performance: " + "\t".join([f"{k}: {np.array(v).mean()}" for k, v in metrics.items()]))
    return np.array(metrics["Accuracy/test"]).mean()


def finetune_internal(model, epochs, label_loader, test_loader, num_class, device, lr=3e-3):
    model = model.to(device)
    num_features = model.feature_dim

    n_classes = num_class  # e.g. CIFAR-10 has 10 classes

    # fine-tune model
    logreg = nn.Sequential(nn.Linear(num_features, n_classes))
    logreg = logreg.to(device)

    # loss / optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=logreg.parameters(), lr=lr)

    # Train fine-tuned model
    model.train()
    logreg.train()
    for epoch in range(epochs):
        metrics = defaultdict(list)
        for step, (h, y) in enumerate(label_loader):
            h = h.to(device)
            y = y.to(device)
            outputs = logreg(model(h))
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # calculate accuracy and save metrics
            accuracy = (outputs.argmax(1) == y).sum().item() / y.size(0)
            metrics["Loss/train"].append(loss.item())
            metrics["Accuracy/train"].append(accuracy)

        if epoch % 100 == 0:
            print("======epoch {}======".format(epoch))
            test_whole(model, logreg, device, test_loader, "test_whole")
    final_accuracy = test_whole(model, logreg, device, test_loader, "test_whole")
    print(metrics)
    return final_accuracy


class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, projection_size),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", type=str, help="cifar10/cifar100.")
    parser.add_argument('--model', default='simsiam', type=str, help='name of the network')
    parser.add_argument("--encoder_network", default="resnet18", type=str, help="Encoder network architecture.")
    parser.add_argument("--model_path", required=True, type=str, help="Path to pre-trained model (e.g. model-10.pt)")
    parser.add_argument("--image_size", default=32, type=int, help="Image size")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="Initial learning rate.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size for training.")
    parser.add_argument("--num_epochs", default=100, type=int, help="Number of epochs to train for.")
    parser.add_argument("--data_distribution", default="class", type=str, help="class/iid")
    parser.add_argument("--label_ratio", default=0.01, type=float, help="ratio of labeled data for fine tune")
    parser.add_argument('--class_per_client', default=2, type=int,
                        help='for non-IID setting, number of class each client, based on CIFAR10')
    parser.add_argument("--use_MLP", action='store_true',
                        help="whether use MLP, if use, one hidden layer MLP, else, Linear Layer.")
    parser.add_argument("--num_workers", default=8, type=int,
                        help="Number of data loading workers (caution with nodes!)")
    args = parser.parse_args()

    print(args)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    print('==> Preparing data..')
    class_per_client = args.class_per_client
    n_classes = 10
    if args.dataset == CIFAR100:
        class_per_client = 10 * class_per_client
        n_classes = 100

    train_loader, test_loader = get_semi_supervised_data_loaders(args.dataset,
                                                                 args.data_distribution,
                                                                 class_per_client,
                                                                 args.label_ratio,
                                                                 args.batch_size,
                                                                 args.num_workers)

    print('==> Building model..')
    resnet = get_encoder_network(args.model, args.encoder_network)
    resnet.load_state_dict(torch.load(args.model_path, map_location=device))
    resnet = resnet.to(device)
    num_features = list(resnet.children())[-1].in_features
    resnet.fc = nn.Identity()

    # fine-tune model
    if args.use_MLP:
        logreg = MLP(num_features, n_classes, 4096)
        logreg = logreg.to(device)
    else:
        logreg = nn.Sequential(nn.Linear(num_features, n_classes))
        logreg = logreg.to(device)

    # loss / optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=logreg.parameters(), lr=args.learning_rate)

    # Train fine-tuned model
    logreg.train()
    resnet.train()
    accs = []
    for epoch in range(args.num_epochs):
        print("======epoch {}======".format(epoch))
        metrics = defaultdict(list)
        for step, (h, y) in enumerate(train_loader):
            h = h.to(device)
            y = y.to(device)

            outputs = logreg(resnet(h))

            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # calculate accuracy and save metrics
            accuracy = (outputs.argmax(1) == y).sum().item() / y.size(0)
            metrics["Loss/train"].append(loss.item())
            metrics["Accuracy/train"].append(accuracy)

        print(f"Epoch [{epoch}/{args.num_epochs}]: " + "\t".join(
            [f"{k}: {np.array(v).mean()}" for k, v in metrics.items()]))

        if epoch % 1 == 0:
            acc = test_whole(resnet, logreg, device, test_loader, args.model_path)
            if epoch <= 100:
                accs.append(acc)
    test_whole(resnet, logreg, device, test_loader, args.model_path)
    print(args.model_path)
    print(f"Best one for 100 epoch is {max(accs):.4f}")
