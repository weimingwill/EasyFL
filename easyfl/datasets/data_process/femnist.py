import torch


def process_x(raw_x_batch):
    raw_x_batch = torch.FloatTensor(raw_x_batch)
    return raw_x_batch.view(-1, 1, 28, 28)


def process_y(raw_y_batch):
    return torch.LongTensor(raw_y_batch)
