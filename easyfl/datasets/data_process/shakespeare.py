import numpy as np
import torch

from easyfl.datasets.data_process.language_utils import word_to_indices, letter_to_vec


def process_x(raw_x_batch):
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    x_batch = np.array(x_batch)
    return torch.LongTensor(x_batch)


def process_y(raw_y_batch):
    y_batch = [np.argmax(letter_to_vec(c)) for c in raw_y_batch]
    return torch.LongTensor(y_batch)
