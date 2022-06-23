import torch

import transform
from model import SimSiam, MoCo


def get_transformation(model):
    if model == SimSiam:
        transformation = transform.SimSiamTransform
    elif model == MoCo:
        transformation = transform.MoCoTransform
    else:
        transformation = transform.SimCLRTransform
    return transformation


def calculate_model_distance(m1, m2):
    distance, count = 0, 0
    d1, d2 = m1.state_dict(), m2.state_dict()
    for name, param in m1.named_parameters():
        if 'conv' in name and 'weight' in name:
            distance += torch.dist(d1[name].detach().clone().view(1, -1), d2[name].detach().clone().view(1, -1), 2)
            count += 1
    return distance / count


def normalize(arr):
    maxx = max(arr)
    minn = min(arr)
    diff = maxx - minn
    if diff == 0:
        return arr
    return [(x - minn) / diff for x in arr]
