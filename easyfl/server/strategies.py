import copy

import torch


def federated_averaging(models, weights):
    """Compute weighted average of model parameters and persistent buffers.
    Using state_dict of model, including persistent buffers like BN stats.

    Args:
        models (list[nn.Module]): List of models to average.
        weights (list[float]): List of weights, corresponding to each model.
            Weights are dataset size of clients by default.
    Returns
        nn.Module: Weighted averaged model.
    """
    if models == [] or weights == []:
        return None

    model, total_weights = weighted_sum(models, weights)
    model_params = model.state_dict()
    with torch.no_grad():
        for name, params in model_params.items():
            model_params[name] = torch.div(params, total_weights)
    model.load_state_dict(model_params)
    return model


def federated_averaging_only_params(models, weights):
    """Compute weighted average of model parameters. Use model parameters only.

    Args:
        models (list[nn.Module]): List of models to average.
        weights (list[float]): List of weights, corresponding to each model.
            Weights are dataset size of clients by default.
    Returns
        nn.Module: Weighted averaged model.
    """
    if models == [] or weights == []:
        return None

    model, total_weights = weighted_sum_only_params(models, weights)
    model_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, params in model_params.items():
            model_params[name].set_(model_params[name] / total_weights)

    return model


def weighted_sum(models, weights):
    """Compute weighted sum of model parameters and persistent buffers.
    Using state_dict of model, including persistent buffers like BN stats.

    Args:
        models (list[nn.Module]): List of models to average.
        weights (list[float]): List of weights, corresponding to each model.
            Weights are dataset size of clients by default.
    Returns
        nn.Module: Weighted averaged model.
        float: Sum of weights.
    """
    if models == [] or weights == []:
        return None
    model = copy.deepcopy(models[0])
    model_sum_params = copy.deepcopy(models[0].state_dict())

    with torch.no_grad():
        for name, params in model_sum_params.items():
            params *= weights[0]
            for i in range(1, len(models)):
                model_params = dict(models[i].state_dict())
                params += model_params[name] * weights[i]
            model_sum_params[name] = params
    model.load_state_dict(model_sum_params)
    return model, sum(weights)


def weighted_sum_only_params(models, weights):
    """Compute weighted sum of model parameters. Use model parameters only.

    Args:
        models (list[nn.Module]): List of models to average.
        weights (list[float]): List of weights, corresponding to each model.
            Weights are dataset size of clients by default.
    Returns
        nn.Module: Weighted averaged model.
        float: Sum of weights.
    """
    if models == [] or weights == []:
        return None

    model_sum = copy.deepcopy(models[0])
    model_sum_params = dict(model_sum.named_parameters())

    with torch.no_grad():
        for name, params in model_sum_params.items():
            params *= weights[0]
            for i in range(1, len(models)):
                model_params = dict(models[i].named_parameters())
                params += model_params[name] * weights[i]
            model_sum_params[name].set_(params)
    return model_sum, sum(weights)


def equal_weight_averaging(models):
    if models == []:
        return None

    model_avg = copy.deepcopy(models[0])
    model_avg_params = dict(model_avg.named_parameters())

    with torch.no_grad():
        for name, params in model_avg_params.items():
            for i in range(1, len(models)):
                model_params = dict(models[i].named_parameters())
                params += model_params[name]
            model_avg_params[name].set_(params / len(models))
    return model_avg
