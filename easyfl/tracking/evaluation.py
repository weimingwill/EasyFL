def model_size(model, param_size=32):
    """Calculate the model parameter sizes, including non-trainable parameters

    Args:
        model (nn.Module): A PyTorch model.
        param_size (int): The size of a parameter, default using float32.

    Returns:
        float: The model size in MB.
    """
    # sum(p.numel() for p in model.parameters() if p.requires_grad) for only trainable parameters
    params = sum(p.numel() for p in model.parameters())
    return bit_to_megabyte(params * param_size)


def bit_to_megabyte(bits):
    return bits / (8 * 1024 * 1024)
