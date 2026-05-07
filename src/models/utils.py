import torch.nn as nn

from .resnet_3d import SimpleEncoder3D
from .activations import get_activation_function, MultiActivationLayer
from .mlp import MLP


def parse_module(config: dict) -> nn.Module:
    """Parses module by description.
    Modules include CNN encoders or decoders, activation functions, pooling, repeat etc.

    Args:
        config (dict): module name and kwargs

    Raises:
        ValueError: in case when module name is unknown

    Returns:
        nn.Module: selected module
    """
    module_name = config['module_name'].lower()
    kwargs = config['kwargs'] if 'kwargs' in config.keys() else dict()
    if module_name == 'simpleencoder3d':
        return SimpleEncoder3D(**kwargs).conv
    elif module_name == 'activation':
        return get_activation_function(**kwargs)
    elif module_name == 'mlp':
        return MLP(**kwargs).layers
    elif module_name == 'multiactivationlayer':
        return MultiActivationLayer(**kwargs)
    else:
        raise ValueError(f"Unknown module name: {module_name}")
