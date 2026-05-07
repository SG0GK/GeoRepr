import torch
import torch.nn as nn


def get_activation_function(activation_name: str, **kwargs) -> nn.Module:
    """
    Returns a PyTorch activation function module based on the input string.

    Args:
        activation_name: A string representing the name of the desired activation function.
                           Supported names: 'relu', 'sigmoid', 'tanh', 'elu', 'leaky_relu', 'softmax',
                           'log_softmax', 'gelu', 'silu' (also known as swish).
                           Case-insensitive.
        **kwargs: Keyword arguments to be passed to the activation function constructor.
                  For example, for LeakyReLU, you can pass `negative_slope=0.2`.

    Returns:
        A PyTorch activation function module (e.g., nn.ReLU(), nn.Sigmoid()).

    Raises:
        ValueError: If the activation_name is not recognized.
    """

    activation_name = activation_name.lower()  # Case-insensitive matching

    if activation_name == 'relu':
        return nn.ReLU(**kwargs)
    elif activation_name == 'sigmoid':
        return nn.Sigmoid(**kwargs)
    elif activation_name == 'tanh':
        return nn.Tanh(**kwargs)
    elif activation_name == 'elu':
        return nn.ELU(**kwargs)
    elif activation_name == 'leaky_relu':
        return nn.LeakyReLU(**kwargs)
    elif activation_name == 'softmax':
        # Allows setting dim in kwargs, defaulting to 1
        dim = kwargs.pop('dim', 1)
        # Specify dim=1 for typical channel-wise softmax
        return nn.Softmax(dim=dim, **kwargs)
    elif activation_name == 'log_softmax':
        # Allows setting dim in kwargs, defaulting to 1
        dim = kwargs.pop('dim', 1)
        # Specify dim=1 for typical channel-wise log softmax
        return nn.LogSoftmax(dim=dim, **kwargs)
    elif activation_name == 'gelu':
        return nn.GELU(**kwargs)
    elif activation_name == 'silu' or activation_name == 'swish':
        # Renamed SiLU from Swish in newer PyTorch versions
        return nn.SiLU(**kwargs)
    elif activation_name == 'identity':
        return nn.Identity(**kwargs)
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")


class MultiActivationLayer(nn.Module):

    def __init__(self, config: list):
        """Selective activations for various indexes of tensor

        Args:
            config (list): list of activations functions, for example:
            activation_config = [
                {
                    'type': 'Softmax',
                    'idx_min': 0,
                    'idx_max': 4,
                },
                {
                    'type': 'ReLU',
                    'idx_min': 4,
                    'idx_max': 8
                },
            ]
        """
        super().__init__()
        self.activations = nn.ModuleList([])
        self.idx_mins = list()
        self.idx_maxs = list()
        for item in config:
            kwargs = item['kwargs'] if 'kwargs' in item.keys() else dict()
            self.activations.append(get_activation_function(item['type'], **kwargs))
            self.idx_mins.append(item['idx_min'])
            self.idx_maxs.append(item['idx_max'])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward operation of applying multiple activations

        Args:
            X (torch.Tensor): 2D Tensor [B, N]

        Returns:
            torch.Tensor: 2D Tensor [B, N] with activations applied
        """
        for idx_min, idx_max, activation in zip(self.idx_mins, self.idx_maxs, self.activations):
            X[:, idx_min:idx_max] = activation(X[:, idx_min:idx_max])
        return X
