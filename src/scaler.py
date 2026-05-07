import torch
import numpy as np


def apply_sequence(X: torch.Tensor, sequence: list) -> torch.Tensor:
    """Apply sequence of transformations to the tensor

    Args:
        X (torch.Tensor): input tensor
        sequence (list): description of transformations and their kwargs

    Raises:
        ValueError: in case when sequence element name in unknown

    Returns:
        torch.Tensor: transformed tensor
    """
    for iter in sequence:
        if iter['iter'] == 'invert':
            # print('Applying invert')
            X = (~X.type(torch.bool)).type(torch.float)
        elif iter['iter'] == 'scale':
            # print(f"Applying divide by {iter['divisor']}")
            X = X / iter['divisor']
        elif iter['iter'] == 'subtract':
            # print(f"Applying subtract by {iter['subtrahend']}")
            X = X - iter['subtrahend']
        elif iter['iter'] == 'log':
            # print("Applying log")
            X = torch.log(X)
        else:
            raise ValueError(f"Unknown iter: {iter['iter']}")
    return X


def apply_backward_sequence(X: torch.Tensor, sequence: list) -> torch.Tensor:
    """Apply reverse sequence of transformations to the tensor

    Args:
        X (torch.Tensor): transformed or generated tensor
        sequence (list): description of transformations and their kwargs

    Raises:
        ValueError: in case when sequence element name in unknown

    Returns:
        torch.Tensor: untransformed tensor
    """
    for iter in sequence[::-1]:
        if iter['iter'] == 'invert':
            X = torch.round(X)
            X = (~X.type(torch.bool)).type(torch.float)
        elif iter['iter'] == 'scale':
            X = X * iter['divisor']
        elif iter['iter'] == 'subtract':
            X = X + iter['subtrahend']
        elif iter['iter'] == 'log':
            X = torch.exp(X)
        else:
            raise ValueError(f"Unknown iter: {iter['iter']}")
    return X


class Scaler(object):

    def __init__(self, config: dict):
        """Scaler applies transformations described by sequences.

        Args:
            config (dict): scaler config - a dict with 0, 1, 2, ... as values
                that represent different channels.
        """
        self.config = config

    def scale(self, X: torch.Tensor) -> torch.Tensor:
        """Scaling tensor, applying sequences for each of its channel

        Args:
            X (torch.Tensor): Original tensor

        Returns:
            torch.Tensor: Scaled tensor
        """
        for i, value in self.config.items():
            # print(f"Working with {value['name']}")
            X[:, i] = apply_sequence(X[:, i], value['sequence'])
            # print(f"dim={i}, min={X[:, i].min()}, max={X[:, i].max()}, mean={X[:, i].mean()}, std={X[:, i].std()}")
        return X

    def unscale(self, X: torch.Tensor) -> torch.Tensor:
        """Undoes the scale operation, brings it back to data min max std etc.

        Args:
            X (torch.Tensor): Scaled or generated tensor

        Returns:
            torch.Tensor: Unscaled tensor
        """
        for i, value in self.config.items():
            X[:, i] = apply_backward_sequence(X[:, i], value['sequence'])
        return X


class YScaler(object):

    def __init__(self, Y, group_list, a=-1.0, b=1.0):
        self.mins = torch.zeros(Y.shape[1])
        self.maxs = torch.zeros(Y.shape[1])
        self.idxs = torch.zeros(Y.shape[1], dtype=torch.bool)
        total_keys = list(Y.keys())
        for group in group_list:
            name = group['name']
            keys = [key for key in list(Y.keys()) if name in key]
            idxs = torch.tensor([idx for idx, key in enumerate(list(Y.keys())) if name in key])
            if len(keys) > 0:
                min_ = np.nanmin(Y[keys].values) if 'min' not in group.keys() else group['min']
                max_ = np.nanmax(Y[keys].values) if 'max' not in group.keys() else group['max']
                self.mins[idxs] = min_
                self.maxs[idxs] = max_
                self.idxs[idxs] = True
                total_keys = [key for key in total_keys if key not in keys]
                print(f'Will scale {name} columns from {min_} to {max_}')
        self.mins[~self.idxs] = torch.from_numpy(np.nanmin(Y[total_keys].values, axis=0).astype(np.float32))
        self.maxs[~self.idxs] = torch.from_numpy(np.nanmax(Y[total_keys].values, axis=0).astype(np.float32))
        self.a = a
        self.b = b
        self.coeffs_r = (self.maxs - self.mins) / (self.b - self.a)
        self.coeffs = 1.0 / self.coeffs_r
        self.bias_r = self.mins - self.a * self.coeffs_r
        self.bias = self.a - self.mins * self.coeffs

    def scale(self, Y):
        return self.coeffs * Y + self.bias

    def unscale(self, Y):
        return self.coeffs_r * Y + self.bias_r

    def clip_unscale(self, Y):
        return self.unscale(Y.clip(min=self.a, max=self.b))

    def to(self, device):
        self.coeffs = self.coeffs.to(device)
        self.coeffs_r = self.coeffs_r.to(device)
        self.bias = self.bias.to(device)
        self.bias_r = self.bias_r.to(device)
