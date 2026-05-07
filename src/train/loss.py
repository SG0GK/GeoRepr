import torch
import torch.nn as nn


def r2(real, pred, eps=1e-10, axis=0):
    return 1.0 - (torch.mean(torch.square(real - pred), axis=axis) / (torch.var(real, axis=axis) + eps))


def mape(real, pred, eps=1e-10, axis=0):
    return 100.0 * torch.mean(torch.abs(real - pred) / (torch.abs(real) + eps), axis=axis)


def parse_loss(loss_type: str, **kwargs) -> nn.Module:
    loss_type = loss_type.lower()
    if loss_type == 'mse':
        return nn.MSELoss(**kwargs)
    elif loss_type == 'ce':
        return nn.CrossEntropyLoss(**kwargs)
    else:
        raise ValueError(f"Unsupported activation function: {loss_type}")


class MultiFeatureLoss(nn.Module):

    def __init__(self, config: list):
        """Selective losses for various indexes of tensor

        Args:
            config (list): list of losses and dictionaries
            titles (list): list of names of each channel
        """
        super().__init__()

        self.losses = nn.ModuleList([])
        self.idx_mins = list()
        self.idx_maxs = list()

        for item in config:
            loss_type = item['loss_type'] if 'loss_type' in item.keys() else 'MSE'
            self.losses.append(parse_loss(loss_type))
            self.idx_mins.append(item['idx_min'])
            self.idx_maxs.append(item['idx_max'])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        """Forward operation of applying multiple loss functions

        Args:
            y (torch.Tensor): real tensor [B, N]
            y_hat (torch.Tensor): pred tensor [B, N]

        Returns:
            torch.Tensor: [1,] - loss value
        """
        return sum([loss(y[:, idx_min:idx_max], y_hat[:, idx_min:idx_max]) for
                    idx_min, idx_max, loss in zip(self.idx_mins, self.idx_maxs, self.losses)])

    def mape(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        return mape(y, y_hat)

    def r2(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        return r2(y, y_hat)
