import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score, roc_auc_score


def mape_(real, pred, eps=1.0e-10, dim=0):
    score = torch.nanmean(
        100.0 * torch.abs(real - pred) / (torch.abs(real) + eps),
        dim=dim
    ).tolist()
    return score


class GeneralStatsLoss(nn.Module):
    dim = None
    titles = list()

    def __init__(self, loss: nn.Module, mask: torch.Tensor):
        super().__init__()
        self.loss = loss
        self.mask = mask
        self.mask_x = mask[1:] * mask[:-1]
        self.mask_y = mask[:, 1:] * mask[:, :-1]
        self.mask_z = mask[:, :, 1:] * mask[:, :, :-1]

    def calc_stats(self, X):
        pass

    def forward(self, X, y):
        y_hat = self.calc_stats(X)
        return self.loss(y_hat, y)

    def accuracy(self, X, y):
        y_hat = self.calc_stats(X)
        return mape_(y_hat, y)


class StatsLoss(GeneralStatsLoss):
    dim = 4
    titles = ['abruption_X', 'abruption_Y', 'abruption_Z', 'collector_frac']

    def calc_stats(self, X):
        a_1 = torch.mean(
            torch.abs(X[:, 0, 1:] - X[:, 0, :-1])[:, self.mask_x],
            axis=1
        )
        a_2 = torch.mean(
            torch.abs(X[:, 0, :, 1:] - X[:, 0, :, :-1])[:, self.mask_y],
            axis=1
        )
        a_3 = torch.mean(
            torch.abs(X[:, 0, :, :, 1:] - X[:, 0, :, :, :-1])[:, self.mask_z],
            axis=1
        )
        frac = torch.mean(X[:, 0, self.mask], axis=1)
        return torch.stack((a_1, a_2, a_3, frac), axis=1)


class VariogramNuggetLoss(GeneralStatsLoss):
    dim = 6
    titles = [
        'nugget_poro_X', 'nugget_poro_Y', 'nugget_poro_Z',
        'nugget_perm_X', 'nugget_perm_Y', 'nugget_perm_Z'
    ]

    def calc_stats(self, X):
        nugget_1_1 = 0.5 * torch.var(
            (X[:, 1, 1:] - X[:, 1, :-1])[:, self.mask_x],
            axis=1
        )
        nugget_1_2 = 0.5 * torch.var(
            (X[:, 1, :, 1:] - X[:, 1, :, :-1])[:, self.mask_y],
            axis=1
        )
        nugget_1_3 = 0.5 * torch.var(
            (X[:, 1, :, :, 1:] - X[:, 1, :, :, :-1])[:, self.mask_z],
            axis=1
        )
        nugget_2_1 = 0.5 * torch.var(
            (X[:, 2, 1:] - X[:, 2, :-1])[:, self.mask_x],
            axis=1
        )
        nugget_2_2 = 0.5 * torch.var(
            (X[:, 2, :, 1:] - X[:, 2, :, :-1])[:, self.mask_y],
            axis=1
        )
        nugget_2_3 = 0.5 * torch.var(
            (X[:, 2, :, :, 1:] - X[:, 2, :, :, :-1])[:, self.mask_z],
            axis=1
        )
        result = torch.stack((
            nugget_1_1, nugget_1_2, nugget_1_3,
            nugget_2_1, nugget_2_2, nugget_2_3
        ), axis=1)
        return result


class DerivativeLoss(GeneralStatsLoss):
    dim = 6
    titles = [
        'd_poro_X', 'd_poro_Y', 'd_poro_Z',
        'd_perm_X', 'd_perm_Y', 'd_perm_Z'
    ]

    def calc_stats(self, X):
        nugget_1_1 = 0.5 * torch.mean(torch.abs(
            (X[:, 1, 1:] - X[:, 1, :-1])[:, self.mask_x]
        ), axis=1)
        nugget_1_2 = 0.5 * torch.mean(torch.abs(
            (X[:, 1, :, 1:] - X[:, 1, :, :-1])[:, self.mask_y]
        ), axis=1)
        nugget_1_3 = 0.5 * torch.mean(torch.abs(
            (X[:, 1, :, :, 1:] - X[:, 1, :, :, :-1])[:, self.mask_z]
        ), axis=1)
        nugget_2_1 = 0.5 * torch.mean(torch.abs(
            (X[:, 2, 1:] - X[:, 2, :-1])[:, self.mask_x]
        ), axis=1)
        nugget_2_2 = 0.5 * torch.mean(torch.abs(
            (X[:, 2, :, 1:] - X[:, 2, :, :-1])[:, self.mask_y]
        ), axis=1)
        nugget_2_3 = 0.5 * torch.mean(torch.abs(
            (X[:, 2, :, :, 1:] - X[:, 2, :, :, :-1])[:, self.mask_z]
        ), axis=1)
        result = torch.stack((
            nugget_1_1, nugget_1_2, nugget_1_3,
            nugget_2_1, nugget_2_2, nugget_2_3
        ), axis=1)
        return result


class MetaStatsLoss(nn.Module):

    def __init__(self, mask: torch.Tensor, loss_types=list(), weights=None, losses_red=None):
        super().__init__()
        self.losses = nn.ModuleList()
        self.weights = [1.0 for _ in range(len(loss_types))] if weights is None else weights
        self.losses_red = [nn.MSELoss(reduction='mean') for _ in range(len(loss_types))] if losses_red is None else losses_red
        self.dims = [0]
        self.titles = list()
        for i, loss_type in enumerate(loss_types):
            if loss_type.lower() == 'statsloss':
                self.losses.append(StatsLoss(loss=self.losses_red[i], mask=mask))
            elif loss_type.lower() == 'variogramnuggetsloss':
                self.losses.append(VariogramNuggetLoss(loss=self.losses_red[i], mask=mask))
            elif loss_type.lower() == 'derivativeloss':
                self.losses.append(DerivativeLoss(loss=self.losses_red[i], mask=mask))
            else:
                raise ValueError(f'Unknown loss type {loss_type}')
            self.dims.append(self.losses[i].dim)
            self.titles = self.titles + self.losses[i].titles
        self.dim = np.sum(self.dims)
        self.dims = np.cumsum(self.dims)

    def calc_stats(self, X):
        stats = list()
        for loss in self.losses:
            stats.append(loss.calc_stats(X))
        return torch.hstack(stats)

    def accuracy(self, X, y):
        y_hat = self.calc_stats(X)
        return {f'stats/{title}': value for title, value in zip(self.titles, mape_(y, y_hat))}

    def forward(self, X, y):
        result = sum([
            self.weights[i] * loss(X, y[:, self.dims[i]:self.dims[i+1]]) if self.weights[i] > 0 else 0 for i, loss in enumerate(self.losses)
        ])
        return result
