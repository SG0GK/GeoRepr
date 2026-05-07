import torch
import torch.nn as nn


class Cropper(nn.Module):

    def __init__(self, mask):
        super().__init__()
        self.mask = mask[None, None, :, :, :]

    def forward(self, X):
        self.mask = self.mask.to(X.device)
        X = torch.where(self.mask, X, 0.0)
        return X


class NTGCropper(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, X):
        mask = X[:, 0].unsqueeze(1).type(torch.bool)
        X = torch.where(mask, X, 0.0)
        return X
