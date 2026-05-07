import torch
import torch.nn as nn
from .activations import get_activation_function


class MLP(nn.Module):

    def __init__(self, block_out_channels=list(), activations=list()):
        super().__init__()

        self.layers = nn.Sequential(nn.Flatten())

        n_layers = 0

        for i in range(len(block_out_channels) - 1):
            in_channels = block_out_channels[i]
            out_channels = block_out_channels[i+1] if i != len(block_out_channels) - 1 else block_out_channels[i]
            self.layers.append(nn.Linear(in_channels, out_channels))
            n_layers += 1
            if len(activations) > n_layers:
                self.layers.append(get_activation_function(activations[n_layers - 1]))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.layers(X)
        return X

    def get_representations(self, X: torch.Tensor) -> torch.Tensor:
        X = self.layers[:-1](X)
        return X
