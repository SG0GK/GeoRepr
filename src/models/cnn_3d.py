import torch
import torch.nn as nn
from .utils import parse_module


class Sequence(nn.Module):

    def __init__(self, config: list):
        """Sequence of modules. Modules include CNN encoders or decoders, activation functions, pooling, repeat etc.

        Args:
            config (list): list of modules configurations
        """
        super().__init__()

        self.sequence = nn.Sequential()

        for description in config:
            self.sequence += parse_module(description)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward passing through all modules

        Args:
            X (torch.Tensor): input data

        Returns:
            torch.Tensor: output data
        """
        X = self.sequence(X)
        return X

    def get_representations(self, X: torch.Tensor, idx: int) -> torch.Tensor:
        """Forward passing through all modules before module number idx

        Args:
            X (torch.Tensor): input data
            idx (int): number of layer on which we want to get representations

        Returns:
            torch.Tensor: output representations
        """
        X = self.sequence[:idx](X)
        return X


def parse_cnn(config: dict) -> Sequence:
    """Getting 3D CNN with Linear layers
    Args:
        config (dict): description of layers, channels etc.

    Returns:
        AutoEncoder: autoencoder built by description
    """
    config_encoder = [
        {
            'module_name': 'SimpleEncoder3D',
            'kwargs': {
                'block_out_channels': config['cnn3d']['block_out_channels'],
                'block_out_types': config['cnn3d']['block_out_types_down'],
                'out_channels': config['cnn3d']['latent_channels'],
                'num_groups': config['cnn3d']['num_groups'],
                'scales_d': config['cnn3d']['scales_d'],
                'scales_w': config['cnn3d']['scales_w'],
                'scales_h': config['cnn3d']['scales_h']
            }
        },
        {
            'module_name': 'MLP',
            'kwargs': {
                'block_out_channels': config['mlp']['block_out_channels'],
                'activations': config['mlp']['activations'],
            }
        },
        # {
        #     'module_name': 'MultiActivationLayer',
        #     'kwargs': {
        #         'config': config['activations']
        #     }
        # }
    ]
    cnn = Sequence(config_encoder)
    print('Model:\n')
    print(cnn)
    return cnn
