"""
BYOL (Bootstrap Your Own Latent) for 3D Images

This package implements BYOL self-supervised learning for 3D volumetric data.
"""

from .model import (
    BYOL,
    BYOLEncoder,
    BYOLOnlineNetwork,
    BYOLTargetNetwork,
    compute_byol_loss,
    byol_loss_fn
)

from .trainer import BYOLTrainer

from .plotting import (
    load_history,
    plot_loss_curves,
    plot_learning_rate,
    plot_target_momentum,
    plot_training_summary,
    plot_smoothed_loss,
    plot_loss_statistics,
    create_all_plots,
    compare_runs
)

from .augmentation import (
    RandomCrop3D,
    RandomFlip3D,
    GaussianNoise3D,
    RandomIntensityScale3D,
    RandomIntensityShift3D,
    ClampValues,
    RandomRotation90_3D,
    Compose,
    get_byol_augmentation,
    get_light_augmentation,
    get_medium_augmentation,
    get_strong_augmentation
)

from .data_utils import (
    BYOLDatasetWrapper,
    BYOLDataLoaderFactory,
    prepare_byol_dataloaders,
    wrap_existing_dataloader
)

__version__ = '1.0.0'

__all__ = [
    # Model classes
    'BYOL',
    'BYOLEncoder',
    'BYOLOnlineNetwork',
    'BYOLTargetNetwork',
    'compute_byol_loss',
    'byol_loss_fn',
    
    # Trainer
    'BYOLTrainer',
    
    # Plotting functions
    'load_history',
    'plot_loss_curves',
    'plot_learning_rate',
    'plot_target_momentum',
    'plot_training_summary',
    'plot_smoothed_loss',
    'plot_loss_statistics',
    'create_all_plots',
    'compare_runs',
    
    # Augmentation
    'RandomCrop3D',
    'RandomFlip3D',
    'GaussianNoise3D',
    'RandomIntensityScale3D',
    'RandomIntensityShift3D',
    'ClampValues',
    'RandomRotation90_3D',
    'Compose',
    'get_byol_augmentation',
    'get_light_augmentation',
    'get_medium_augmentation',
    'get_strong_augmentation',
    
    # Data utilities
    'BYOLDatasetWrapper',
    'BYOLDataLoaderFactory',
    'prepare_byol_dataloaders',
    'wrap_existing_dataloader'
]

