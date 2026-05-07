"""
Data utilities for BYOL training.

Provides dataset wrappers and utilities to convert standard 3D datasets
to BYOL format (returning two augmented views).
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union, Tuple, Callable
import warnings

from .augmentation import get_medium_augmentation


class BYOLDatasetWrapper(Dataset):
    """
    Wrapper for existing datasets to create BYOL-compatible format.
    
    Takes a dataset that returns single volumes and creates two augmented views.
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        transform: Optional[Callable] = None,
        return_format: str = 'dict'
    ):
        """
        Args:
            base_dataset: Original dataset that returns 3D volumes
            transform: Augmentation function to apply. If None, uses default medium augmentation.
            return_format: Format to return data ('dict' or 'tuple')
                          'dict': returns {'view1': ..., 'view2': ...}
                          'tuple': returns (view1, view2)
        """
        self.base_dataset = base_dataset
        self.transform = transform
        self.return_format = return_format
        
        if self.transform is None:
            print("No transform provided. Using default medium augmentation.")
            # We'll set crop_size to None by default, user should provide it
            self.transform = get_medium_augmentation(crop_size=None)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """
        Get item from base dataset and create two augmented views.
        
        Args:
            idx: Index
        
        Returns:
            dict or tuple: Two augmented views in specified format
        """
        # Get data from base dataset
        item = self.base_dataset[idx]
        
        # Extract volume from item (handle different dataset formats)
        if isinstance(item, torch.Tensor):
            volume = item
        elif isinstance(item, tuple):
            # Assume first element is the volume
            volume = item[0]
        elif isinstance(item, dict):
            # Try common keys for volumes
            if 'volume' in item:
                volume = item['volume']
            elif 'image' in item:
                volume = item['image']
            elif 'data' in item:
                volume = item['data']
            else:
                raise ValueError(
                    f"Could not find volume in dict with keys: {item.keys()}. "
                    "Expected 'volume', 'image', or 'data' key."
                )
        else:
            raise TypeError(
                f"Unsupported dataset item type: {type(item)}. "
                "Expected Tensor, tuple, or dict."
            )
        
        # Ensure volume is a tensor
        if not isinstance(volume, torch.Tensor):
            volume = torch.tensor(volume)
        
        # Create two augmented views
        view1 = self.transform(volume.clone())
        view2 = self.transform(volume.clone())
        
        # Return in specified format
        if self.return_format == 'dict':
            return {'view1': view1, 'view2': view2}
        elif self.return_format == 'tuple':
            return view1, view2
        else:
            raise ValueError(f"Unknown return_format: {self.return_format}")


def prepare_byol_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int,
    transform: Optional[Callable] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    return_format: str = 'dict'
) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare train and validation dataloaders for BYOL training.
    
    This function wraps your existing datasets with BYOLDatasetWrapper
    to create two augmented views of each volume.
    
    Args:
        train_dataset: Training dataset (returns single 3D volumes)
        val_dataset: Validation dataset (returns single 3D volumes)
        batch_size: Batch size for dataloaders
        transform: Augmentation transform to apply. If None, uses default.
        num_workers: Number of workers for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        return_format: Format to return data ('dict' or 'tuple')
    
    Returns:
        tuple: (train_loader, val_loader) ready for BYOL training
    
    Example:
        >>> from byol.data_utils import prepare_byol_dataloaders
        >>> from byol.augmentation import get_strong_augmentation
        >>> 
        >>> # Create augmentation
        >>> transform = get_strong_augmentation(crop_size=(64, 64, 32))
        >>> 
        >>> # Prepare dataloaders
        >>> train_loader, val_loader = prepare_byol_dataloaders(
        ...     train_dataset=my_train_dataset,
        ...     val_dataset=my_val_dataset,
        ...     batch_size=16,
        ...     transform=transform
        ... )
        >>> 
        >>> # Use with BYOLTrainer
        >>> trainer = BYOLTrainer(
        ...     model_config=config,
        ...     train_loader=train_loader,
        ...     val_loader=val_loader
        ... )
    """
    # Wrap datasets
    train_dataset_wrapped = BYOLDatasetWrapper(
        train_dataset,
        transform=transform,
        return_format=return_format
    )
    
    val_dataset_wrapped = BYOLDatasetWrapper(
        val_dataset,
        transform=transform,
        return_format=return_format
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset_wrapped,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop last incomplete batch for stable training
    )
    
    val_loader = DataLoader(
        val_dataset_wrapped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader


def wrap_existing_dataloader(
    dataloader: DataLoader,
    transform: Optional[Callable] = None,
    return_format: str = 'dict'
) -> DataLoader:
    """
    Wrap an existing dataloader to create BYOL-compatible format.
    
    Note: This creates a new dataloader with the same parameters as the original,
    but with the dataset wrapped.
    
    Args:
        dataloader: Existing dataloader
        transform: Augmentation transform. If None, uses default.
        return_format: Format to return data ('dict' or 'tuple')
    
    Returns:
        DataLoader with wrapped dataset
    
    Example:
        >>> from byol.data_utils import wrap_existing_dataloader
        >>> 
        >>> # Wrap existing dataloader
        >>> train_loader_byol = wrap_existing_dataloader(
        ...     train_loader,
        ...     transform=my_transform
        ... )
    """
    # Wrap the dataset
    wrapped_dataset = BYOLDatasetWrapper(
        dataloader.dataset,
        transform=transform,
        return_format=return_format
    )
    
    # Create new dataloader with same parameters
    new_dataloader = DataLoader(
        wrapped_dataset,
        batch_size=dataloader.batch_size,
        shuffle=isinstance(dataloader.sampler, torch.utils.data.RandomSampler),
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last
    )
    
    return new_dataloader


class BYOLDataLoaderFactory:
    """
    Factory class for creating BYOL-compatible dataloaders.
    
    This class provides a convenient interface for preparing dataloaders
    with customizable augmentation strategies.
    """
    
    def __init__(
        self,
        augmentation_level: str = 'medium',
        crop_size: Optional[Union[int, Tuple[int, int, int]]] = None,
        custom_transform: Optional[Callable] = None
    ):
        """
        Args:
            augmentation_level: Level of augmentation ('light', 'medium', 'strong')
            crop_size: Size for random cropping. If None, no cropping is applied.
            custom_transform: Custom augmentation transform. If provided, overrides augmentation_level.
        """
        self.augmentation_level = augmentation_level
        self.crop_size = crop_size
        self.custom_transform = custom_transform
        
        # Create transform
        if custom_transform is not None:
            self.transform = custom_transform
        else:
            from .augmentation import (
                get_light_augmentation,
                get_medium_augmentation,
                get_strong_augmentation
            )
            
            if augmentation_level == 'light':
                self.transform = get_light_augmentation(crop_size)
            elif augmentation_level == 'medium':
                self.transform = get_medium_augmentation(crop_size)
            elif augmentation_level == 'strong':
                self.transform = get_strong_augmentation(crop_size)
            else:
                raise ValueError(
                    f"Unknown augmentation_level: {augmentation_level}. "
                    "Choose from 'light', 'medium', 'strong'."
                )
    
    def create_dataloaders(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int,
        num_workers: int = 4,
        pin_memory: bool = True,
        return_format: str = 'dict'
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create BYOL-compatible dataloaders.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory
            return_format: Data return format ('dict' or 'tuple')
        
        Returns:
            tuple: (train_loader, val_loader)
        """
        return prepare_byol_dataloaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=batch_size,
            transform=self.transform,
            num_workers=num_workers,
            pin_memory=pin_memory,
            return_format=return_format
        )
    
    def wrap_dataloader(
        self,
        dataloader: DataLoader,
        return_format: str = 'dict'
    ) -> DataLoader:
        """
        Wrap an existing dataloader.
        
        Args:
            dataloader: Existing dataloader
            return_format: Data return format ('dict' or 'tuple')
        
        Returns:
            Wrapped dataloader
        """
        return wrap_existing_dataloader(
            dataloader=dataloader,
            transform=self.transform,
            return_format=return_format
        )


if __name__ == "__main__":
    # Test the data utilities
    print("Testing BYOL data utilities...")
    print("=" * 80)
    
    # Create a dummy dataset
    class DummyDataset(Dataset):
        def __init__(self, num_samples=10):
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Return a 3D volume
            return torch.randn(3, 64, 64, 32)
    
    # Test 1: Using prepare_byol_dataloaders
    print("\nTest 1: Using prepare_byol_dataloaders")
    print("-" * 80)
    
    from augmentation import get_medium_augmentation
    
    train_dataset = DummyDataset(100)
    val_dataset = DummyDataset(20)
    
    transform = get_medium_augmentation(crop_size=(48, 48, 24))
    
    train_loader, val_loader = prepare_byol_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=4,
        transform=transform,
        num_workers=0
    )
    
    print(f"Train loader batches: {len(train_loader)}")
    print(f"Val loader batches: {len(val_loader)}")
    
    # Test a batch
    batch = next(iter(train_loader))
    print(f"Batch type: {type(batch)}")
    print(f"View1 shape: {batch['view1'].shape}")
    print(f"View2 shape: {batch['view2'].shape}")
    
    # Test 2: Using BYOLDataLoaderFactory
    print("\nTest 2: Using BYOLDataLoaderFactory")
    print("-" * 80)
    
    factory = BYOLDataLoaderFactory(
        augmentation_level='strong',
        crop_size=(48, 48, 24)
    )
    
    train_loader2, val_loader2 = factory.create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=8,
        num_workers=0
    )
    
    print(f"Train loader batches: {len(train_loader2)}")
    print(f"Val loader batches: {len(val_loader2)}")
    
    batch2 = next(iter(train_loader2))
    print(f"Batch type: {type(batch2)}")
    print(f"View1 shape: {batch2['view1'].shape}")
    print(f"View2 shape: {batch2['view2'].shape}")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")

