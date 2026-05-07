"""
Wrapper for DataCubes dataset to make it BYOL-compatible.

This wrapper extracts only X (the volume) from the dataset and 
applies BYOL augmentation to create two views.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from byol.augmentation import get_medium_augmentation


class BYOLDataCubesWrapper(Dataset):
    """
    Wrapper for DataCubes that extracts only X and creates BYOL views.
    
    The original dataset returns (X, X_C, y) but we only need X for BYOL training.
    This wrapper:
    1. Extracts X from the dataset
    2. Applies augmentation twice to create two views
    3. Returns in BYOL format
    """
    
    def __init__(self, base_dataset, transform=None, return_format='dict'):
        """
        Args:
            base_dataset: Original DataCubes dataset (returns X, X_C, y)
            transform: Augmentation function. If None, uses default medium augmentation.
            return_format: 'dict' or 'tuple' - format to return views
        """
        self.base_dataset = base_dataset
        self.return_format = return_format
        
        # Set up transform
        if transform is None:
            print("No transform provided. Using default medium augmentation without cropping.")
            self.transform = get_medium_augmentation(crop_size=None)
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """
        Get item from base dataset and create two augmented views.
        
        Args:
            idx: Index
        
        Returns:
            dict or tuple: Two augmented views of X
        """
        # Get data from base dataset
        X = self.base_dataset[idx]

        print(X.shape)
        
        # # Extract X based on what the dataset returns
        # if isinstance(item, torch.Tensor):
        #     # Just X
        #     X = item
        # elif isinstance(item, tuple):
        #     # (X, X_C, y) or (X, y) - take first element
        #     X = item[0]
        # else:
        #     raise TypeError(f"Unexpected dataset item type: {type(item)}")
        
        # Ensure X is float32 tensor
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        else:
            X = X.float()
        
        # Create two augmented views
        view1 = self.transform(X.clone())
        view2 = self.transform(X.clone())
        
        # Return in specified format
        if self.return_format == 'dict':
            return {'view1': view1, 'view2': view2}
        else:
            return view1, view2


def prepare_byol_dataloaders_from_datacubes(
    datasets,
    batch_size,
    transform=None,
    num_workers=4,
    return_format='dict'
):
    """
    Prepare BYOL dataloaders from split_dataset output.
    
    Args:
        datasets (dict): Output from split_dataset(config), contains 'train' and 'test' keys
        batch_size (int): Batch size for dataloaders
        transform: Augmentation transform. If None, uses default.
        num_workers (int): Number of data loading workers
        return_format (str): 'dict' or 'tuple'
    
    Returns:
        dict: Dictionary with 'train' and 'test' (or 'val') dataloaders
    
    Example:
        >>> from byol.data_parce.data import split_dataset
        >>> from byol.data_parce.byol_wrapper import prepare_byol_dataloaders_from_datacubes
        >>> 
        >>> # Load your datasets
        >>> datasets = split_dataset(config)
        >>> 
        >>> # Prepare BYOL dataloaders
        >>> dataloaders = prepare_byol_dataloaders_from_datacubes(
        ...     datasets=datasets,
        ...     batch_size=8,
        ...     transform=my_transform
        ... )
        >>> 
        >>> train_loader = dataloaders['train']
        >>> test_loader = dataloaders['test']
    """
    dataloaders = {}
    
    for split_name, dataset in datasets.items():
        # Wrap dataset for BYOL
        wrapped_dataset = BYOLDataCubesWrapper(
            base_dataset=dataset,
            transform=transform,
            return_format=return_format
        )
        
        # Create dataloader
        shuffle = True if 'train' in split_name else False
        
        dataloader = DataLoader(
            wrapped_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True if shuffle else False  # Drop last for training
        )
        
        dataloaders[split_name] = dataloader
    
    return dataloaders


if __name__ == "__main__":
    # Test the wrapper
    print("Testing BYOLDataCubesWrapper...")
    print("=" * 80)
    
    # Create a mock dataset that mimics DataCubesWithStatsAndVerticalWells
    class MockDataCubesDataset(Dataset):
        def __init__(self, num_samples=10):
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Mimic DataCubesWithStatsAndVerticalWells output: (X, X_C, y)
            X = torch.randn(3, 150, 144, 80)  # Your actual data shape
            X_C = torch.randn(4, 150, 144, 80)
            y = torch.randn(10)
            return X, X_C, y
    
    # Create mock dataset
    mock_dataset = MockDataCubesDataset(20)
    
    # Wrap it for BYOL
    from byol.augmentation import get_light_augmentation
    
    transform = get_light_augmentation(crop_size=(128, 128, 64))
    
    wrapped_dataset = BYOLDataCubesWrapper(
        mock_dataset,
        transform=transform
    )
    
    # Test single item
    item = wrapped_dataset[0]
    print(f"\nSingle item test:")
    print(f"  Type: {type(item)}")
    print(f"  View1 shape: {item['view1'].shape}")
    print(f"  View2 shape: {item['view2'].shape}")
    print(f"  View1 dtype: {item['view1'].dtype}")
    
    # Test with dataloader
    dataloader = DataLoader(wrapped_dataset, batch_size=4, shuffle=False)
    batch = next(iter(dataloader))
    
    print(f"\nBatch test:")
    print(f"  Batch type: {type(batch)}")
    print(f"  View1 batch shape: {batch['view1'].shape}")
    print(f"  View2 batch shape: {batch['view2'].shape}")
    
    # Verify views are different
    diff = (batch['view1'] - batch['view2']).abs().mean()
    print(f"  Mean difference between views: {diff:.4f}")
    assert diff > 0, "Views should be different!"
    
    print("\n" + "=" * 80)
    print("✓ Wrapper test passed!")

