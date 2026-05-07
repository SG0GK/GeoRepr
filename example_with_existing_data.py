"""
Complete example of using BYOL with existing 3D datasets.

This example shows how to:
1. Use your existing datasets/dataloaders
2. Prepare them for BYOL training using data_utils
3. Train with BYOLTrainer
4. Visualize results
"""

import torch
from torch.utils.data import Dataset, DataLoader

# Import BYOL components
from trainer import BYOLTrainer
from data_utils import prepare_byol_dataloaders, BYOLDataLoaderFactory
from augmentation import get_medium_augmentation, get_strong_augmentation
from plotting import create_all_plots


# ============================================================================
# STEP 1: Define Your Existing Dataset
# ============================================================================

class My3DDataset(Dataset):
    """
    Example of your existing 3D dataset.
    
    Replace this with your actual dataset implementation.
    Your dataset should return 3D volumes in one of these formats:
    - Just the tensor: torch.Tensor of shape [C, H, W, D]
    - Tuple: (volume, label) or (volume, metadata)
    - Dict: {'volume': tensor, 'label': label} or similar
    """
    
    def __init__(self, data_path, split='train'):
        """
        Args:
            data_path: Path to your data
            split: 'train' or 'val'
        """
        self.data_path = data_path
        self.split = split
        
        # TODO: Initialize your data loading here
        # For example: self.file_list = load_file_list(data_path, split)
        
        # For demonstration, we'll use dummy data
        self.num_samples = 100 if split == 'train' else 20
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Load and return a 3D volume.
        
        Returns:
            torch.Tensor: 3D volume [C, H, W, D]
            
        Note: The BYOLDatasetWrapper will handle this and create two augmented views.
        """
        # TODO: Replace this with your actual data loading
        # For example:
        # - volume = load_nifti(self.file_list[idx])
        # - volume = preprocess(volume)
        # - return volume
        
        # For demonstration, return a random volume
        volume = torch.randn(3, 64, 64, 32)  # [C, H, W, D]
        return volume


# ============================================================================
# STEP 2: Prepare BYOL Dataloaders (Method 1 - Direct Function)
# ============================================================================

def method1_direct_function():
    """Method 1: Use prepare_byol_dataloaders function directly."""
    
    print("\n" + "=" * 80)
    print("METHOD 1: Using prepare_byol_dataloaders()")
    print("=" * 80)
    
    # Create your existing datasets
    train_dataset = My3DDataset(data_path='/path/to/data', split='train')
    val_dataset = My3DDataset(data_path='/path/to/data', split='val')
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    
    # Define augmentation
    # Choose crop size based on your data
    transform = get_medium_augmentation(crop_size=(64, 64, 32))
    
    # Prepare BYOL dataloaders
    train_loader, val_loader = prepare_byol_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=8,
        transform=transform,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\nDataLoader info:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    
    # Test the dataloader
    batch = next(iter(train_loader))
    print(f"\nBatch format: {type(batch)}")
    print(f"  View1 shape: {batch['view1'].shape}")
    print(f"  View2 shape: {batch['view2'].shape}")
    
    return train_loader, val_loader


# ============================================================================
# STEP 3: Prepare BYOL Dataloaders (Method 2 - Factory Class)
# ============================================================================

def method2_factory_class():
    """Method 2: Use BYOLDataLoaderFactory for more convenience."""
    
    print("\n" + "=" * 80)
    print("METHOD 2: Using BYOLDataLoaderFactory")
    print("=" * 80)
    
    # Create your existing datasets
    train_dataset = My3DDataset(data_path='/path/to/data', split='train')
    val_dataset = My3DDataset(data_path='/path/to/data', split='val')
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    
    # Create factory with desired augmentation level
    factory = BYOLDataLoaderFactory(
        augmentation_level='medium',  # 'light', 'medium', or 'strong'
        crop_size=(64, 64, 32)
    )
    
    # Create BYOL dataloaders
    train_loader, val_loader = factory.create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=8,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"\nDataLoader info:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    
    # Test the dataloader
    batch = next(iter(train_loader))
    print(f"\nBatch format: {type(batch)}")
    print(f"  View1 shape: {batch['view1'].shape}")
    print(f"  View2 shape: {batch['view2'].shape}")
    
    return train_loader, val_loader


# ============================================================================
# STEP 4: Custom Augmentation
# ============================================================================

def method3_custom_augmentation():
    """Method 3: Define custom augmentation pipeline."""
    
    print("\n" + "=" * 80)
    print("METHOD 3: Using Custom Augmentation")
    print("=" * 80)
    
    from augmentation import (
        Compose, RandomCrop3D, RandomFlip3D, GaussianNoise3D,
        RandomIntensityScale3D, ClampValues
    )
    
    # Define custom augmentation pipeline
    custom_transform = Compose([
        RandomCrop3D(output_size=(64, 64, 32)),
        RandomFlip3D(p_h=0.5, p_w=0.5, p_d=0.5),
        GaussianNoise3D(mean=0.0, std=0.05),
        RandomIntensityScale3D(scale_range=(0.95, 1.05)),
        ClampValues(min_val=-1.0, max_val=1.0)
    ])
    
    # Create datasets
    train_dataset = My3DDataset(data_path='/path/to/data', split='train')
    val_dataset = My3DDataset(data_path='/path/to/data', split='val')
    
    # Use factory with custom transform
    factory = BYOLDataLoaderFactory(custom_transform=custom_transform)
    
    train_loader, val_loader = factory.create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=8,
        num_workers=4
    )
    
    print(f"\nUsing custom augmentation pipeline")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    
    return train_loader, val_loader


# ============================================================================
# STEP 5: Complete Training Pipeline
# ============================================================================

def complete_training_example():
    """Complete example with training and visualization."""
    
    print("\n" + "=" * 80)
    print("COMPLETE TRAINING EXAMPLE")
    print("=" * 80)
    
    # Step 1: Prepare data
    print("\nStep 1: Preparing data...")
    
    train_dataset = My3DDataset(data_path='/path/to/data', split='train')
    val_dataset = My3DDataset(data_path='/path/to/data', split='val')
    
    factory = BYOLDataLoaderFactory(
        augmentation_level='medium',
        crop_size=(64, 64, 32)
    )
    
    train_loader, val_loader = factory.create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=8,
        num_workers=4
    )
    
    print(f"  ✓ Train dataset: {len(train_dataset)} samples")
    print(f"  ✓ Val dataset:   {len(val_dataset)} samples")
    
    # Step 2: Configure model
    print("\nStep 2: Configuring model...")
    
    model_config = {
        'encoder_3d': {
            'in_channels': 3,
            'block_out_channels': [16, 32, 64],
            'block_out_types': ['down', 'down', 'same'],
            'num_groups': 16,
            'dropout': 0.01,
            'scales_w': [2, 2],
            'scales_h': [2, 2],
            'scales_d': [2, 2]
        }
    }
    
    print("  ✓ Model configuration ready")
    
    # Step 3: Initialize trainer
    print("\nStep 3: Initializing trainer...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")
    
    trainer = BYOLTrainer(
        model_config=model_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.2,
        weight_decay=1.5e-6,
        projection_dim=256,
        hidden_dim=4096,
        base_momentum=0.996,
        max_epochs=50,  # Reduced for demo
        warmup_epochs=5,
        checkpoint_dir='checkpoints_example',
        log_interval=5,
        save_interval=10
    )
    
    print("  ✓ Trainer initialized")
    
    # Step 4: Train
    print("\nStep 4: Training...")
    print("-" * 80)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n  Training interrupted by user")
    
    # Step 5: Visualize results
    print("\nStep 5: Generating plots...")
    
    try:
        create_all_plots(
            'checkpoints_example/training_history.json',
            output_dir='plots_example',
            show=False
        )
        print("  ✓ Plots saved to plots_example/")
    except FileNotFoundError:
        print("  No training history found")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nResults:")
    print(f"  - Checkpoints: checkpoints_example/")
    print(f"  - Plots: plots_example/")
    print(f"  - Best val loss: {trainer.best_val_loss:.4f}")


# ============================================================================
# STEP 6: Quick Start Template
# ============================================================================

def quick_start_template():
    """
    Quick start template - copy this for your own use!
    """
    
    # 1. Import required modules
    from byol import BYOLTrainer, BYOLDataLoaderFactory
    
    # 2. Create your datasets (replace with your actual datasets)
    # train_dataset = YourDataset(split='train')
    # val_dataset = YourDataset(split='val')
    train_dataset = My3DDataset(data_path='/path/to/data', split='train')
    val_dataset = My3DDataset(data_path='/path/to/data', split='val')
    
    # 3. Prepare BYOL dataloaders
    factory = BYOLDataLoaderFactory(
        augmentation_level='medium',  # Choose: 'light', 'medium', 'strong'
        crop_size=(64, 64, 32)        # Adjust to your data
    )
    
    train_loader, val_loader = factory.create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=16,               # Adjust based on GPU memory
        num_workers=4
    )
    
    # 4. Configure model
    model_config = {
        'encoder_3d': {
            'in_channels': 3,
            'block_out_channels': [16, 32, 64],
            'block_out_types': ['down', 'down', 'same'],
            'num_groups': 16,
            'dropout': 0.01,
            'scales_w': [2, 2],
            'scales_h': [2, 2],
            'scales_d': [2, 2]
        }
    }
    
    # 5. Create trainer and train
    trainer = BYOLTrainer(
        model_config=model_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cuda',
        max_epochs=500
    )
    
    trainer.train()
    
    # 6. Generate plots
    from byol.plotting import create_all_plots
    create_all_plots('checkpoints/training_history.json', output_dir='plots')


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='BYOL Training Examples')
    parser.add_argument(
        '--method',
        type=str,
        default='complete',
        choices=['method1', 'method2', 'method3', 'complete', 'quickstart'],
        help='Which example to run'
    )
    
    args = parser.parse_args()
    
    if args.method == 'method1':
        method1_direct_function()
    elif args.method == 'method2':
        method2_factory_class()
    elif args.method == 'method3':
        method3_custom_augmentation()
    elif args.method == 'complete':
        complete_training_example()
    elif args.method == 'quickstart':
        quick_start_template()
    
    print("\n" + "=" * 80)
    print("Example completed!")
    print("=" * 80)

