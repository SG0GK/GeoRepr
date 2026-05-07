"""
Example training script for BYOL on 3D images.

This script demonstrates how to:
1. Set up data loaders with augmentation
2. Configure and initialize the BYOL trainer
3. Train the model
4. Generate visualization plots
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

from trainer import BYOLTrainer
from plotting import create_all_plots


class Dummy3DDataset(Dataset):
    """
    Dummy dataset for demonstration purposes.
    Replace this with your actual 3D dataset.
    """
    
    def __init__(self, num_samples=100, shape=(3, 64, 64, 32)):
        """
        Args:
            num_samples (int): Number of samples in the dataset
            shape (tuple): Shape of 3D volumes (C, H, W, D)
        """
        self.num_samples = num_samples
        self.shape = shape
    
    def __len__(self):
        return self.num_samples
    
    def _augment(self, volume):
        """
        Apply augmentation to create a view.
        Replace with your actual augmentation pipeline.
        """
        # Simple augmentation: add random noise and flip
        augmented = volume.clone()
        
        # Random Gaussian noise
        noise = torch.randn_like(augmented) * 0.1
        augmented = augmented + noise
        
        # Random horizontal flip
        if np.random.rand() > 0.5:
            augmented = torch.flip(augmented, [2])  # Flip W dimension
        
        # Random vertical flip
        if np.random.rand() > 0.5:
            augmented = torch.flip(augmented, [1])  # Flip H dimension
        
        # Clip values
        augmented = torch.clamp(augmented, -1, 1)
        
        return augmented
    
    def __getitem__(self, idx):
        """
        Returns two augmented views of the same volume.
        """
        # Generate random volume (replace with actual data loading)
        volume = torch.randn(*self.shape)
        
        # Create two different augmented views
        view1 = self._augment(volume)
        view2 = self._augment(volume)
        
        # Return as dictionary
        return {'view1': view1, 'view2': view2}


def main():
    """Main training function."""
    
    print("="*80)
    print("BYOL Training Example for 3D Images")
    print("="*80)
    
    # ========== Configuration ==========
    
    # Model configuration
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
    
    # Training hyperparameters
    batch_size = 8  # Adjust based on your GPU memory
    num_workers = 4
    max_epochs = 50  # Reduced for demo (paper uses 1000)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # ========== Create Datasets ==========
    
    print("\nCreating datasets...")
    train_dataset = Dummy3DDataset(num_samples=100)
    val_dataset = Dummy3DDataset(num_samples=20)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    
    # ========== Create Data Loaders ==========
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # ========== Initialize Trainer ==========
    
    print("\nInitializing trainer...")
    trainer = BYOLTrainer(
        model_config=model_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=0.2,           # Base learning rate
        weight_decay=1.5e-6,
        momentum=0.9,
        projection_dim=256,
        hidden_dim=4096,
        base_momentum=0.996,
        max_epochs=max_epochs,
        warmup_epochs=5,             # 5 epochs warmup
        checkpoint_dir='checkpoints_example',
        log_interval=5,              # Log every 5 steps
        save_interval=10             # Save checkpoint every 10 epochs
    )
    
    # ========== Train ==========
    
    print("\nStarting training...")
    print("="*80)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    # ========== Generate Plots ==========
    
    print("\nGenerating plots...")
    history_path = Path('checkpoints_example') / 'training_history.json'
    
    if history_path.exists():
        create_all_plots(
            history_path,
            output_dir='plots_example',
            show=False
        )
        print("\n✓ Plots saved to 'plots_example/'")
    else:
        print("No training history found. Skipping plot generation.")
    
    print("\n" + "="*80)
    print("Training completed!")
    print("="*80)
    
    # ========== Summary ==========
    
    print("\nSummary:")
    print(f"  - Checkpoints: checkpoints_example/")
    print(f"  - Plots: plots_example/")
    print(f"  - Best model: checkpoints_example/best_model.pt")
    print(f"  - Best validation loss: {trainer.best_val_loss:.4f}")
    
    print("\nTo resume training:")
    print("  trainer.train(resume_from='checkpoints_example/latest_checkpoint.pt')")
    
    print("\nTo load the trained model:")
    print("""
    from byol import BYOL
    import torch
    
    model = BYOL(model_config)
    checkpoint = torch.load('checkpoints_example/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    """)


if __name__ == "__main__":
    main()

