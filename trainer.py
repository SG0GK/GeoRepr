"""
BYOL Trainer class for training the model on 3D volumetric data.

This trainer handles:
- Training loop with EMA updates for target network
- Learning rate scheduling
- Checkpointing
- Logging metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
import numpy as np
from pathlib import Path

from .model import BYOL, compute_byol_loss


class BYOLTrainer:
    """
    Trainer class for BYOL model.
    
    Handles the complete training pipeline including:
    - Training loop with self-supervised learning
    - Target network EMA updates
    - Learning rate scheduling
    - Checkpointing
    - Metrics logging
    """
    
    def __init__(
        self,
        model_config,
        train_loader,
        device='cuda',
        learning_rate=0.2,
        weight_decay=1.5e-6,
        momentum=0.9,
        projection_dim=256,
        hidden_dim=4096,
        base_momentum=0.996,
        max_epochs=1000,
        warmup_epochs=10,
        checkpoint_dir='checkpoints_freeze',
        log_interval=10,
        save_interval=10,
        run=None,
        save_representations_interval=None,
        repr_loader=None,
        freeze_2d_encoder=True,
        is_train=True,
        fine_tune=True
    ):
        """
        Args:
            model_config (dict): Configuration for the encoder model
            train_loader (DataLoader): Training data loader
            device (str): Device to train on ('cuda' or 'cpu')
            learning_rate (float): Base learning rate (scaled by batch_size/256)
            weight_decay (float): Weight decay for optimizer
            momentum (float): Momentum for SGD optimizer
            projection_dim (int): Dimension of projection space
            hidden_dim (int): Hidden dimension for MLP layers
            base_momentum (float): Base EMA decay rate for target network
            max_epochs (int): Maximum number of training epochs
            warmup_epochs (int): Number of warmup epochs for learning rate
            checkpoint_dir (str): Directory to save checkpoints
            log_interval (int): Log metrics every N steps
            save_interval (int): Save checkpoint every N epochs
            run: Optional experiment run object for logging metrics
            save_representations_interval (int, optional): Save representations every N epochs (None to disable)
            repr_loader (DataLoader, optional): Separate dataloader for extracting representations (without augmentation)
            freeze_2d_encoder (bool, optional): Wherether to freeze encoder
            is_train (bool, optional): Train or infer mode
            fine_tune (bool, optional): Whether to fine-tune the 2D encoder
        """
        self.run = run
        self.device = torch.device(device)
        self.train_loader = train_loader
        self.repr_loader = repr_loader if repr_loader is not None else train_loader
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.base_momentum = base_momentum
        self.save_representations_interval = save_representations_interval
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        print("Initializing BYOL model...")
        self.model = BYOL(
            config=model_config,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            moving_average_decay=base_momentum,
            freeze_2d_encoder=freeze_2d_encoder,
            is_train=is_train,
            fine_tune=fine_tune
        ).to(self.device)
        
        # Scale learning rate by batch size (as in the paper)
        batch_size = train_loader.batch_size
        self.base_lr = learning_rate * batch_size / 256
        
        # Initialize optimizer (LARS optimizer is used in paper, but we use SGD here)
        # For production, consider using LARS optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.base_lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler (cosine annealing after warmup)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max_epochs - warmup_epochs,
            eta_min=0
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_train_loss = float('inf')
        self.last_repr_epoch = None  # Track last epoch where representations were saved
        
        # Metrics history
        self.history = {
            'train_loss': [],
            'learning_rate': [],
            'target_momentum': []
        }
        
        if self.run is not None:
            # Store model summary for experiment tracking
            self.run["model/summary"] = str(self.model)
        
        print(f"Model initialized. Training on {self.device}")
        print(f"Base learning rate: {self.base_lr:.6f}")
        print(f"Batch size: {batch_size}")
        print(f"Total epochs: {max_epochs}")
    
    def _update_learning_rate(self, epoch):
        """
        Update learning rate with warmup and cosine annealing.
        
        During warmup: linear increase from 0 to base_lr
        After warmup: cosine annealing
        
        Args:
            epoch (int): Current epoch
        """
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # Cosine annealing (handled by scheduler)
            self.scheduler.step()
    
    def _compute_momentum(self, epoch):
        """
        Compute target network momentum (EMA decay rate).
        
        As in the paper, momentum increases from base_momentum to 1.0
        using a cosine schedule.
        
        Args:
            epoch (int): Current epoch
        
        Returns:
            float: Momentum value for current epoch
        """
        # Momentum schedule: increases from base_momentum to 1.0
        momentum = 1 - (1 - self.base_momentum) * (
            np.cos(np.pi * epoch / self.max_epochs) + 1
        ) / 2
        return momentum
    
    def train_epoch(self, epoch):
        """
        Train for one epoch.
        
        Args:
            epoch (int): Current epoch number
        
        Returns:
            float: Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Expect batch to contain two augmented views
            # batch should be: (view1, view2) or {'view1': ..., 'view2': ...}
            # if isinstance(batch, dict):
            #     view1 = batch['view1'].to(self.device)
            #     view2 = batch['view2'].to(self.device)
            # elif isinstance(batch, (list, tuple)) and len(batch) == 2:
            #     view1, view2 = batch
            #     view1 = view1.to(self.device)
            #     view2 = view2.to(self.device)
            # else:
            #     raise ValueError(
            #         "Batch must be a dict with 'view1' and 'view2' keys, "
            #         "or a tuple/list of (view1, view2)"
            #     )

            batch = batch.to(self.device)
            
            # Forward pass
            outputs = self.model(batch)
            
            # Compute loss
            loss = compute_byol_loss(outputs)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update target network with EMA
            momentum = self._compute_momentum(epoch)
            self.model.update_target_network(momentum)
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            if self.run is not None and batch_idx % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                self.run["train/batch/loss"].append(loss.item())
                self.run["train/batch/learning_rate"].append(current_lr)
                self.run["train/batch/momentum"].append(momentum)
                self.run["train/batch/global_step"].append(self.global_step)
            
            # Print progress
            if batch_idx % self.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Batch {batch_idx}/{num_batches} - "
                      f"Loss: {loss.item():.4f}, "
                      f"LR: {current_lr:.6f}, "
                      f"Momentum: {momentum:.6f}")
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def save_representations(self, epoch):
        """
        Extract and save representations from non-augmented data.
        Deletes previous epoch-specific files to save memory.
        
        Args:
            epoch (int): Current epoch number
        """
        self.model.eval()
        
        print(f"\n  Extracting representations from non-augmented data...")
        representations_dir = self.checkpoint_dir / 'representations'
        representations_dir.mkdir(exist_ok=True)
        
        # Delete previous epoch-specific files if they exist
        if self.last_repr_epoch is not None:
            old_repr_path = representations_dir / f'representations_best_epoch_{self.last_repr_epoch}.npy'
            old_proj_path = representations_dir / f'projections_best_epoch_{self.last_repr_epoch}.npy'
            
            if old_repr_path.exists():
                old_repr_path.unlink()
                print(f"  Deleted old representation file: epoch {self.last_repr_epoch}")
            if old_proj_path.exists():
                old_proj_path.unlink()
        
        all_representations = []
        all_projections = []
        
        # Use repr_loader (non-augmented data)
        for batch_idx, batch in enumerate(self.repr_loader):
            # Get data (single view, no augmentation)
            if isinstance(batch, (tuple, list)):
                # Expecting (data, labels) or just (data,)
                data = batch[0].to(self.device)
            else:
                data = batch.to(self.device)
            
            # Extract representations using online network
            representations = self.model.get_representations(
                data,
                use_target=False,  # Use online network
                from_projector=False,  # Get encoder output
                normalize=False  # Don't normalize for raw features
            )
            all_representations.append(representations.cpu().numpy())
            
            # Also save projections
            projections = self.model.get_representations(
                data,
                use_target=False,
                from_projector=True,  # Get projector output
                normalize=False
            )
            all_projections.append(projections.cpu().numpy())
        
        # Concatenate all batches
        all_representations = np.concatenate(all_representations, axis=0)
        all_projections = np.concatenate(all_projections, axis=0)
        
        # Save to .npy files (overwrite with best epoch)
        repr_path = representations_dir / f'representations_best_epoch_{epoch}.npy'
        proj_path = representations_dir / f'projections_best_epoch_{epoch}.npy'
        
        # Also save as 'best' without epoch number for easy access
        repr_path_best = representations_dir / 'representations_best.npy'
        proj_path_best = representations_dir / 'projections_best.npy'
        
        np.save(repr_path, all_representations)
        np.save(proj_path, all_projections)
        np.save(repr_path_best, all_representations)
        np.save(proj_path_best, all_projections)
        
        # Update last saved epoch
        self.last_repr_epoch = epoch
        
        print(f"\n  Saved representations to {representations_dir}/")
        print(f"    - Representations shape: {all_representations.shape}")
        print(f"    - Projections shape: {all_projections.shape}")
        print(f"    - Files: representations_best.npy, projections_best.npy")
        print(f"    - Epoch-specific: representations_best_epoch_{epoch}.npy")
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Current epoch
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_train_loss': self.best_train_loss,
            'last_repr_epoch': self.last_repr_epoch,
            'history': self.history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path (str or Path): Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_train_loss = checkpoint.get('best_train_loss', checkpoint.get('best_val_loss', float('inf')))
        self.last_repr_epoch = checkpoint.get('last_repr_epoch', None)
        self.history = checkpoint['history']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        if self.last_repr_epoch is not None:
            print(f"Last representation saved at epoch: {self.last_repr_epoch}")
    
    def save_history(self):
        """Save training history to JSON file."""
        history_path = self.checkpoint_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def train(self, resume_from=None):
        """
        Main training loop.
        
        Args:
            resume_from (str or Path, optional): Path to checkpoint to resume from
        """
        # Resume from checkpoint if provided
        if resume_from is not None:
            self.load_checkpoint(resume_from)
            start_epoch = self.current_epoch + 1
        else:
            start_epoch = 0
        
        print(f"\nStarting training from epoch {start_epoch}...")
        print("=" * 80)
        
        try:
            for epoch in range(start_epoch, self.max_epochs):
                self.current_epoch = epoch
                
                # Update learning rate
                self._update_learning_rate(epoch)
                current_lr = self.optimizer.param_groups[0]['lr']
                current_momentum = self._compute_momentum(epoch)
                
                print(f"\nEpoch {epoch+1}/{self.max_epochs}")
                print(f"Learning rate: {current_lr:.6f}")
                print(f"Target momentum: {current_momentum:.6f}")
                print("-" * 80)
                
                # Train for one epoch
                train_loss = self.train_epoch(epoch)
                
                # Update history
                self.history['train_loss'].append(train_loss)
                self.history['learning_rate'].append(current_lr)
                self.history['target_momentum'].append(current_momentum)
                
                # Print epoch summary
                print(f"\nEpoch {epoch+1} Summary:")
                print(f"  Train Loss: {train_loss:.4f}")
                
                # Epoch-level logging
                if self.run is not None:
                    self.run["train/epoch/loss"].append(train_loss)
                    self.run["train/epoch/learning_rate"].append(current_lr)
                    self.run["train/epoch/momentum"].append(current_momentum)
                
                # Check if best model (based on training loss)
                is_best = train_loss < self.best_train_loss
                if is_best:
                    self.best_train_loss = train_loss
                    print(f"  ✓ New best training loss!")
                    if self.run is not None:
                        self.run["train/best_loss"].append(train_loss)
                        self.run["train/best_epoch"] = epoch
                
                # Save checkpoint
                if (epoch + 1) % self.save_interval == 0 or is_best:
                    self.save_checkpoint(epoch, is_best=is_best)
                    if self.run is not None:
                        self.run["checkpoints/last_epoch"] = epoch
                
                # Save history
                self.save_history()
                
                print("=" * 80)
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user.")
            print("Saving checkpoint...")
            self.save_checkpoint(self.current_epoch, is_best=False)
            self.save_history()
        
        print("\nTraining completed!")
        print(f"Best training loss: {self.best_train_loss:.4f}")
        print(f"Checkpoints saved in: {self.checkpoint_dir}")
        if self.save_representations_interval is not None:
            print(f"Best representations saved in: {self.checkpoint_dir}/representations/")


if __name__ == "__main__":
    # Example usage
    print("BYOL Trainer Example")
    print("=" * 80)
    print("\nThis is an example of how to use the BYOLTrainer class.")
    print("You need to provide your own data loaders.\n")
    
    # Example model config
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
    
    print("Example configuration:")
    print(json.dumps(model_config, indent=2))
    print("\n" + "=" * 80)
    print("\nTo use the trainer, create your data loaders and initialize:")
    print("""
    trainer = BYOLTrainer(
        model_config=model_config,
        train_loader=train_loader,
        device='cuda',
        max_epochs=1000
    )
    
    # Start training
    trainer.train()
    
    # Or resume from checkpoint
    trainer.train(resume_from='checkpoints/latest_checkpoint.pt')
    """)

