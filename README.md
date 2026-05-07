# BYOL for 3D Images

Implementation of **Bootstrap Your Own Latent (BYOL)** for 3D volumetric data, based on the paper ["Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning"](https://arxiv.org/pdf/2006.07733).

## Overview

This implementation adapts BYOL for 3D medical/volumetric images by:
1. Using `Enhanced3DTo2DEncoder` to convert 3D volumes to 2D representations
2. Using ResNet50 as the backbone encoder
3. Training the 3D encoder jointly with BYOL

## Architecture

```
3D Input [B, 3, H, W, D]
    ↓
Enhanced3DTo2DEncoder
    ↓
2D Representation [B, 3, H', W']
    ↓
ResNet50 Encoder
    ↓
Features [B, 2048]
    ↓
┌─────────────────────┬─────────────────────┐
│  Online Network     │  Target Network     │
│  ├─ Projector       │  ├─ Projector       │
│  └─ Predictor       │  (no predictor)     │
└─────────────────────┴─────────────────────┘
```

## Key Features

- **No negative pairs**: Unlike contrastive methods, BYOL learns from positive pairs only
- **EMA target network**: Target network is updated via exponential moving average
- **Momentum scheduling**: Momentum increases from base value to 1.0 using cosine schedule
- **Flexible configuration**: Easy to customize encoder architecture and training parameters

## Files

- `model.py`: BYOL architecture (online/target networks, encoders, loss functions)
- `trainer.py`: Training pipeline with checkpointing and logging
- `plotting.py`: Visualization utilities for training metrics

## Installation

Required dependencies:
```bash
pip install torch torchvision matplotlib seaborn numpy
```

## Usage

### 1. Prepare Your Data

Your data loader should return two augmented views of the same 3D volume:

```python
# Option 1: Return as tuple
def __getitem__(self, idx):
    volume = load_volume(idx)  # [3, H, W, D]
    view1 = augment(volume)
    view2 = augment(volume)
    return view1, view2

# Option 2: Return as dictionary
def __getitem__(self, idx):
    volume = load_volume(idx)
    view1 = augment(volume)
    view2 = augment(volume)
    return {'view1': view1, 'view2': view2}
```

### 2. Create Configuration

```python
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
```

### 3. Initialize Trainer

```python
from byol import BYOLTrainer

trainer = BYOLTrainer(
    model_config=model_config,
    train_loader=train_loader,
    val_loader=val_loader,
    device='cuda',
    learning_rate=0.2,           # Base LR (scaled by batch_size/256)
    weight_decay=1.5e-6,
    projection_dim=256,
    hidden_dim=4096,
    base_momentum=0.996,         # Base EMA decay rate
    max_epochs=1000,
    warmup_epochs=10,
    checkpoint_dir='checkpoints',
    log_interval=10,
    save_interval=10
)
```

### 4. Train the Model

```python
# Start training from scratch
trainer.train()

# Or resume from checkpoint
trainer.train(resume_from='checkpoints/latest_checkpoint.pt')
```

### 5. Visualize Results

```python
from byol.plotting import create_all_plots

# Generate all plots
create_all_plots(
    'checkpoints/training_history.json',
    output_dir='plots',
    show=False
)
```

## Advanced Usage

### Using the Model Directly

```python
from byol import BYOL, compute_byol_loss
import torch

# Create model
model = BYOL(
    config=model_config,
    projection_dim=256,
    hidden_dim=4096,
    moving_average_decay=0.996
)

# Forward pass
view1 = torch.randn(4, 3, 64, 64, 32)  # [B, 3, H, W, D]
view2 = torch.randn(4, 3, 64, 64, 32)
outputs = model(view1, view2)

# Compute loss
loss = compute_byol_loss(outputs)

# Update target network
model.update_target_network()
```

### Custom Plotting

```python
from byol.plotting import (
    load_history,
    plot_loss_curves,
    plot_learning_rate,
    plot_target_momentum,
    compare_runs
)

# Load history
history = load_history('checkpoints/training_history.json')

# Plot individual metrics
plot_loss_curves(history, save_path='loss.png')
plot_learning_rate(history, save_path='lr.png')
plot_target_momentum(history, save_path='momentum.png')

# Compare multiple training runs
compare_runs(
    history_paths=['run1/training_history.json', 'run2/training_history.json'],
    labels=['Baseline', 'Modified'],
    save_path='comparison.png'
)
```

### Extract Learned Representations

```python
# Load trained model
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Extract features using the online encoder
model.eval()
with torch.no_grad():
    features = model.online_network.encoder(volume_3d)
    # features: [B, 2048]
```

## Hyperparameters

### Key Hyperparameters from the Paper

- **Learning rate**: 0.2 × batch_size / 256
- **Weight decay**: 1.5e-6
- **Projection dimension**: 256
- **Hidden dimension**: 4096
- **Base momentum (τ)**: 0.996 (increases to 1.0)
- **Batch size**: 4096 (paper), adjust based on your GPU memory
- **Optimizer**: LARS (paper) or SGD (this implementation)

### Learning Rate Schedule

1. **Warmup**: Linear increase from 0 to base_lr over `warmup_epochs`
2. **Cosine annealing**: Decreases from base_lr to 0 over remaining epochs

### Momentum Schedule

Target network momentum increases from `base_momentum` to 1.0 using:

```
τ(t) = 1 - (1 - τ_base) × (cos(πt/T) + 1) / 2
```

where `t` is current epoch and `T` is total epochs.

## Augmentation Strategies

BYOL is relatively robust to augmentations compared to contrastive methods. Recommended augmentations for 3D volumes:

1. **Random cropping** (different sizes and positions)
2. **Random flipping** (along spatial axes)
3. **Gaussian noise**
4. **Intensity scaling**
5. **Elastic deformation** (optional)
6. **Random rotation** (small angles)

Example:
```python
def augment_3d(volume):
    volume = random_crop(volume, size=(64, 64, 32))
    volume = random_flip_3d(volume)
    volume = add_gaussian_noise(volume, std=0.1)
    volume = scale_intensity(volume, scale_range=(0.9, 1.1))
    return volume
```

## Training Tips

1. **Batch size**: Use the largest batch size your GPU can handle. Scale learning rate accordingly.
2. **Warmup**: 10 epochs of warmup helps stabilize training
3. **Momentum**: Default 0.996 works well, but you can experiment with 0.99-0.999
4. **Epochs**: Paper uses 1000 epochs for ImageNet. For smaller datasets, 300-500 may suffice
5. **Augmentations**: Start with basic augmentations and add more if needed
6. **Validation**: Monitor validation loss to check for overfitting

## Troubleshooting

### Loss stays constant / doesn't decrease
- Check data augmentation (views should be different)
- Ensure target network is being updated (check momentum values)
- Try increasing learning rate or batch size

### Training instability
- Reduce learning rate
- Increase warmup epochs
- Check for NaN values in loss

### Out of memory
- Reduce batch size
- Use gradient accumulation
- Reduce projection_dim or hidden_dim

## Citation

If you use this implementation, please cite the original BYOL paper:

```bibtex
@inproceedings{grill2020bootstrap,
  title={Bootstrap your own latent: A new approach to self-supervised learning},
  author={Grill, Jean-Bastien and Strub, Florian and Altch{\'e}, Florent and Tallec, Corentin and Richemond, Pierre H and Buchatskaya, Elena and Doersch, Carl and Pires, Bernardo Avila and Guo, Zhaohan Daniel and Azar, Mohammad Gheshlaghi and others},
  booktitle={Advances in Neural Information Processing Systems},
  volume={33},
  pages={21271--21284},
  year={2020}
}
```

## License

This implementation is provided as-is for research purposes.

