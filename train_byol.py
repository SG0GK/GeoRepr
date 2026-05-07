"""
Complete BYOL training pipeline for DataCubes dataset.

This script:
1. Loads your config
2. Creates datasets using split_dataset(config)
3. Wraps them with BYOL augmentation
4. Trains using BYOLTrainer
5. Generates visualization plots
"""

import os
import sys
import yaml
import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# Add paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from byol.trainer import BYOLTrainer
from byol.augmentation import (
    get_light_augmentation,
    get_medium_augmentation,
    get_strong_augmentation,
    Compose,
    RandomCrop3D,
    RandomFlip3D,
    GaussianNoise3D,
    RandomIntensityScale3D,
    ClampValues
)
from byol.plotting import create_all_plots
from byol.data_parce.data import DataCubes
from byol.data_parce.byol_wrapper import prepare_byol_dataloaders_from_datacubes
from byol.clearml_manager import ClearMLManager


def create_custom_augmentation(config):
    """
    Create augmentation pipeline based on config.
    
    Args:
        config (dict): Configuration dictionary with augmentation settings
    
    Returns:
        Compose: Augmentation pipeline
    """
    aug_config = config.get('augmentation', {})
    
    # Get augmentation level
    level = aug_config.get('level', 'medium')
    crop_size = aug_config.get('crop_size', None)
    
    if level == 'light':
        return get_light_augmentation(crop_size=crop_size)
    elif level == 'medium':
        return get_medium_augmentation(crop_size=crop_size)
    elif level == 'strong':
        return get_strong_augmentation(crop_size=crop_size)
    elif level == 'custom':
        # Build custom augmentation from config
        transforms = []
        
        if crop_size is not None:
            transforms.append(RandomCrop3D(crop_size))
        
        if aug_config.get('flip', True):
            flip_prob = aug_config.get('flip_prob', 0.5)
            transforms.append(RandomFlip3D(p_h=flip_prob, p_w=flip_prob, p_d=flip_prob))
        
        if aug_config.get('noise', True):
            noise_std = aug_config.get('noise_std', 0.1)
            transforms.append(GaussianNoise3D(std=noise_std))
        
        if aug_config.get('intensity_scale', True):
            scale_range = aug_config.get('intensity_scale_range', (0.9, 1.1))
            transforms.append(RandomIntensityScale3D(scale_range=scale_range))
        
        if aug_config.get('clamp', True):
            clamp_range = aug_config.get('clamp_range', (-1.0, 1.0))
            transforms.append(ClampValues(*clamp_range))
        
        return Compose(transforms)
    else:
        raise ValueError(f"Unknown augmentation level: {level}")


def setup_model_config(config):
    """
    Setup model configuration from config file.
    
    Args:
        config (dict): Configuration dictionary
    
    Returns:
        dict: Model configuration for BYOL
    """
    model_cfg = config.get('model', {})
    
    # Default encoder_3d configuration
    encoder_3d_config = model_cfg.get('encoder_3d', {
        'in_channels': 3,
        'block_out_channels': [16, 32, 64],
        'block_out_types': ['down', 'down', 'same'],
        'num_groups': 16,
        'dropout': 0.01,
        'scales_w': [2, 2],
        'scales_h': [2, 2],
        'scales_d': [2, 2]
    })
    
    return {'encoder_3d': encoder_3d_config}


def train_byol(config_path, device='cuda', resume_from=None, clearml_project=None, run_name='', is_train=True):
    """
    Main training function.
    
    Args:
        config_path (str): Path to configuration YAML file
        device (str): Device to use ('cuda' or 'cpu')
        resume_from (str, optional): Path to checkpoint to resume from
        clearml_project (str, optional): ClearML project for logging (overrides config)
        run_name (str, optional): Name for ClearML run (overrides config)
    """
    
    print("=" * 80)
    print("BYOL Training Pipeline for DataCubes")
    print("=" * 80)
    
    # Load configuration
    print(f"\nLoading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Print some config info
    print(f"  Data path: {config['data']['data_path']}")
    print(f"  Data dim: {config['data']['dim']}")
    print(f"  Batch size: {config.get('batch_size', 8)}")
    
    clearml_manager = None
    clearml_run = None

    # Optional ClearML logging
    default_clearml_conf = Path(__file__).resolve().parent / "configs" / "clearml.conf"
    if default_clearml_conf.exists() and "CLEARML_CONFIG_FILE" not in os.environ:
        os.environ["CLEARML_CONFIG_FILE"] = str(default_clearml_conf)
        print(f"\nUsing ClearML config: {default_clearml_conf}")

    clearml_cfg = config.get('clearml', {})
    project_name = clearml_project or clearml_cfg.get('project')
    run_label = run_name or clearml_cfg.get('name', '')
    if project_name:
        print("\nStarting ClearML logging...")
        clearml_manager = ClearMLManager(
            project=project_name,
            config=config,
            name=run_label
        )
        clearml_run = clearml_manager.run
        if resume_from is not None:
            clearml_run["train/resume_from"] = str(resume_from)
    
    # Setup device
    if device == 'cuda' and not torch.cuda.is_available():
        print("\nWarning: CUDA not available, using CPU")
        device = 'cpu'
    print(f"\nUsing device: {device}")
    
    # ========== Load Datasets ==========
    print("\n" + "-" * 80)
    print("Loading datasets...")
    print("-" * 80)
    
    dataset = DataCubes(config)
    

    
    # ========== Setup Augmentation ==========
    print("\n" + "-" * 80)
    print("Setting up augmentation...")
    print("-" * 80)
    
    transform = create_custom_augmentation(config)
    
    aug_config = config.get('augmentation', {})
    print(f"  Augmentation level: {aug_config.get('level', 'medium')}")
    print(f"  Crop size: {aug_config.get('crop_size', 'None')}")
    
    # ========== Prepare BYOL DataLoaders ==========
    print("\n" + "-" * 80)
    print("Preparing BYOL dataloaders...")
    print("-" * 80)
    
    batch_size = config.get('batch_size', 8)
    num_workers = config.get('num_workers', 4)

    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers)
    
    # dataloaders = prepare_byol_dataloaders_from_datacubes(
    #     datasets=datasets,
    #     batch_size=batch_size,
    #     transform=transform,
    #     num_workers=num_workers
    # )
    
    # train_loader = dataloaders['train']
    
    # Create representation dataloader (no augmentation, no shuffle)

    
    print(f"\nDataloaders ready:")
    print(f"  Train batches: {len(train_loader)}")
    
    
    # ========== Setup Model Configuration ==========
    print("\n" + "-" * 80)
    print("Setting up model configuration...")
    print("-" * 80)
    
    model_config = setup_model_config(config)
    print(f"  Encoder 3D config: {model_config['encoder_3d']}")
    
    # ========== Setup Training Configuration ==========
    print("\n" + "-" * 80)
    print("Setting up training configuration...")
    print("-" * 80)
    
    training_config = config.get('training', {})
    
    learning_rate = training_config.get('learning_rate', 0.2)
    weight_decay = training_config.get('weight_decay', 1.5e-6)
    projection_dim = training_config.get('projection_dim', 256)
    hidden_dim = training_config.get('hidden_dim', 4096)
    base_momentum = training_config.get('base_momentum', 0.996)
    max_epochs = training_config.get('max_epochs', 500)
    warmup_epochs = training_config.get('warmup_epochs', 10)
    checkpoint_dir = training_config.get('checkpoint_dir', 'checkpoints_byol')
    save_interval = training_config.get('save_interval', 10)
    save_representations_interval = training_config.get('save_representations_interval', None)
    
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Projection dim: {projection_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Base momentum: {base_momentum}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Warmup epochs: {warmup_epochs}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    if save_representations_interval is not None:
        print(f"  Save representations every: {save_representations_interval} epochs")
    
    # ========== Initialize Trainer ==========
    print("\n" + "-" * 80)
    print("Initializing BYOL trainer...")
    print("-" * 80)
    
    trainer = BYOLTrainer(
        model_config=model_config,
        train_loader=train_loader,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        projection_dim=projection_dim,
        hidden_dim=hidden_dim,
        base_momentum=base_momentum,
        max_epochs=max_epochs,
        warmup_epochs=warmup_epochs,
        checkpoint_dir=checkpoint_dir,
        log_interval=training_config.get('log_interval', 10),
        save_interval=save_interval,
        run=clearml_run,
        save_representations_interval=save_representations_interval,
        freeze_2d_encoder=training_config.get('freeze_2d_encoder', True),
        is_train=is_train,
        fine_tune=training_config.get('fine_tune', False)
    )
    
    print("  ✓ Trainer initialized")
    
    # ========== Train ==========
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80 + "\n")
    
    try:
        trainer.train(resume_from=resume_from)
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback
        traceback.print_exc()
        return
    finally:
        if clearml_manager is not None:
            # Log final summary before closing
            clearml_run["train/final/best_train_loss"] = trainer.best_train_loss
            clearml_run["artifacts/checkpoint_dir"] = str(checkpoint_dir)
            clearml_run["artifacts/plots_dir"] = training_config.get('plots_dir', 'plots_byol')
            clearml_manager.finish()
    
    # ========== Generate Plots ==========
    print("\n" + "-" * 80)
    print("Generating visualization plots...")
    print("-" * 80)
    
    history_path = Path(checkpoint_dir) / 'training_history.json'
    plots_dir = training_config.get('plots_dir', 'plots_byol')
    
    if history_path.exists():
        try:
            create_all_plots(
                history_path,
                output_dir=plots_dir,
                show=False
            )
            print(f"  ✓ Plots saved to {plots_dir}/")
        except Exception as e:
            print(f"  Error generating plots: {e}")
    else:
        print("  No training history found. Skipping plot generation.")
    
    # ========== Summary ==========
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    
    print(f"\nResults:")
    print(f"  - Best training loss: {trainer.best_train_loss:.4f}")
    print(f"  - Checkpoints: {checkpoint_dir}/")
    print(f"  - Best model: {checkpoint_dir}/best_model.pt")
    print(f"  - Plots: {plots_dir}/")
    
    print(f"\nTo resume training:")
    print(f"  python train_byol.py --config {config_path} --resume {checkpoint_dir}/latest_checkpoint.pt")
    
    print(f"\nTo use the trained encoder:")
    print(f"""
    from byol import BYOL
    import torch
    
    # Load model
    model = BYOL(model_config)
    checkpoint = torch.load('{checkpoint_dir}/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Extract features
    model.eval()
    with torch.no_grad():
        features = model.online_network.encoder(volume_3d)
    """)


def main():
    parser = argparse.ArgumentParser(description='Train BYOL on DataCubes dataset')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for training'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--project',
        type=str,
        default="GeoR/BYOL",
        help='ClearML project name to enable logging'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default='',
        help='Optional ClearML run name'
    )
    
    parser.add_argument(
        '--is-train',
        type=bool,
        default=True,
        help='Train or infer mode'
    )
    
    args = parser.parse_args()
    
    # Train
    train_byol(
        config_path=args.config,
        device=args.device,
        resume_from=args.resume,
        clearml_project=args.project,
        run_name=args.run_name,
        is_train=args.is_train
    )


if __name__ == "__main__":
    main()

