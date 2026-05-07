import argparse
import yaml
import torch
from torch.utils.data import DataLoader, random_split
from img_datasets import GeologicalDataset3D, GeologicalDataset3DWithLabels
from img_transforms import MultiCrop3DTransform, Transform3D
from trainer import UniversalTrainer
from neptune_manager import NeptuneManager


def create_dataloaders_3d(config: dict, data_dirs, train_ratio: float = 0.8):
    """Create train and validation dataloaders for 3D geological data 
    with trainable encoder."""
    
    if config['model']['type'] == 'dino':
        # Self-supervised DINO training with trainable 3D encoder
        # Use 3D multi-crop transform (no encoding in transform)
        transform = MultiCrop3DTransform(config)
        dataset = GeologicalDataset3D(
            data_dirs=data_dirs, 
            transform=transform,
            max_samples=config.get('max_samples', None)
        )
        
        train_loader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=4
        )
        return train_loader, None
        
    else:
        # Supervised training (classification/regression) with trainable encoder
        # Use simple 3D transform (no encoding in transform)
        transform = Transform3D(config)
        
        # For supervised learning, we need labels
        # Use labels_files from config if provided, otherwise default to output.csv
        if 'labels_files' in config:
            labels_files = config['labels_files']
        elif isinstance(data_dirs, str):
            labels_files = f"{data_dirs}output.csv"
        else:
            # Multiple directories - default to output.csv in each directory
            labels_files = [
                f"{data_dir}output.csv" for data_dir in data_dirs
            ]
        
        dataset = GeologicalDataset3DWithLabels(
            data_dirs=data_dirs,
            labels_files=labels_files,
            transform=transform,
            max_samples=config.get('max_samples', None)
        )
        
        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = total_size - train_size
        
        # Split dataset
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=4
        )
        
        return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='3D geological data training')
    parser.add_argument('-c', '--config', type=str, required=True, 
                       help='Path to config file')
    parser.add_argument('-d', '--data-dir', type=str, 
                        default=None,
                        help='Single directory containing 3D data files '
                             '(overrides config)')
    parser.add_argument('-p', '--project', type=str, 
                       default='geological-3d', 
                       help='Neptune project name')
    parser.add_argument('-s', '--sampler', type=str, 
                       default='TPESampler', 
                       help='Optuna sampler')
    parser.add_argument('-t', '--trials', type=int, default=100, 
                       help='Number of Optuna trials')
    parser.add_argument('-n', '--name', type=str, default='', 
                       help='Experiment name')
    parser.add_argument('--train-ratio', type=float, default=0.8, 
                       help='Ratio of training data to total data')
    args = parser.parse_args()
    
    # Load config
    print('Loading config')
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine data directories from config or command line argument
    if args.data_dir:
        # Command line argument overrides config
        data_dirs = args.data_dir
        print(f"Using data directory from command line: {data_dirs}")
    elif 'data_dirs' in config:
        # Use multiple directories from config
        data_dirs = config['data_dirs']
        print(f"Using data directories from config: {data_dirs}")
    elif 'data_dir' in config:
        # Use single directory from config (backward compatibility)
        data_dirs = config['data_dir']
        print(f"Using data directory from config: {data_dirs}")
    else:
        raise ValueError(
            "No data directories specified. Use --data-dir argument "
            "or set data_dirs/data_dir in config."
        )
    
    # Initialize Neptune manager
    print('Starting Neptune')
    neptune_manager = NeptuneManager(
        project=args.project, 
        config=config, 
        name=args.name
    )
    
    if neptune_manager.sweep:
        def inner_objective(trial_config, run_trial_level):
            # Create dataloaders with trial config
            train_loader, val_loader = create_dataloaders_3d(
                trial_config, 
                data_dirs,
                train_ratio=args.train_ratio
            )
            
            # Initialize and train model
            trainer = UniversalTrainer(trial_config, run_trial_level)
            best_metrics = trainer.train(train_loader, val_loader)
            
            return best_metrics
            
        neptune_manager.optimize(
            inner_objective, 
            sampler=args.sampler, 
            n_trials=args.trials
        )
    else:
        # Create dataloaders with base config
        train_loader, val_loader = create_dataloaders_3d(
            neptune_manager.config, 
            data_dirs,
            train_ratio=args.train_ratio
        )
        
        # Initialize and train model using standard 2D DINO
        trainer = UniversalTrainer(neptune_manager.config, neptune_manager.run)
        trainer.train(train_loader, val_loader)
        
    neptune_manager.finish()


if __name__ == "__main__":
    main() 