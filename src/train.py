import argparse
import numpy as np
import yaml
from torch.utils.data import DataLoader, random_split
from img_datasets import LatentDataset, LatentClassesDataset, LatentContextDataset
from img_transforms import MultiCropTensorTransform, TensorTransform
from trainer import UniversalTrainer
from neptune_manager import NeptuneManager
import torch
import pandas as pd


def load_data():
    """Load and concatenate data from all classes."""
    classes = ['Bar', 'Shelf', 'Tidal', 'WD']
    data_list = []
    
    for cls in classes:
        data = np.load(f'../data/VAEB-426/{cls}/latent.npy')
        print(data.shape)
        data = np.transpose(data, (0, 4, 2, 3, 1))
        data = np.squeeze(data, axis=4)
        data_list.append(data)
    
    return np.concatenate(data_list, axis=0)


def load_context(fname: str):
    classes = ['Bar', 'Shelf', 'Tidal', 'WD']
    data_list = []
    for cls in classes:
        context = pd.read_csv(f'../data/VAEB-426/{cls}/{fname}.csv', index_col=0)
        context = context.to_numpy()
        data_list.append(context)
    return np.concatenate(data_list, axis=0)


def create_labels(num_samples_per_class: int, num_classes: int):
    """Create one-hot encoded labels."""
    labels = np.repeat(np.arange(num_classes), num_samples_per_class)
    return np.eye(num_classes)[labels]


def create_dataloaders(config: dict, data: np.ndarray, train_ratio: float = 0.8):
    """Create train and validation dataloaders based on config."""
    if config['model']['type'] == 'dino':
        transform = MultiCropTensorTransform(config)
        dataset = LatentDataset(data, transform=transform)
        train_loader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=4
        )
        return train_loader, None
    elif config['model']['type'] == 'classifier':
        transform = TensorTransform(config)
        labels = create_labels(5001, 4)  # Assuming 5001 samples per class and 4 classes
        dataset = LatentClassesDataset(data, labels, transform=transform)
        
        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = total_size - train_size
        
        # Split dataset
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
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
            shuffle=False,  # No need to shuffle validation data
            num_workers=4
        )
        
        return train_loader, val_loader
    
    elif config['model']['type'] == 'regressor':
        transform = TensorTransform(config)
        context = load_context(config['data']['context_name'])
        dataset = LatentContextDataset(data, context, transform=transform)

        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = total_size - train_size
        
        # Split dataset
        train_dataset, val_dataset = random_split(
            dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)  # For reproducibility
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
            shuffle=False,  # No need to shuffle validation data
            num_workers=4
        )
        
        return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Universal training script')
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config file')
    parser.add_argument('-p', '--project', type=str, default='your-project-name', help='Neptune project name')
    parser.add_argument('-s', '--sampler', type=str, default='TPESampler', help='Optuna sampler')
    parser.add_argument('-t', '--trials', type=int, default=100, help='Number of Optuna trials')
    parser.add_argument('-n', '--name', type=str, default='', help='Experiment name')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Ratio of training data to total data')
    args = parser.parse_args()
    
    # Load data
    print('Loading data')
    data = load_data()
    
    # Load config
    print('Loading config')
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize Neptune manager
    print('Starting Neptune')
    neptune_manager = NeptuneManager(project=args.project, config=config, name=args.name)
    
    if neptune_manager.sweep:
        def inner_objective(trial_config, run_trial_level):
            # Create dataloaders with trial config
            train_loader, val_loader = create_dataloaders(
                trial_config, 
                data,
                train_ratio=args.train_ratio
            )
            _, train_context = next(iter(train_loader))
            context_size = train_context.shape[1]
            trial_config['model']['classifier_head']['context_dim'] = context_size
            # Initialize and train model
            trainer = UniversalTrainer(trial_config, run_trial_level)
            best_metrics = trainer.train(train_loader, val_loader)
            
            return best_metrics
            
        neptune_manager.optimize(inner_objective, sampler=args.sampler, n_trials=args.trials)
    else:
        # Create dataloaders with base config
        train_loader, val_loader = create_dataloaders(
            neptune_manager.config, 
            data,
            train_ratio=args.train_ratio
        )
        _, train_context = next(iter(train_loader))
        context_size = train_context.shape[1]
        neptune_manager.config['model']['classifier_head']['context_dim'] = context_size
        # Initialize and train model
        trainer = UniversalTrainer(neptune_manager.config, neptune_manager.run)
        trainer.train(train_loader, val_loader)
        
    neptune_manager.finish()


if __name__ == "__main__":
    main() 