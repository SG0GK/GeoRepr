# Universal Training Pipeline

This repository contains a universal training pipeline that supports both self-supervised learning (SSL) using a DINO-like approach and supervised learning for image classification tasks. The pipeline is designed to be flexible and configurable through YAML configuration files.

## Features

- Support for both self-supervised and supervised learning approaches
- Ability to train models from scratch or fine-tune pretrained models
- Configuration-based training setup using YAML files
- Optional backbone freezing for transfer learning
- Integration with Neptune.ai for experiment tracking
- Support for Optuna hyperparameter optimization
- Early stopping and learning rate scheduling
- Checkpoint saving and loading

## Directory Structure

```
src/
├── configs/
│   ├── base_config.yaml
│   ├── ssl_config.yaml
│   └── supervised_config.yaml
├── model.py
├── trainer.py
├── train.py
├── img_datasets.py
├── img_transforms.py
└── utils.py
```

## Usage

1. Choose or create a configuration file in the `configs` directory. Two example configurations are provided:
   - `ssl_config.yaml`: For self-supervised learning using DINO
   - `supervised_config.yaml`: For supervised learning with optional backbone freezing

2. Run the training script with your chosen configuration:
```bash
python train.py --config configs/ssl_config.yaml  # For self-supervised learning
python train.py --config configs/supervised_config.yaml  # For supervised learning
```

## Configuration Options

### Training Parameters
- `batch_size`: Number of samples per batch
- `num_epochs`: Total number of training epochs
- `learning_rate`: Initial learning rate
- `weight_decay`: Weight decay for optimizer
- `warmup_epochs`: Number of warmup epochs for learning rate scheduling
- `device`: Training device ("cuda" or "cpu")

### Model Parameters
- `type`: Model type
- `mode`: Training mode ("from_scratch" or "pretrained")
- `backbone`: Vision transformer backbone configuration
- `freeze_backbone`: Whether to freeze backbone layers (for supervised learning)

### Data Parameters
- `input_channels`: Number of input channels
- `transform_type`: Type of data augmentation ("multicrop" or "single")
- `global_crop_size`: Size of global crops
- `local_crop_size`: Size of local crops (for SSL only)

### Optuna Parameters
- `enabled`: Whether to use Optuna for hyperparameter optimization
- `n_trials`: Number of optimization trials
- `direction`: Optimization direction ("minimize" or "maximize")
- `pruner`: Pruning strategy
- `sampler`: Sampling strategy

## Example: Training a Model

1. Self-supervised pretraining:
```bash
python train.py --config configs/ssl_config.yaml
```

2. Supervised fine-tuning:
```bash
python train.py --config configs/supervised_config.yaml
```

## Extending the Pipeline

To add new features or modify existing ones:

1. Model modifications: Edit `model.py`
2. Training logic changes: Edit `trainer.py`
3. Data processing changes: Edit `img_datasets.py` or `img_transforms.py`
4. New configuration options: Add to the YAML files in `configs/`

## Requirements

- PyTorch
- Optuna
- PyYAML
- NumPy
- tqdm 
