import argparse
import torch
import yaml
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Union

from img_transforms import TensorTransform, MultiCropTensorTransform
from img_datasets import LatentDataset
from model import DINOV2Analog, DinoClassifier, DinoRegressor


'''def load_data(data_paths: list) -> np.ndarray:
    """Load and concatenate data from multiple paths."""
    classes = ['Bar', 'Shelf', 'Tidal', 'WD']
    data_list = []
    
    for cls in classes:
    data_list = []
    for path in data_paths:
        data = np.load(path)
        data = np.transpose(data, (0, 4, 2, 3, 1))
        data = np.squeeze(data, axis=4)
        data_list.append(data)
    return np.concatenate(data_list, axis=0)'''


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


def load_model(
    checkpoint_path: str,
    model_type: str,
    device: torch.device
) -> Union[DINOV2Analog, DinoClassifier]:
    """Load model from checkpoint."""
    if model_type == "ssl":
        model = DINOV2Analog()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Handle both single model state dict and dictionary with student/teacher
        if 'student_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['student_state_dict'])
        elif 'student' in checkpoint:
            model.load_state_dict(checkpoint['student'])
        else:
            model.load_state_dict(checkpoint)
    else:  # supervised
        model = DinoClassifier()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    return model.to(device)


def create_dataloader(
    data: np.ndarray,
    config: dict,
    num_workers: int = 4
) -> DataLoader:
    """Create appropriate dataloader based on model type."""
    if config['model']['type'] == "dino":
        transform = MultiCropTensorTransform(config)
    else:  # supervised
        transform = TensorTransform(config)
    
    dataset = LatentDataset(data, transform=transform)
    return DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers
    )


def extract_features(
    model: Union[DINOV2Analog, DinoClassifier],
    dataloader: DataLoader,
    device: torch.device,
    model_type: str
) -> torch.Tensor:
    """Extract features from the model."""
    model.eval()
    features = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            # For SSL models, we only use the first global view
            if model_type == "dino":
                batch = batch[0].to(device)
            else:
                batch = batch.to(device)
            
            # Get features from the model
            _, feature = model(batch)
            features.append(feature.cpu())
    
    return torch.cat(features, dim=0)


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load data
    data = load_data()
    
    # Create dataloader
    dataloader = create_dataloader(data, config)
    
    # Set device
    device = torch.device(config['training']['device'])
    
    # Load model
    if config['model']['type'] == "dino":
        model = DINOV2Analog(config).to(device)
    elif config['model']['type'] == "classifier":
        model = DinoClassifier(config).to(device)
    elif config['model']['type'] == "regressor":
        model = DinoRegressor(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'student_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['student_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Get representations
    representations = extract_features(model, dataloader, device, config['model']['type'])
    
    # Save representations
    output_path = Path(args.output_dir) / f"{args.name}"
    np.save(output_path, representations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file")
    parser.add_argument("--data_paths", type=str, nargs="+", required=True,
                        help="Paths to data files")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save representations")
    parser.add_argument("--name", type=str, required=True,
                        help="Name for the output file")
    args = parser.parse_args()
    main(args) 
