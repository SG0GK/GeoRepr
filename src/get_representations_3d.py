#!/usr/bin/env python3
"""
3D Inference Pipeline for Geological Data Representations

This script loads a trained 3D model and generates representations from 3D 
geological data. Supports both self-supervised (DINO) and supervised models 
with trainable 3D encoders.
"""

import argparse
import torch
import yaml
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Union, List
import os

from img_transforms import Transform3D
from img_datasets import GeologicalDataset3D, GeologicalDataset3DWithLabels
from model import DINOV2Analog3D, DinoClassifier3D, DinoRegressor3D, DINOV2Analog3DPretrained, DinoClassifier3DPretrained, DinoRegressor3DPretrained


def load_3d_model(
    checkpoint_path: str,
    config: dict,
    device: torch.device
) -> Union[DINOV2Analog3D, DinoClassifier3D, DinoRegressor3D, DINOV2Analog3DPretrained, DinoClassifier3DPretrained]:
    """Load 3D model from checkpoint."""
    
    model_type = config['model']['type']
    
    # Create the appropriate model
    if model_type == "dino":
        # Check if using pretrained DINOv2 backbone
        if config['model'].get('mode') == 'pretrained' and config['model'].get('data_type') == '3d':
            model = DINOV2Analog3DPretrained(config)
        else:
            model = DINOV2Analog3D(config)
    elif model_type == "classifier":
        # Choose classifier based on mode and data type
        if config['model'].get('mode') == 'pretrained' and config['model'].get('data_type') == '3d':
            model = DinoClassifier3DPretrained(config)
        else:
            model = DinoClassifier3D(config)
    elif model_type == "regressor":
        # Choose regressor based on mode and data type (same logic as trainer)
        if config['model'].get('data_type') == '3d' and config['model']['mode'] == 'pretrained':
            model = DinoRegressor3DPretrained(config)
        elif config['model'].get('data_type') == '3d':
            model = DinoRegressor3D(config)
        else:
            # Fallback for 2D regressors (if any)
            model = DinoRegressor3D(config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded from 'model_state_dict'")
    elif 'student_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['student_state_dict'])
        print("Loaded from 'student_state_dict' (DINO teacher-student)")
    elif 'student' in checkpoint:
        model.load_state_dict(checkpoint['student'])
        print("Loaded from 'student' (DINO teacher-student)")
    else:
        # Assume the checkpoint is the state dict itself
        model.load_state_dict(checkpoint)
        print("Loaded direct state dict")
    
    return model.to(device)


def create_3d_dataloader(
    data_dirs: Union[str, List[str]],
    config: dict,
    labels_files: Union[str, List[str]] = None,
    batch_size: int = None,
    max_samples: int = None,
    num_workers: int = 4
) -> DataLoader:
    """Create dataloader for 3D geological data."""
    
    model_type = config['model']['type']
    
    # Use inference-friendly batch size if not specified
    if batch_size is None:
        batch_size = config['training'].get('batch_size', 8)
        # Reduce batch size for inference to avoid memory issues
        batch_size = min(batch_size, 4)
    
    # Create appropriate transform (no multi-crop for inference)
    if model_type == "dino":
        # For inference, use simple transform instead of multi-crop
        transform = Transform3D(config)
    else:
        transform = Transform3D(config)
    
    # Create dataset
    if labels_files is not None:
        # Supervised dataset with labels
        dataset = GeologicalDataset3DWithLabels(
            data_dirs=data_dirs,
            labels_files=labels_files,
            transform=transform,
            max_samples=max_samples
        )
    else:
        # Unsupervised dataset (for DINO or general feature extraction)
        dataset = GeologicalDataset3D(
            data_dirs=data_dirs,
            transform=transform,
            max_samples=max_samples
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for inference
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  # Only pin memory if CUDA available
    )


def extract_3d_features(
    model: Union[DINOV2Analog3D, DinoClassifier3D, DinoRegressor3D, DINOV2Analog3DPretrained, DinoClassifier3DPretrained, DinoRegressor3DPretrained],
    dataloader: DataLoader,
    device: torch.device,
    model_type: str,
    return_labels: bool = True,
    save_predictions: bool = True
) -> Union[torch.Tensor, tuple]:
    """Extract features from 3D model."""
    
    model.eval()
    features = []
    labels = [] if return_labels else None
    predictions = [] if save_predictions and model_type == 'regressor' else None
    
    print(f"Extracting features using {model_type} model...")
    if save_predictions and model_type == 'regressor':
        print("Saving predictions for context regression model...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Processing batches")):
            
            if return_labels:
                batch, batch_labels = batch_data
                batch_labels = batch_labels.cpu()
                labels.append(batch_labels)
            else:
                batch = batch_data
            
            batch = batch.to(device)
            
                        # Forward pass through model
            if model_type == "dino":
                # For DINO models, get the CLS token representation
                output = model(batch)
                if isinstance(output, dict):
                    # 3D DINO models return dict with cls_token, all_tokens, projection
                    feature = output['cls_token']
                elif isinstance(output, tuple):
                    # If model returns (projection, features), use features
                    feature = output[1] if len(output) > 1 else output[0]
                else:
                    feature = output
            else:
                # For supervised models (DinoClassifier3D, DinoRegressor3D)
                output = model(batch)
                if isinstance(output, tuple):
                    # Supervised models return (predictions, features)
                    prediction, feature = output
                    # Save predictions for context regression
                    if save_predictions and model_type == 'regressor':
                        predictions.append(prediction.cpu())
                elif isinstance(output, dict):
                    # If backbone returns dict, get cls_token
                    feature = output['cls_token']
                else:
                    # Fallback: assume output is the feature
                    feature = output
            
            features.append(feature.cpu())
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # Concatenate all features
    all_features = torch.cat(features, dim=0)
    print(f"Extracted features shape: {all_features.shape}")
    
    # Simplified return logic for SVL case
    result = [all_features]
    
    # Always add labels if requested and available
    if return_labels and labels:
        all_labels = torch.cat(labels, dim=0)
        print(f"Labels shape: {all_labels.shape}")
        result.append(all_labels)
    
    # Always add predictions for regression models if available
    if save_predictions and predictions:
        all_predictions = torch.cat(predictions, dim=0)
        print(f"Predictions shape: {all_predictions.shape}")
        result.append(all_predictions)
    
    # Return based on what we have
    if len(result) == 1:
        return result[0]  # Just features
    elif len(result) == 2:
        if save_predictions and not return_labels:
            return result[0], result[1]  # (features, predictions)
        else:
            return result[0], result[1]  # (features, labels) or (features, predictions)
    else:
        return tuple(result)  # (features, labels, predictions)


def save_representations(
    features: torch.Tensor,
    output_path: str,
    labels: torch.Tensor = None,
    predictions: torch.Tensor = None,
    metadata: dict = None
):
    """Save representations and optional labels/metadata."""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save features
    features_np = features.numpy()
    np.save(output_path, features_np)
    print(f"Saved features to: {output_path}")
    
    # Save labels if provided (y_true for context regression)
    if labels is not None:
        labels_path = output_path.with_suffix('.y_true.npy')
        labels_np = labels.numpy()
        np.save(labels_path, labels_np)
        print(f"Saved y_true (ground truth) to: {labels_path}")
    
    # Save predictions if provided (y_pred for context regression)
    if predictions is not None:
        predictions_path = output_path.with_suffix('.y_pred.npy')
        predictions_np = predictions.numpy()
        np.save(predictions_path, predictions_np)
        print(f"Saved y_pred (predictions) to: {predictions_path}")
    
    # Save metadata if provided
    if metadata is not None:
        metadata_path = output_path.with_suffix('.metadata.yaml')
        with open(metadata_path, 'w') as f:
            yaml.dump(metadata, f, default_flow_style=False)
        print(f"Saved metadata to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract 3D representations from trained geological models'
    )
    parser.add_argument('-c', '--config', type=str, required=True,
                        help='Path to model config file')
    parser.add_argument('-m', '--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('-d', '--data-dirs', type=str, nargs='+', required=True,
                        help='Directories containing 3D data files')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output path for representations (.npy file)')
    parser.add_argument('-l', '--labels-files', type=str, nargs='*',
                        help='Optional labels files for supervised models')
    parser.add_argument('-b', '--batch-size', type=int, default=4,
                        help='Batch size for inference (default: 4)')
    parser.add_argument('-w', '--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Load config
    print("Loading configuration...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device with proper fallback
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Using CPU.")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Validate data directories
    for data_dir in args.data_dirs:
        if not os.path.exists(data_dir):
            print(f"Warning: Data directory does not exist: {data_dir}")
    
    # Load model
    model = load_3d_model(args.checkpoint, config, device)
    print(f"Loaded {config['model']['type']} model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dataloader
    return_labels = args.labels_files is not None
    dataloader = create_3d_dataloader(
        data_dirs=args.data_dirs,
        config=config,
        labels_files=args.labels_files,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        num_workers=args.num_workers
    )
    print(f"Created dataloader with {len(dataloader)} batches")
    
    # Simplified logic: For SVL regressors, always save predictions
    is_regressor = config['model']['type'] == 'regressor'
    save_predictions = is_regressor  # Always save predictions for regression models
    
    print(f"Model type: {config['model']['type']}")
    print(f"Return labels: {return_labels}")
    print(f"Save predictions: {save_predictions}")
    
    result = extract_3d_features(
        model, dataloader, device, config['model']['type'], 
        return_labels=return_labels, save_predictions=save_predictions
    )
    
    # Simplified parsing: handle all combinations clearly
    features = None
    labels = None
    predictions = None
    
    if isinstance(result, tuple):
        features = result[0]  # Features are always first
        
        if len(result) == 2:
            if return_labels and not save_predictions:
                # (features, labels)
                labels = result[1]
            elif save_predictions and not return_labels:
                # (features, predictions)
                predictions = result[1]
            elif save_predictions and return_labels:
                # This shouldn't happen with current logic, but handle gracefully
                labels = result[1]
        elif len(result) == 3:
            # (features, labels, predictions)
            labels = result[1]
            predictions = result[2]
    else:
        # Just features
        features = result
    
    # Prepare metadata
    metadata = {
        'model_type': config['model']['type'],
        'data_dirs': args.data_dirs,
        'checkpoint_path': args.checkpoint,
        'config_path': args.config,
        'feature_dim': features.shape[1],
        'num_samples': features.shape[0],
        'device_used': str(device),
        'batch_size': args.batch_size
    }
    
    # Save results
    save_representations(features, args.output, labels, predictions, metadata)
    
    print(f"\n✅ Successfully extracted {features.shape[0]} representations!")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main() 