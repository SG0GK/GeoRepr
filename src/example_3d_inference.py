#!/usr/bin/env python3
"""
Example usage of the 3D inference pipeline.
"""

import subprocess
import sys
from pathlib import Path

def run_3d_inference_example():
    """Example showing how to use the 3D inference pipeline."""
    
    print("=== 3D Inference Pipeline Example ===")
    print()
    
    # Example paths (adjust these to your actual paths)
    config_path = "src/configs/dino_3d_config.yaml"
    checkpoint_path = "checkpoints/best_model.pth"  # Your trained model
    data_dirs = [
        "../data/wd_tiny",
        "../data/tidal_tiny", 
        "../data/shelf_tiny",
        "../data/bar_tiny"
    ]
    output_path = "output/3d_representations.npy"
    
    print("Example 1: Self-supervised DINO model inference")
    print("=" * 50)
    
    cmd_dino = [
        "python", "src/get_representations_3d.py",
        "-c", config_path,
        "-m", checkpoint_path,
        "-d"] + data_dirs + [
        "-o", output_path,
        "-b", "2",  # Small batch size for safety
        "--max-samples", "20"  # Limit samples for testing
    ]
    
    print("Command:")
    print(" ".join(cmd_dino))
    print()
    
    print("Example 2: Supervised model with labels")
    print("=" * 50)
    
    labels_files = [
        "../data/wd_tiny/stats.csv",
        "../data/tidal_tiny/stats.csv",
        "../data/shelf_tiny/stats.csv", 
        "../data/bar_tiny/stats.csv"
    ]
    
    cmd_supervised = [
        "python", "src/get_representations_3d.py",
        "-c", config_path,
        "-m", checkpoint_path,
        "-d"] + data_dirs + [
        "-l"] + labels_files + [
        "-o", "output/3d_representations_with_labels.npy",
        "-b", "2",
        "--max-samples", "20"
    ]
    
    print("Command:")
    print(" ".join(cmd_supervised))
    print()
    
    print("Example 3: CPU inference")
    print("=" * 50)
    
    cmd_cpu = [
        "python", "src/get_representations_3d.py",
        "-c", config_path,
        "-m", checkpoint_path,
        "-d", data_dirs[0],  # Single directory
        "-o", "output/3d_representations_cpu.npy",
        "-b", "1",
        "--device", "cpu",
        "--max-samples", "10"
    ]
    
    print("Command:")
    print(" ".join(cmd_cpu))
    print()
    
    print("Usage Notes:")
    print("=" * 50)
    print("1. Make sure you have a trained 3D model checkpoint (.pth file)")
    print("2. Adjust data directory paths to match your data location")
    print("3. Use smaller batch sizes to avoid memory issues")
    print("4. The output will be saved as .npy files with optional labels and metadata")
    print("5. For supervised models, labels will be saved as .labels.npy")
    print("6. Metadata will be saved as .metadata.yaml")
    print()
    
    print("Output files structure:")
    print("- representations.npy: Main feature vectors [N, feature_dim]")
    print("- representations.labels.npy: Labels if provided [N, label_dim]") 
    print("- representations.metadata.yaml: Metadata about the extraction")


def check_requirements():
    """Check if required files exist."""
    print("Checking requirements...")
    
    required_files = [
        "src/get_representations_3d.py",
        "src/configs/dino_3d_config.yaml",
        "src/img_datasets.py",
        "src/img_transforms.py", 
        "src/model.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    else:
        print("✅ All required files found")
        return True


if __name__ == "__main__":
    print("3D Inference Pipeline Setup and Examples")
    print("=" * 60)
    print()
    
    if check_requirements():
        print()
        run_3d_inference_example()
    else:
        print()
        print("Please ensure all required files are present before running inference.")
        sys.exit(1) 