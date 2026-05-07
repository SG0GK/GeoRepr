"""
BYOL inference script for extracting representations from DataCubes.

This script:
1. Loads config + checkpoint
2. Builds BYOL model
3. Runs non-augmented dataloader
4. Saves representations + projections to .npy files
"""

import os
import sys
import argparse
from pathlib import Path

import yaml
import torch
import numpy as np
from typing import Union
from torch.utils.data import DataLoader, Dataset

# Add paths for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from byol.model import BYOL
from byol.data_parce.data import DataCubes
from byol.train_byol import setup_model_config


def _load_checkpoint_state(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def _build_repr_loader(datasets, batch_size, num_workers):
    return DataLoader(
        datasets,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

class _NpyDataset(Dataset):
    """Dataset that loads slices from a numpy array and returns them as tensors with given dtype."""

    def __init__(self, arr: np.ndarray, dtype: torch.dtype = torch.float32):
        self.arr = arr
        self.dtype = dtype

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, index):
        return torch.from_numpy(self.arr[index]).to(self.dtype)


def npy_to_dataloader(
    npy_path: Union[str, Path],
    dtype: torch.dtype = torch.float32,
    **kwargs,
) -> DataLoader:
    """
    Load best_cubes.npy from each subfolder of a directory and return a PyTorch DataLoader.

    Expects the directory to contain subfolders; each subfolder must have best_cubes.npy
    with shape [4, 75, 72, 40]. Arrays are stacked along axis 0 into shape [n_folders, 4, 75, 72, 40].

    Args:
        npy_path: Path to the directory containing subfolders with best_cubes.npy.
        batch_size: Batch size for the DataLoader.
        shuffle: Whether to shuffle the data.
        num_workers: Number of worker processes for loading.
        dtype: Torch dtype for the loaded tensors (e.g. float32, float16).
        **kwargs: Additional arguments passed to DataLoader (e.g. pin_memory, drop_last).

    Returns:
        DataLoader yielding batches of tensors (each sample shape [4, 75, 72, 40]).
    """
    dir_path = Path(npy_path)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Expected a directory: {dir_path}")
    subdirs = sorted(d for d in dir_path.iterdir() if d.is_dir())
    if not subdirs:
        raise FileNotFoundError(f"No subfolders found in {dir_path}")
    arrays = []
    for subdir in subdirs:
        npy_file = subdir / "best_cube.npy"
        if not npy_file.exists():
            raise FileNotFoundError(f"best_cubes.npy not found in {subdir}")
        arr = np.load(str(npy_file))
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        arrays.append(arr)
    data = np.stack(arrays, axis=0)  # [n_folders, 4, 75, 72, 40]
    dataset = _NpyDataset(data, dtype)
    return dataset


@torch.no_grad()
def extract_representations(
    config,
    checkpoint_path,
    device="cuda",
    output_dir=None,
    batch_size=None,
    num_workers=None,
    use_target=False,
    normalize=False
):
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = "cpu"

    # Dataset and dataloader (non-augmented)
    if config["data"]["data_path"] == "/gpfs/gpfs0/georoerich-lab/MVP/" \
        or config["data"]["data_path"] == "/trinity/home/egor.miroshnichenko/representations/byol/data/":
        datasets = DataCubes(config)
    else:
        datasets = npy_to_dataloader(config["data"]["data_path"])
    effective_batch_size = batch_size or config.get("batch_size", 12)
    effective_num_workers = num_workers if num_workers is not None else config.get("num_workers", 4)
    repr_loader = _build_repr_loader(datasets, effective_batch_size, effective_num_workers)

    # Model config
    model_config = setup_model_config(config)
    training_config = config.get("training", {})
    projection_dim = training_config.get("projection_dim", 384)
    hidden_dim = training_config.get("hidden_dim", 4096)
    base_momentum = training_config.get("base_momentum", 0.996)
    freeze_2d_encoder=training_config.get('freeze_2d_encoder', True)

    model = BYOL(
            config=model_config,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            moving_average_decay=base_momentum,
            freeze_2d_encoder=freeze_2d_encoder,
            is_train=False
        ).to(device)

    # Load checkpoint
    state_dict = _load_checkpoint_state(checkpoint_path, device)
    model.load_state_dict(state_dict, strict=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")    
    model.eval()

    all_representations = []
    all_projections = []

    for batch in repr_loader:
        if isinstance(batch, (tuple, list)):
            data = batch[0].to(device)
        else:
            data = batch.to(device)

        # representations = model.get_representations(
        #     data,
        #     use_target=use_target,
        #     from_projector=False,
        #     normalize=normalize
        # )
        projections = model.get_representations(
            data,
            use_target=use_target,
            from_projector=True,
            normalize=normalize
        )

        # all_representations.append(representations.cpu().numpy())
        all_projections.append(projections.cpu().numpy())

    # all_representations = np.concatenate(all_representations, axis=0)
    all_projections = np.concatenate(all_projections, axis=0)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "representations.npy", all_representations)
        # np.save(output_dir / f'{config["data"]["datasets"][0]["dirname"]}.npy', all_projections)

    return all_projections


def main():
    parser = argparse.ArgumentParser(description="Extract BYOL representations")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (.pt)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for inference"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="representations_inference",
        help="Directory to save representations/projections"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size (defaults to config batch_size)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override num_workers (defaults to config num_workers)"
    )
    parser.add_argument(
        "--use-target",
        action="store_true",
        help="Use target (EMA) network instead of online"
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Apply L2 normalization to outputs"
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    projs = extract_representations(
        config=config,
        checkpoint_path=args.checkpoint,
        device=args.device,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_target=args.use_target,
        normalize=args.normalize
    )

    print("Inference complete.")
    # print(f"Representations shape: {reps.shape}")
    print(f"Projections shape: {projs.shape}")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
