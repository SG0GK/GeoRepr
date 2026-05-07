import torch
import numpy as np
from torch.utils.data import Dataset
import os


class LatentDataset(Dataset):
    """Memory-efficient latent dataset with lazy loading."""
    
    def __init__(self, features, transform=None):
        self.transform = transform
        
        if isinstance(features, str):
            # Use memory mapping for large files
            self.data = np.load(features, mmap_mode='r')
        elif isinstance(features, np.ndarray):
            self.data = features
        else:
            raise TypeError("features must be path to .npy file or np.ndarray")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Use torch.from_numpy for zero-copy conversion when possible
        image = torch.from_numpy(self.data[idx].copy()).float()
        if self.transform:
            image = self.transform(image)
        return image


class LatentClassesDataset(Dataset):
    """Memory-efficient latent classes dataset with lazy loading."""
    
    def __init__(self, features, classes, transform=None):
        self.transform = transform
        
        if isinstance(features, str):
            self.data = np.load(features, mmap_mode='r')
        elif isinstance(features, np.ndarray):
            self.data = features
        else:
            raise TypeError("features must be path to .npy file or np.ndarray")

        if isinstance(classes, str):
            self.targets = np.load(classes, mmap_mode='r')
        elif isinstance(classes, np.ndarray):
            self.targets = classes
        else:
            raise TypeError("classes must be path to .npy file or np.ndarray")
            
        assert self.targets.shape[0] == self.data.shape[0], "Number of classes must match number of data"
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.data[idx].copy()).float()
        if self.transform:
            image = self.transform(image)
        classes = torch.from_numpy(self.targets[idx].copy()).float()
        return image, classes
    

class LatentContextDataset(Dataset):
    """Memory-efficient latent context dataset with lazy loading."""
    
    def __init__(self, data, context, transform=None):
        self.transform = transform
        
        if isinstance(data, str) and data.endswith('.npy'):
            self.data = np.load(data, mmap_mode='r')
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            raise TypeError("data must be path to .npy file or np.ndarray")

        if isinstance(context, str) and context.endswith('.csv'):
            # Load CSV more efficiently
            self.targets = np.genfromtxt(context, delimiter=',')[1:, 1:]
        elif isinstance(context, np.ndarray):
            self.targets = context
        else:
            raise TypeError("context must be path to .csv file or np.ndarray")
            
        assert self.targets.shape[0] == self.data.shape[0], \
            "Number of contexts must match number of data"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.data[idx].copy()).float()
        if self.transform:
            image = self.transform(image)
        context = torch.from_numpy(self.targets[idx].copy()).float()
        return image, context
    
    def target_size(self):
        return self.targets.shape[1]


class GeologicalDataset3D(Dataset):
    """Memory-efficient 3D geological dataset with lazy loading from multiple 
    directories."""
    
    def __init__(self, data_dirs, transform=None, max_samples=None):
        self.transform = transform
        
        # Handle single directory or list of directories
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]
        
        self.data_dirs = data_dirs
        
        # Instead of loading all data, store file paths and metadata
        self.file_info = []
        total_samples = 0
        
        for data_dir in data_dirs:
            print(f"Scanning directory: {data_dir}")
            
            # Check file existence and get shape info
            facies_path = os.path.join(data_dir, 'facies.npy')
            poro_path = os.path.join(data_dir, 'poro.npy')
            perm_path = os.path.join(data_dir, 'perm.npy')
            
            files = [facies_path, poro_path, perm_path]
            if not all(os.path.exists(p) for p in files):
                raise FileNotFoundError(f"Missing data files in {data_dir}")
            
            # Get shape info without loading full data
            facies_shape = np.load(facies_path, mmap_mode='r').shape
            
            self.file_info.append({
                'data_dir': data_dir,
                'facies_path': facies_path,
                'poro_path': poro_path,
                'perm_path': perm_path,
                'n_samples': facies_shape[0],
                'start_idx': total_samples,
                'end_idx': total_samples + facies_shape[0]
            })
            
            total_samples += facies_shape[0]
            print(f"  Found {facies_shape[0]} samples in {data_dir}")
        
        self.n_samples = total_samples
        if max_samples:
            self.n_samples = min(self.n_samples, max_samples)
            print(f"Limited to {self.n_samples} samples")
        
        print(f"Total samples available: {self.n_samples}")
        
        # Memory map the files for efficient access
        self._memory_maps = {}
        
    def _get_memory_map(self, file_path):
        """Get or create memory map for a file."""
        if file_path not in self._memory_maps:
            self._memory_maps[file_path] = np.load(file_path, mmap_mode='r')
        return self._memory_maps[file_path]
            
    def __len__(self):
        return self.n_samples
        
    def __getitem__(self, idx):
        if idx >= self.n_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size "
                           f"{self.n_samples}")
        
        # Find which directory this index belongs to
        for info in self.file_info:
            if info['start_idx'] <= idx < info['end_idx']:
                local_idx = idx - info['start_idx']
                
                # Get memory mapped arrays
                facies_mm = self._get_memory_map(info['facies_path'])
                poro_mm = self._get_memory_map(info['poro_path'])
                perm_mm = self._get_memory_map(info['perm_path'])
                
                # Create sample efficiently using views
                facies_data = facies_mm[local_idx]
                poro_data = poro_mm[local_idx]
                perm_data = perm_mm[local_idx]
                
                # Stack into tensor (this creates a copy, but only for one sample)
                sample = torch.stack([
                    torch.from_numpy(facies_data.copy()).float(),
                    torch.from_numpy(poro_data.copy()).float(),
                    torch.from_numpy(perm_data.copy()).float()
                ], dim=0)
                
                if self.transform:
                    sample = self.transform(sample)
                    
                return sample
        
        raise IndexError(f"Index {idx} not found in any directory")

    class GeologicalDataset3DWithLabels(Dataset):
        """Memory-efficient 3D geological dataset with labels for supervised learning."""

        def __init__(self, data_dirs, labels_files, transform=None, max_samples=None):
            self.transform = transform

            # Handle single directory/file or lists
            if isinstance(data_dirs, str):
                data_dirs = [data_dirs]
            if isinstance(labels_files, str):
                labels_files = [labels_files]

            # Ensure we have matching numbers of data dirs and label files
            if len(labels_files) == 1 and len(data_dirs) > 1:
                labels_files = labels_files * len(data_dirs)
            elif len(labels_files) != len(data_dirs):
                raise ValueError(
                    f"Number of data directories ({len(data_dirs)}) must match "
                    f"number of label files ({len(labels_files)})"
                )

            self.data_dirs = data_dirs
            self.labels_files = labels_files
            self.labels = None
            labels_num = 0
            # Store file info and load labels efficiently
            self.file_info = []
            total_samples = 0

            for data_dir, labels_file in zip(data_dirs, labels_files):
                print(f"Scanning directory: {data_dir}")
                print(f"Loading labels from: {labels_file}")

                # Check file existence 
                facies_path = os.path.join(data_dir, 'facies.npy')
                poro_path = os.path.join(data_dir, 'poro.npy')
                perm_path = os.path.join(data_dir, 'perm.npy')

                files = [facies_path, poro_path, perm_path]
                if not all(os.path.exists(p) for p in files):
                    raise FileNotFoundError(f"Missing data files in {data_dir}")

                # Get shape info without loading full data
                facies_shape = np.load(facies_path, mmap_mode='r').shape
                if labels_num == 0:
                    # Load labels based on file type
                    if labels_file.endswith('.csv'):
                        # CSV files (context regression): skip header row and index column
                        labels = np.genfromtxt(labels_file, delimiter=',')[1:, 1:]
                        print(f"Loaded CSV labels shape: {labels.shape}")
                        print(f"First few CSV labels:\n{labels[:5]}")
                    elif labels_file.endswith('.npy'):
                        # NPY files (classification): load directly as class indices
                        labels = np.load(labels_file)
                        print(f"Loaded NPY labels shape: {labels.shape}")
                        print(f"First few NPY labels: {labels[:5]}")
                    else:
                        raise ValueError(f"Unsupported label file format: {labels_file}")
                    
                    self.labels = labels
                    labels_num += 1
                else:
                    # Load additional labels based on file type
                    if labels_file.endswith('.csv'):
                        labels = np.genfromtxt(labels_file, delimiter=',')[1:, 1:]
                    elif labels_file.endswith('.npy'):
                        labels = np.load(labels_file)
                    else:
                        raise ValueError(f"Unsupported label file format: {labels_file}")
                    
                    self.labels = np.concatenate((self.labels, labels))

                # Ensure data and labels have same number of samples
                min_samples = min(facies_shape[0], labels.shape[0])
                labels = labels[:min_samples]

                self.file_info.append({
                    'data_dir': data_dir,
                    'facies_path': facies_path,
                    'poro_path': poro_path,
                    'perm_path': perm_path,
                    #'labels': labels,  # Store labels in memory (small)
                    'n_samples': min_samples,
                    'start_idx': total_samples,
                    'end_idx': total_samples + min_samples
                })

                total_samples += min_samples
                print(f"  Found {min_samples} samples in {data_dir}")

            self.n_samples = total_samples
            if max_samples:
                self.n_samples = min(self.n_samples, max_samples)
                print(f"Limited to {self.n_samples} samples")

            print(f"Total samples available: {self.n_samples}")

            # Memory map the files for efficient access
            self._memory_maps = {}

        def _get_memory_map(self, file_path):
            """Get or create memory map for a file."""
            if file_path not in self._memory_maps:
                self._memory_maps[file_path] = np.load(file_path, mmap_mode='r')
            return self._memory_maps[file_path]

        def __len__(self):
            return self.n_samples

        def __getitem__(self, idx):
            if idx >= self.n_samples:
                raise IndexError(f"Index {idx} out of range for dataset of size "
                            f"{self.n_samples}")

            # Find which directory this index belongs to
            for info in self.file_info:
                if info['start_idx'] <= idx < info['end_idx']:
                    local_idx = idx - info['start_idx']

                    # Get memory mapped arrays
                    facies_mm = self._get_memory_map(info['facies_path'])
                    poro_mm = self._get_memory_map(info['poro_path'])
                    perm_mm = self._get_memory_map(info['perm_path'])

                    # Create sample efficiently using views
                    facies_data = facies_mm[local_idx]
                    poro_data = poro_mm[local_idx]
                    perm_data = perm_mm[local_idx]

                    # Stack into tensor (this creates a copy, but only for one sample)
                    sample = torch.stack([
                        torch.from_numpy(facies_data.copy()).float(),
                        torch.from_numpy(poro_data.copy()).float(),
                        torch.from_numpy(perm_data.copy()).float()
                    ], dim=0)

                    # Get labels efficiently
                    #labels = torch.from_numpy(info['labels'][local_idx].copy()).float()
                    label = self.labels[idx]
                    if self.transform:
                        sample = self.transform(sample)

                    return sample, label

            raise IndexError(f"Index {idx} not found in any directory")
