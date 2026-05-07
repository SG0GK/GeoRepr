import os
import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from .scaler import ChannelScaler
from .padder import Padder3D
from losses.statistics_loss import MetaStatsLoss
# from models.vae_3d_new.utils import parse_channels


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))


def parse_dtype(dtype):
    dtype = dtype.lower()
    if dtype == 'float32':
        return np.float32
    elif dtype in ('byte', 'int8'):
        return np.int8
    elif dtype == 'uint8':
        return np.uint8
    elif dtype == 'float64':
        return np.float64
    elif dtype == 'int16':
        return np.int16
    elif dtype == 'int32':
        return np.int32
    else:
        raise ValueError(f"Unknown dtype: {dtype}")


class DataCubes(Dataset):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.X = dict()
        self.num_samples = list()
        self.dirnames = list()
        self.idxs = list()
        for dataset in config['data']['datasets']:
            self.X[dataset['dirname']] = dict()
            for file in config['data']['fnames']['X']:
                path = os.path.join(
                    config['data']['data_path'],
                    dataset['dirname'],
                    file['name']
                )
                print(f'Loading {path} ...')
                # self.X[dataset['dirname']][file['name']] = np.memmap(
                #     path, dtype=parse_dtype(file['dtype']),
                #     mode='r+', shape=(dataset['n'], *config['data']['dim']),
                # )
                self.X[dataset['dirname']][file['name']] = np.memmap(
                    path, mode='r', dtype=parse_dtype(file['dtype']), shape=(dataset.get('n', 100), *config['data']['dim']),
                )
            self.num_samples.append(dataset.get('n', 100))
            self.dirnames.append(dataset['dirname'])
            self.idxs.append(np.arange(dataset.get('n', 100)))
        self.idxs = np.hstack(self.idxs)
        self.num_samples = np.array(self.num_samples)
        self.num_samples_ = np.cumsum(self.num_samples)
        scaler_config = config['scaler'] if 'scaler' in config.keys() else {}
        self.mask = torch.from_numpy(np.load(os.path.join(
            config['data']['data_path'], config['data']['mask']
        )))
        self.grid = torch.from_numpy(np.load(os.path.join(
            config['data']['data_path'], config['data']['grid']
        )))
        self.scaler = ChannelScaler(scaler_config)
        # self.scaler = ChannelScalerWithCropper(scaler_config, mask=self.mask)
        # self.padder = Padder3D(scale=2)
        # self.scaler = ChannelScalerWithNTGCropper(scaler_config)
        # self.padder = Padder3D(scale=2)

    def __len__(self):
        return np.sum(self.num_samples)

    def __getitem__(self, index):
        dirname = self.dirnames[np.where(self.num_samples_ > index)[0][0]]
        X = torch.stack(list(
            torch.from_numpy(
                self.X[dirname][key][self.idxs[index]].copy()
            ) for key in self.X[dirname].keys()
        ), axis=0)
        X = X.unsqueeze(0)
        X = self.scaler.scale(X)
        X = X.squeeze(0)
        return X


class DataCubesWithStats(DataCubes):

    def __init__(self, config):
        super().__init__(config)
        self.meta_statistics_loss = MetaStatsLoss(
            mask=self.mask,
            loss_types=config['meta_statistics_loss']['loss_types'],
            weights=config['meta_statistics_loss']['weights']
        )
        self.y = list([None for _ in range(np.sum(self.num_samples))])

    def __getitem__(self, index):
        X = super().__getitem__(index)
        if self.y[index] is None:
            # print(f'Y at {index} not calculated yet...')
            self.y[index] = self.meta_statistics_loss.calc_stats(X.unsqueeze(0)).squeeze()
            # print(self.y[index].shape)
        return X, self.y[index]


def sample_vertical_wells(
        mask: torch.Tensor,
        min_wells=0,
        max_wells=9
) -> torch.Tensor:
    """Generates wells

    Args:
        mask (torch.Tensor): mask of size [3, 150, 144, 80]
        min_wells (int, optional): minimum wells to sample. Defaults to 0.
        max_wells (int, optional): maximum wells to sample. Defaults to 9.

    Returns:
    """
    n_wells = torch.randint(low=min_wells, high=max_wells+1, size=(1,))

    hor_mask = mask[:, :, 0]
    total_points = hor_mask.sum()

    wells_pos = torch.hstack((
        torch.ones(n_wells),
        torch.zeros(total_points - n_wells)
    ))[torch.randperm(total_points)].type(torch.bool)

    well_mask = torch.zeros_like(mask[:, :, 0]).type(torch.bool)

    well_mask[hor_mask] = wells_pos

    well_mask = torch.tile(well_mask[:, :, None], (1, 1, mask.shape[2]))

    return well_mask


class DataCubesWithStatsAndVerticalWells(DataCubesWithStats):

    def __init__(self, config):
        super().__init__(config)
        self.min_wells = config['data'].get('min_wells', 0)
        self.max_wells = config['data'].get('max_wells', 9)

    def __getitem__(self, index):
        X, y = super().__getitem__(index)
        well_mask = sample_vertical_wells(
            mask=self.mask,
            min_wells=self.min_wells,
            max_wells=self.max_wells
        )
        wells_i, wells_j, wells_k = torch.where(well_mask)
        X_C = torch.tile(
            well_mask[None], (X.shape[0] + 1, 1, 1, 1)
        ).type(torch.float32)
        X_C[0] = well_mask.type(torch.float32)
        X_C[1:, wells_i, wells_j, wells_k] = X[:, wells_i, wells_j, wells_k]
        return X, X_C, y


def split_dataset(config):
    dataset = DataCubes(config)
    generator = torch.Generator('cpu').manual_seed(config['seed'])
    n_test = config['data']['n_test']
    n_train = config['data']['n_train']
    if n_train is None:
        datasets = {'test': dataset}
    else:
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_test], generator=generator
        )
        datasets = {
            'train': train_dataset,
            'test': test_dataset
        }
    return datasets


def get_dataloaders(datasets, config):
    print(f"Batch size: {config['batch_size']}")
    dataloaders = dict()
    for key, dataset in datasets.items():
        shuffle = False 
        dataloaders[key] = DataLoader(
            dataset,
            batch_size=config['batch_size'],
            shuffle=shuffle
        )
    return dataloaders
