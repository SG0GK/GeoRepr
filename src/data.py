import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from scaler import Scaler, YScaler
from utils import read_yaml


def parse_data_config(config: dict, mode: str, i: int) -> dict:
    """Reads a separate data config (description) that is specified at config['data_path']
    Adds it to the main config. Also adds n_train, n_test, seed and scaler parameters to data

    Args:
        config (dict): configuration (description) of a run
        mode (str): train or test

    Returns:
        dict: data config
    """
    data_config = read_yaml(config['data_path'][mode][i])
    data_config.update({'seed': config['seed']})
    data_config.update({'scaler': config['scaler']})
    return data_config


class Data(object):

    def __init__(self, config: dict):
        """Class for creating datasets and dataloaders from numpy files

        Args:
            config (dict): data config that contains paths to .npy files and also seed, n_train and n_test
        """
        self.config = config
        self.train_dataset, self.test_dataset = None, None
        self.scaler = dict({key: Scaler(value)
                           for key, value in config['scaler'].items()})
        self.yscaler = None
        self.seed = config['seed']
        self.n_train = config['n_train']
        self.n_test = config['n_test']
        self.mask = None
        self.arrays = tuple()
        self.mode = str()
        self.titles = None

    def get_dataset(self, X, y):
        """Get train and test datasets
        """
        generator = torch.Generator('cpu').manual_seed(42)
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(
            TensorDataset(X, y), [self.n_train, self.n_test], generator=generator
        )

    def get_dataloader(self, batch_size: int) -> dict:
        """Get data loaders

        Args:
            batch_size (int): batch size
        Returns:
            dict: dict with train and test dataloaders
        """
        print('Getting dataloaders ...')
        train_dataloader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False)
        dataloaders = {
            'train': train_dataloader,
            'test': test_dataloader
        }
        return dataloaders


class DataCubes(Data):

    def load_data(self, batch_size=100):
        """Loads 3D cubes from npy files

        Returns:
            tuple: cubes, labels of cubes (names of datasets)
        """
        n_samples, Y_labels = list(), list()
        Y = list()
        for i, data_item in enumerate(self.config['data']['datasets']):
            n_samples.append(data_item['n'])
            Y_labels.append(torch.full((data_item['n'],), i))
            df_dataset = list()
            for df_name in self.config['data']['stats_dfs']:
                df_path = os.path.join(os.path.dirname(data_item['data_path']), df_name)
                df = pd.read_csv(df_path, index_col=0)[:data_item['n']]
                df_dataset.append(df.reset_index(drop=True))
            df_dataset = pd.concat(df_dataset, axis=1)
            Y.append(df_dataset)
        Y = pd.concat(Y, axis=0)
        self.labels = torch.hstack(Y_labels)
        self.titles = list(Y.keys())

        self.yscaler = YScaler(
            Y,
            group_list=self.config['yscaler']['group_names'],
            a=self.config['yscaler']['a'],
            b=self.config['yscaler']['b']
        )
        Y = torch.from_numpy(Y.values.astype(np.float32))
        Y = self.yscaler.scale(Y)
        Y_nan_mask = torch.isnan(Y)
        Y[Y_nan_mask] = 0.0

        print('Allocating memory...')
        X = torch.zeros(
            sum(n_samples), *self.config['data']['dim'], dtype=torch.float32
        )
        self.n_train = X.shape[0] - self.n_test

        print('Loading data...')
        for i, data_item in enumerate(self.config['data']['datasets']):
            i_min, i_max = sum(n_samples[:i]), sum(n_samples[:i + 1])
            data_config = read_yaml(data_item['data_path'])
            for c, fname in enumerate(data_config['fnames']['X']):
                print(f'Loading {fname} from {data_config["data_path"]} ...')
                X[i_min:i_max, c] = torch.from_numpy(np.load(
                    os.path.join(data_config['data_path'], fname)
                )[:n_samples[i]])

        print('Scaling')
        with torch.inference_mode():
            for i in range(0, X.shape[0], batch_size):
                i_min, i_max = i, min(i + batch_size, X.shape[0])
                X[i_min:i_max, 0] = torch.clip(X[i_min:i_max, 0], min=0, max=1)
                X[i_min:i_max] = self.scaler['data'].scale(X[i_min:i_max])
        print('Scaling successfull')

        self.mask = torch.from_numpy(np.load(self.config['data']['mask']))

        return X, Y  #, Y_labels, Y_nan_mask


class DataCubesTest(DataCubes):

    def get_dataset(self, X, Y):
        """Get test datasets
        """
        self.test_dataset = TensorDataset(X, Y)

    def get_dataloader(self, batch_size: int) -> dict:
        """Get data loaders

        Args:
            batch_size (int): batch size
        Returns:
            dict: dict with train and test dataloaders
        """
        print('Getting dataloaders ...')
        test_dataloader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False)
        dataloaders = {
            'test': test_dataloader
        }
        return dataloaders
