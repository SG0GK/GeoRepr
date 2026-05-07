import neptune
import os
import yaml
import torch
from torch import nn
import numpy as np

from data import Data
from padder import Padder3D
from tqdm import tqdm
from timeit import default_timer
from collections import OrderedDict


def save_model(config: dict, state_dict: OrderedDict, neptune_run: neptune.Run,
               experiments_path='../experiments', name='model'):
    """Creates experiment directory.
    Saves model state dict into threre.
    Also saves the run (tiral) config in there.

    Args:
        config (dict): run (tiral) configuration
        state_dict (OrderedDict): model state dict
        neptune_run (neptune.Run): neptune run
        experiments_path (str): path to experiment folder. Defaults to '../experiments'
        name (str, optional): model name. Defaults to 'model'.
    """
    id = neptune_run["sys/id"].fetch()
    path = os.path.join(experiments_path, id)
    state_dict_path = os.path.join(path, f'{name}_state_dict.pt')
    config['model'][name]['state_dict_path'] = state_dict_path
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    torch.save(state_dict, state_dict_path)


class Train(object):

    def __init__(self, config: dict, data: Data, neptune_run: neptune.Run):
        """Train pipeline for ML model.

        Args:
            config (dict): training config
            data (Data): data class object
            neptune_run (neptune.Run): neptune run
        """
        self.config = config
        self.neptune_run = neptune_run
        self.data = data
        self.model, self.model_name = nn.Module(), str()
        self.key_metric = config['metrics'][0]['name']
        print(f'Early stopping according to {self.key_metric}')
        self.best_state_dict = dict()
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.bce_loss = nn.BCELoss(reduction='mean')
        self.loss = nn.MSELoss()
        self.current_loss = dict()
        self.if_save, self.if_load, self.if_train = bool(), bool(), bool()
        self.device, self.n_epochs = config['device'], config['n_epochs']
        self.patience = config['n_patience']
        self.epoch_num, self.epoch_title = None, str()
        self.inference_epochs = config['inference_epochs']
        self.dataloaders = self.data.get_dataloader(config['batch_size'])
        self.order = config['order']
        self.scaler = data.scaler
        self.yscaler = data.yscaler
        self.yscaler.to(self.device)
        torch.manual_seed(config['seed'])
        self.padder = dict({key: Padder3D(scale=value)
                           for key, value in config['padder'].items()})

    def to_device(self, device: str):
        """Puts model to device.

        Args:
            device (str): torch device name
        """
        if device != self.device:
            self.device = device
        self.model.to(self.device)

    def train(self) -> dict:
        """Training loop for epochs. Includes early stopping

        Returns:
            dict: best losses that were achieved during training
        """
        if self.if_train:
            best_loss, best_losses, patience_counter = np.inf, dict(), 0
            pbar = tqdm(range(self.n_epochs), colour="MAGENTA")
            for epoch in pbar:
                self.epoch_num = epoch
                t0 = default_timer()
                losses = dict()
                self.model.train()
                losses.update(self.epoch('train'))
                self.model.eval()
                losses.update(self.epoch('test'))
                t1 = default_timer()

                self.current_loss = round(self.current_loss, self.order)
                if self.current_loss < best_loss and losses['loss_train'] >= losses['loss_test']:
                    best_loss = self.current_loss
                    best_losses = losses
                    self.best_state_dict = self.model.state_dict()
                    patience_counter = 0
                    self.save()
                else:
                    patience_counter += 1

                self.neptune_run['train/patience'].append(
                    1 - patience_counter / self.patience)
                self.neptune_run['train/rate'].append(t1-t0)
                self.neptune_run['train'].append(losses)
                self.neptune_run['train/best_loss'].append(best_loss)
                self.neptune_run['train/best'].append(best_losses)

                pbar.set_description(
                    f"Epoch {epoch + 1} / {self.n_epochs}, "
                    f"Current Loss: {self.current_loss:.4f}, "
                    f"Patience: {patience_counter} / {self.patience}"
                )

                if patience_counter == self.patience:
                    break

            return best_losses

    def load(self):
        """Loads model, if state dict is specifed in config.
        """
        if self.if_load:
            path = self.config['model'][self.model_name]['state_dict_path']
            print(f"Will load {self.model_name} from {path}")
            self.best_state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(self.best_state_dict)

    def save(self):
        """Saves model.
        """
        if self.if_save:
            save_model(self.config, self.best_state_dict, self.neptune_run, name=self.model_name)

    def inference(self):
        """Inference of the model.
        """
        try:
            self.model.load_state_dict(self.best_state_dict)
        except RuntimeError:
            print('WARNING: NO BEST STATE DICT')
            pass
        self.model.eval()
        self.epoch_title = f'_ep{self.epoch_num}' if self.epoch_num is not None else ''

    def epoch(self, mode: str):
        """One epoch, loops over train or test dataloaders

        Args:
            mode (str): 'train' or 'test'

        Returns:
            dict: losses of the epoch
        """
        losses = dict()
        for X, y in self.dataloaders[mode]:
            # this function should be defined for each model type separately
            losses = self.batch(X, y, losses, mode)
        losses = dict({f"{key}_{mode}": np.mean(loss)
                      for key, loss in losses.items()})
        self.current_loss = losses[self.key_metric] if mode == 'test' else None
        return losses

    def log_batch(self, batch_losses: dict, losses: dict, mode: str) -> dict:
        """Log batch into neptune and updates epoch losses

        Args:
            batch_losses (dict): Losses over batch
            losses (dict): Epoch losses
            mode (str): 'train' or 'test'

        Returns:
            dict: updateds epoch losses
        """
        for key, value in batch_losses.items():
            if key not in losses.keys():
                losses[key] = list()
            losses[key].append(value)
            if 'mape' not in key:
                self.neptune_run[f"train/batch/{key}_{mode}"].append(value)
        return losses


def parse_optimizer(config: dict, model: nn.Module) -> torch.optim.Optimizer:
    """Creates an optimizer for the model

    Args:
        config (dict): training config
        model (nn.Module): model

    Returns:
        torch.optim.Optimizer: optimizer for the model
    """
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    return optimizer
