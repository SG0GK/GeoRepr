import os
import argparse

from train.cnn import TrainCNN
from utils import read_yaml
from data import DataCubesTest
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from neptune_manager import NeptuneManager


def r2(real, pred):
    mask = ~np.isnan(real) & ~np.isnan(pred)
    return r2_score(real[mask], pred[mask])


def draw_crossplot(real, pred, fig, ax, xmin=None, xmax=None, cbar=False, **kwargs):
    if xmin is None:
        xmin = np.min([np.nanmin(real), np.nanmin(pred)])
    if xmax is None:
        xmax = np.max([np.nanmax(real), np.nanmax(pred)])
    df = (xmax - xmin)
    xmin = xmin - df / 10
    xmax = xmax + df / 10
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(xmin, xmax)
    ax.plot((xmin, xmax), (xmin, xmax), 'k--', alpha=0.5, zorder=2)
    sc = ax.scatter(real, pred, **kwargs)
    if cbar:
        fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.04)
    ax.set_aspect('equal')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-p', '--project', type=str, default='GeoR/Representations')
    parser.add_argument('-n', '--name', type=str, default='inference')
    parser.add_argument('-d', '--data', type=str)
    parser.add_argument('-b', '--batch_size', type=int)
    args = parser.parse_args()

    config = read_yaml(args.config)
    exp_path = os.path.dirname(args.config)
    config['model']['load'] = True
    config['model']['save'] = False
    config['model']['train'] = False
    if args.data is not None:
        if args.data == 'small':
            config['data']['datasets'] = [
                {'data_path': '../../data/bar_tiny/config.yaml', 'n': 100},
                {'data_path': '../../data/tidal_tiny/config.yaml', 'n': 100},
                {'data_path': '../../data/shelf_tiny/config.yaml', 'n': 100},
                {'data_path': '../../data/wd_tiny/config.yaml', 'n': 100}
            ]
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size

    neptune_manager = NeptuneManager(project=args.project, config=config, name=args.name)

    print('Loading data ...')
    data = DataCubesTest(config)
    data.get_dataset(*data.load_data())
    labels = data.labels.detach().cpu().numpy()
    print(f'labels shape: {labels.shape}')
    train = TrainCNN(config=config, data=data, neptune_run=neptune_manager.run)

    y_real, y_real_scaled = list(), list()
    y_pred, y_pred_scaled = list(), list()
    representations = list()

    train.model.eval()
    print('Model:')
    print(train.model)
    a = data.yscaler.a
    b = data.yscaler.b

    with torch.inference_mode():
        print('Starting inference')
        for X, y in tqdm(train.dataloaders['test']):
            X, y = X.to(train.device), y.to(train.device)
            y_hat = train.model(X).detach()
            y_real_scaled.append(y)
            y_pred_scaled.append(y_hat)
            y_pred.append(train.yscaler.unscale(y_hat.clip(min=a, max=b)))
            y_real.append(train.yscaler.unscale(y))
            representations.append(train.model.sequence[:-1](X))

        y_real = torch.vstack(y_real)
        y_pred = torch.vstack(y_pred)
        y_real_scaled = torch.vstack(y_real_scaled)
        y_pred_scaled = torch.vstack(y_pred_scaled)
        representations = torch.vstack(representations)
        y_pred = y_pred.detach().cpu().numpy()
        y_real = y_real.detach().cpu().numpy()
        y_real_scaled = y_real_scaled.detach().cpu().numpy()
        y_pred_scaled = y_pred_scaled.detach().cpu().numpy()
        y_pred = pd.DataFrame(columns=data.titles, data=y_pred)
        y_real = pd.DataFrame(columns=data.titles, data=y_real)
        y_pred_scaled = pd.DataFrame(columns=data.titles, data=y_pred_scaled)
        y_real_scaled = pd.DataFrame(columns=data.titles, data=y_real_scaled)
        representations = representations.detach().cpu().numpy()
        np.save(os.path.join(exp_path, 'representations.npy'), representations)
        y_pred.to_csv(os.path.join(exp_path, 'y_pred.csv'))
        y_real.to_csv(os.path.join(exp_path, 'y_real.csv'))
        y_real_scaled.to_csv(os.path.join(exp_path, 'y_real_scaled.csv'))
        y_pred_scaled.to_csv(os.path.join(exp_path, 'y_pred_scaled.csv'))

    neptune_manager.finish()


if __name__ == '__main__':
    main()
