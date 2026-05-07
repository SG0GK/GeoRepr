import torch

from models.cnn_3d import parse_cnn
from .train import Train, parse_optimizer
from .loss import MultiFeatureLoss


class TrainCNN(Train):

    def __init__(self, config, data, neptune_run):
        """Train pipeline for AutoEncoder model.

        Args:
            config (dict): training config
            data (Data): data class object
            neptune_run (neptune.Run): neptune run
        """
        super().__init__(config, data, neptune_run)
        model_config = config['model']
        self.model = parse_cnn(model_config)
        self.model_name = 'cnn3d'
        neptune_run['model'] = self.model.__str__()
        self.if_save = model_config['save']
        self.if_load = model_config['load']
        self.if_train = model_config['train']
        if self.if_train:
            self.optimizer = parse_optimizer(config, self.model)
        self.load()
        self.to_device(self.device)
        self.scaler = data.scaler['data']
        self.loss = MultiFeatureLoss(config['loss'])

    def batch(self, X: torch.Tensor, y: torch.Tensor, losses: dict, mode: str):
        X = X.to(self.device)
        y = y.to(self.device)
        if mode == 'train':
            y_hat = self.model(X)
            loss = self.loss(y, y_hat)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        elif mode == 'test':
            with torch.inference_mode():
                y_hat = self.model(X)
                loss = self.loss(y, y_hat)
                scores = self.loss.mape(
                    self.yscaler.unscale(y),
                    self.yscaler.unscale(y_hat)
                ).tolist()

        with torch.inference_mode():
            batch_losses = {
                'loss': loss.item()
            }
            if mode == 'test':
                batch_losses.update({
                    f'{title}_mape': value for title, value in zip(self.data.titles, scores)
                })
        losses = self.log_batch(batch_losses, losses, mode)
        return losses
