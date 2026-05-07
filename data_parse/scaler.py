import torch
import torch.nn as nn
from .cropper import Cropper, NTGCropper


class Exp(nn.Module):

    def forward(self, X):
        return torch.exp(X)


class Log(nn.Module):

    def forward(self, X):
        return torch.log(X)


class Subtract(nn.Module):

    def __init__(self, subtrahend):
        super().__init__()
        self.subtrahend = subtrahend

    def forward(self, X):
        return X - self.subtrahend


class Add(nn.Module):

    def __init__(self, term):
        super().__init__()
        self.term = term

    def forward(self, X):
        return X + self.term


class Divide(nn.Module):

    def __init__(self, divisor):
        super().__init__()
        self.divisor = divisor

    def forward(self, X):
        return X / self.divisor


class Multiply(nn.Module):

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, X):
        return X * self.factor


class Cumsum(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, X):
        return torch.cumsum(X, dim=self.dim)


class Diff(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.slices = [slice(None) for _ in range(dim)] + [slice(0, 1)]

    def forward(self, X):
        return torch.cat((X[self.slices], torch.diff(X, dim=self.dim)), dim=self.dim)


class Scaler(nn.Module):

    def __init__(self, sequence):
        super().__init__()
        self.forward_sequence = nn.Sequential()
        self.inverse_sequence = nn.Sequential()
        for sequence_item in sequence:
            iteration = sequence_item['iter'].lower()
            if iteration == 'exp':
                self.forward_sequence.append(Exp())
                self.inverse_sequence.append(Log())
            elif iteration == 'log':
                self.forward_sequence.append(Log())
                self.inverse_sequence.append(Exp())
            elif iteration == 'subtract':
                self.forward_sequence.append(
                    Subtract(subtrahend=sequence_item['number'])
                )
                self.inverse_sequence.append(
                    Add(term=sequence_item['number'])
                )
            elif iteration == 'add':
                self.forward_sequence.append(
                    Add(term=sequence_item['number'])
                )
                self.inverse_sequence.append(
                    Subtract(subtrahend=sequence_item['number'])
                )
            elif iteration == 'divide':
                self.forward_sequence.append(
                    Divide(divisor=sequence_item['number'])
                )
                self.inverse_sequence.append(
                    Multiply(factor=sequence_item['number'])
                )
            elif iteration == 'multiply':
                self.forward_sequence.append(
                    Multiply(factor=sequence_item['number'])
                )
                self.inverse_sequence.append(
                    Divide(divisor=sequence_item['number'])
                )
            elif iteration == 'diff':
                self.forward_sequence.append(Diff(dim=sequence_item['dim']))
                self.inverse_sequence.append(Cumsum(dim=sequence_item['dim']))
            elif iteration == 'cumsum':
                self.forward_sequence.append(Cumsum(dim=sequence_item['dim']))
                self.inverse_sequence.append(Diff(dim=sequence_item['dim']))
            else:
                raise ValueError(f'Unknown Scaler iter: {iteration}')
        self.inverse_sequence = self.inverse_sequence[::-1]

    def scale(self, X):
        return self.forward_sequence(X)

    def unscale(self, X):
        return self.inverse_sequence(X)


class ChannelScaler(nn.Module):

    def __init__(self, sequence_dict, total_channels=4):
        super().__init__()
        self.scalers = nn.ModuleList()
        self.channels = list()
        self.channel_masks = list()
        for channel, sequence in sequence_dict.items():
            self.scalers.append(Scaler(sequence=sequence))
            self.channels.append(channel)
            mask = [False for _ in range(total_channels)]
            mask[channel] = True
            mask = torch.tensor(mask).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            self.channel_masks.append(mask)

    def scale(self, X):
        for channel, mask, scaler in zip(self.channels, self.channel_masks, self.scalers):
            mask = mask.to(X.device)
            X = torch.where(mask, scaler.scale(X[:, channel]).unsqueeze(1), X)
        return X

    def unscale(self, X):
        for channel, mask, scaler in zip(self.channels, self.channel_masks, self.scalers):
            mask = mask.to(X.device)
            X = torch.where(mask, scaler.unscale(X[:, channel]).unsqueeze(1), X)
        return X


class ChannelScalerWithCropper(ChannelScaler):

    def __init__(self, sequence_dict, mask, total_channels=3):
        super().__init__(sequence_dict, total_channels)
        self.cropper = Cropper(mask)

    def scale(self, X):
        X = super().scale(X)
        X = self.cropper(X)
        return X

    def unscale(self, X):
        X = super().unscale(X)
        X = self.cropper(X)
        return X


class ChannelScalerWithNTGCropper(ChannelScaler):

    def __init__(self, sequence_dict, total_channels=3):
        super().__init__(sequence_dict, total_channels)
        self.cropper = NTGCropper()

    def scale(self, X):
        X = super().scale(X)
        X = self.cropper(X)
        return X

    def unscale(self, X):
        X = super().unscale(X)
        X = self.cropper(X)
        return X


# def parse_scaler(sequence_dict, total_channels´‹@, cropper=None):
#     if cropper:
        