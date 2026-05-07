import torch
import torch.nn.functional as F


class Padder3D(object):

    def __init__(self, scale=8):
        """A tool for padding torch tensors

        Args:
            scale (int, optional): A number, for which torch shape must be divisible of. Defaults to 8.
        """
        self.scale = scale
        self.padding_height, self.padding_width, self.padding_depth = 0, 0, 0

    def pad(self, X: torch.Tensor) -> torch.Tensor:
        """Pads tensor and remember its original shape

        Args:
            X (torch.Tensor): original tensor

        Returns:
            torch.Tensor: padded tensor (with larger or equal shape)
        """
        height, width, depth = X.shape[-3], X.shape[-2], X.shape[-2]
        if height % self.scale:
            self.padding_height = height // self.scale * self.scale + self.scale - height
        if width % self.scale:
            self.padding_width = width // self.scale * self.scale + self.scale - width
        if depth % self.scale:
            self.padding_width = depth // self.scale * self.scale + self.scale - depth
        return F.pad(X, (0, self.padding_depth, 0, self.padding_width, 0, self.padding_height), 'constant', 0)

    def unpad(self, X: torch.Tensor) -> torch.Tensor:
        """Unpads tensor to its original shape.
        The original shape is stored as shape of last tensor that was padded.

        Args:
            X (torch.Tensor): padded tensor

        Returns:
            torch.Tensor: tensor of original shape (with smaller or equal shape)
        """
        unpadding_depth = -self.padding_depth if self.padding_depth > 0 else X.shape[-1]
        unpadding_width = -self.padding_width if self.padding_width > 0 else X.shape[-2]
        unpadding_height = -self.padding_height if self.padding_height > 0 else X.shape[-3]
        return X[..., :unpadding_height, :unpadding_width, :unpadding_depth]
