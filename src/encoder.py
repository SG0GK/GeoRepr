import torch
import torch.nn as nn
import torch.nn.functional as F


class Padder3D(object):

    def __init__(self, scale=8):
        """A tool for padding torch tensors

        Args:
            scale (int, optional): A number, for which torch shape must be 
                divisible of. Defaults to 8.
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
            self.padding_height = (height // self.scale * self.scale + 
                                   self.scale - height)
        if width % self.scale:
            self.padding_width = (width // self.scale * self.scale + 
                                  self.scale - width)
        if depth % self.scale:
            self.padding_width = (depth // self.scale * self.scale + 
                                  self.scale - depth)
        return F.pad(X, (0, self.padding_depth, 0, self.padding_width, 
                        0, self.padding_height), 'constant', 0)

    def unpad(self, X: torch.Tensor) -> torch.Tensor:
        """Unpads tensor to its original shape.
        The original shape is stored as shape of last tensor that was padded.

        Args:
            X (torch.Tensor): padded tensor

        Returns:
            torch.Tensor: tensor of original shape (with smaller or equal shape)
        """
        unpadding_depth = (-self.padding_depth if self.padding_depth > 0 
                          else X.shape[-1])
        unpadding_width = (-self.padding_width if self.padding_width > 0 
                          else X.shape[-2])
        unpadding_height = (-self.padding_height if self.padding_height > 0 
                           else X.shape[-3])
        return X[..., :unpadding_height, :unpadding_width, :unpadding_depth]


class ResNetBaseBlock3D(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv_shortcut = nn.Sequential()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Does 2 convolutions and skip connections

        Args:
            X (torch.Tensor): input data [N, C, W, H, D]

        Returns:
            torch.Tensor: output data [N, C, W, H, D]
        """
        hidden_states = self.conv(X)
        hidden_states = hidden_states + self.conv_shortcut(X)
        return hidden_states


class ResNetDownBlock3D(ResNetBaseBlock3D):

    def __init__(
        self, in_channels=32, out_channels=32, num_groups=2, dropout=0.01,
        scale_d=2, scale_w=2, scale_h=2
    ):
        """ResNet 3D that decreases spatial dimensions by 2.

        Args:
            in_channels (int, optional): input channels. Defaults to 32.
            out_channels (int, optional): output channels. Defaults to 32.
            num_groups (int, optional): number of groups for groupnorm. Defaults to 32.
            dropout (float, optional): dropout probability. Defaults to 0.01.
            scale_d (int, optional): how many times to decrease depth. Defaults to 2.
            scale_w (int, optional): how many times to decrease width. Defaults to 2.
            scale_h (int, optional): how many times to decrease height. Defaults to 2.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(scale_w, scale_h, scale_d), stride=(scale_w, scale_h, scale_d)),
            nn.GroupNorm(num_channels=in_channels, num_groups=num_groups),
            nn.SiLU(),
            nn.Conv3d(
                in_channels, out_channels,
                kernel_size=3, stride=1, padding=1
            ),
            nn.GroupNorm(num_channels=out_channels, num_groups=num_groups),
            nn.SiLU(),
            nn.Dropout3d(p=dropout),
            nn.Conv3d(
                out_channels, out_channels,
                kernel_size=3, stride=1, padding=1
            )
        )

        self.conv_shortcut = nn.Sequential(
            nn.MaxPool3d(kernel_size=(scale_w, scale_h, scale_d), stride=(scale_w, scale_h, scale_d)),
        )

        if in_channels != out_channels:
            self.conv_shortcut.append(nn.Conv3d(
                in_channels, out_channels,
                kernel_size=1, stride=1, padding=0
            ))


class ResNetBlock3D(ResNetBaseBlock3D):

    def __init__(self, in_channels=32, out_channels=32, num_groups=32, dropout=0.01):
        """ResNet 3D that keeps all spatial dimensions.

        Args:
            in_channels (int, optional): input channels. Defaults to 32.
            out_channels (int, optional): output channels. Defaults to 32.
            num_groups (int, optional): number of groups for groupnorm. Defaults to 32.
            dropout (float, optional): dropout probability. Defaults to 0.01.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Sequential(
            nn.GroupNorm(num_channels=in_channels, num_groups=num_groups),
            nn.SiLU(),
            nn.Conv3d(
                in_channels, out_channels,
                kernel_size=3, stride=1, padding=1
            ),
            nn.GroupNorm(num_channels=out_channels, num_groups=num_groups),
            nn.SiLU(),
            nn.Dropout3d(p=dropout),
            nn.Conv3d(
                out_channels, out_channels,
                kernel_size=3, stride=1, padding=1
            )
        )

        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv3d(
                in_channels, out_channels,
                kernel_size=1, stride=1, padding=0
            )
        else:
            self.conv_shortcut = nn.Identity()


class ResNetBase(nn.Module):

    def __init__(self, in_channels=3, out_channels=3, block_out_channels=[3, 3]):
        """Resnet placeholder, creates input and output convolutions

        Args:
            in_channels (int, optional): input channels. Defaults to 3.
            out_channels (int, optional): output channels. Defaults to 3.
            block_out_channels (list, optional): channels of intermediate representations. Defaults to [3, 3].
        """
        super().__init__()

        if in_channels != block_out_channels[0]:
            self.conv_in = nn.Conv3d(
                in_channels=in_channels, out_channels=block_out_channels[0],
                kernel_size=1, padding=0
            )
        else:
            self.conv_in = nn.Identity()

        if out_channels != block_out_channels[-1]:
            self.conv_out = nn.Conv3d(
                in_channels=block_out_channels[-1], out_channels=out_channels,
                kernel_size=1, padding=0
            )
        else:
            self.conv_out = nn.Identity()

        self.conv = nn.Sequential()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Full resnet pipeline, with conv_in layer, then intermediate layers
        and then output layers

        Args:
            X (torch.Tensor): input data

        Returns:
            torch.Tensor: output data
        """
        X = self.conv_in(X)
        X = self.conv(X)
        X = self.conv_out(X)
        return X





class Encoder3D(nn.Module):
    """3D Encoder that takes 3-channel 3D data and outputs 1-channel 3D data."""
    
    def __init__(self, in_channels=3, out_channels=1, reduction_factor=4):
        """
        Args:
            in_channels (int): Number of input channels (default: 3)
            out_channels (int): Number of output channels (default: 1)
            reduction_factor (int): Factor to reduce spatial dimensions
        """
        super().__init__()
        self.reduction_factor = reduction_factor
        self.padder = Padder3D(scale=reduction_factor)
        
        # Progressive downsampling encoder
        self.encoder = nn.Sequential(
            # First block: 3 -> 16 channels, reduce by 2
            ResNetDownBlock3D(
                in_channels=in_channels, 
                out_channels=16,
                num_groups=min(16, in_channels),
                scale_d=2, scale_w=2, scale_h=2
            ),
            # Second block: 16 -> 32 channels, reduce by 2
            ResNetDownBlock3D(
                in_channels=16, 
                out_channels=32,
                num_groups=16,
                scale_d=2, scale_w=2, scale_h=2
            ),
            # Third block: 32 -> 64 channels, keep spatial dimensions
            ResNetBlock3D(
                in_channels=32, 
                out_channels=64,
                num_groups=32
            ),
            # Output projection: 64 -> 1 channel
            nn.Conv3d(64, out_channels, kernel_size=1, padding=0),
            nn.Tanh()  # Normalize output to [-1, 1]
        )
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor [B, C, H, W, D]
        Returns:
            torch.Tensor: Output tensor [B, 1, H//4, W//4, D//4]
        """
        # Pad input to ensure divisibility
        x = self.padder.pad(x)
        
        # Apply encoder
        x = self.encoder(x)
        
        # Return encoded tensor
        return x
    
    
class SimpleEncoder3D(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=3,
        block_out_channels=[16, 32], block_out_types=['down', 'same'],
        num_groups=16, dropout=0.01,
        scales_w=[2], scales_h=[2], scales_d=[2],
    ):
        """ResNet-based encoder

        Args:
            in_channels (int, optional): input channels. Defaults to 3.
            out_channels (int, optional): output channels. Defaults to 3.
            block_out_channels (list, optional): channels of intermediate representations.
                Defaults to [16, 32].
            block_out_types (list, optional): type of resnets, down - ResNetDownBlock3D, same - ResNetBlock3D.
                Defaults to ['down', 'same'].
            num_groups (int, optional): number of groups for groupnorm. Defaults to 16.
            dropout (float, optional): resnet dropout. Defaults to 0.01.
            scales_w (list, optional): how many times to decrease width on each down layer. Defaults to [2].
            scales_h (list, optional): how many times to decrease height on each down layer. Defaults to [2].
            scales_d (list, optional): how many times to decrease depth on each down layer. Defaults to [2].

        Raises:
            ValueError: in case when block_out_type is unknown.
        """
        super().__init__()

        if in_channels != block_out_channels[0]:
            conv_in = nn.Conv3d(
                in_channels=in_channels, out_channels=block_out_channels[0],
                kernel_size=1, padding=0
            )
        else:
            conv_in = nn.Identity()

        self.conv = nn.Sequential(conv_in)

        counts_decrease = 0

        for i in range(len(block_out_channels) - 1):
            self.conv.append(ResNetBlock3D(
                in_channels=block_out_channels[i],
                out_channels=block_out_channels[i+1],
                num_groups=num_groups, dropout=dropout
            ))
            if block_out_types[i] == 'down':
                self.conv.append(
                    nn.MaxPool3d(
                        kernel_size=(scales_w[counts_decrease], scales_h[counts_decrease], scales_d[counts_decrease]),
                        stride=(scales_w[counts_decrease], scales_h[counts_decrease], scales_d[counts_decrease])
                    )
                )
                counts_decrease = counts_decrease + 1
            elif block_out_types[i] == 'same':
                pass
            else:
                raise ValueError(f'Unknown Encoder block: {block_out_types[i]}')

        if out_channels != block_out_channels[-1]:
            conv_out = nn.Conv3d(
                in_channels=block_out_channels[-1], out_channels=out_channels,
                kernel_size=1, padding=0
            )
        else:
            conv_out = nn.Identity()

        self.conv.append(conv_out)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Full resnet pipeline, with conv_in layer, then intermediate layers
        and then output layers

        Args:
            X (torch.Tensor): input data

        Returns:
            torch.Tensor: output data
        """
        X = self.conv(X)
        return X

