"""
BYOL (Bootstrap Your Own Latent) model architecture for 3D images.

Based on the paper: "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning"
https://arxiv.org/pdf/2006.07733

This implementation adapts BYOL for 3D volumetric data by:
1. Using Enhanced3DTo2DEncoder to convert 3D volumes to 2D representations
2. Using ResNet50 as the main encoder
3. Training the 3D encoder together with BYOL
"""

from tkinter.constants import X
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision import transforms
import copy
import sys
import os
import timm
from torchinfo import summary

from augmentation import (
    RandomResizedCrop2D,
    ColorJitter2D,
    RandomGrayscale2D,
    GaussianBlur2D,
    RandomHorizontalFlip2D,
    Solarize2D,
    RandomApply2D,
    Normalize2D,
    Compose2D
)

# Import the 3D to 2D encoder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from encoder import Enhanced3DTo2DEncoder

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'moco-v3'))
import moco.loader


class MLP(nn.Module):
    """Multi-Layer Perceptron used for projector and predictor."""
    
    def __init__(self, input_dim, hidden_dim=4096, output_dim=256):
        """
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden layer dimension
            output_dim (int): Output dimension
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class BYOLEncoder(nn.Module):
    """3D to 2D encoder with augmentations."""
    
    def __init__(self, config, is_train=True):
        """
        Args:
            config (dict): Configuration containing encoder_3d settings
        """
        super().__init__()
        
        # 3D to 2D encoder
        self.encoder_3d_to_2d = Enhanced3DTo2DEncoder(config)

        total_params = sum(p.numel() for p in self.encoder_3d_to_2d.parameters())
        print(f"Total Encoder parameters: {total_params}")    

        self.config = config

        self.is_train = is_train

        normalize = transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)

                # Differentiable augmentation pipeline 1 (stronger blur)
        augmentation1 = [
            RandomResizedCrop2D(224, scale=(0.08, 1.)),
            RandomApply2D([
                ColorJitter2D(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),
            RandomGrayscale2D(p=0.2),
            RandomApply2D([
                GaussianBlur2D(kernel_size=23, sigma_range=(0.1, 2.0))
            ], p=1.0),
            RandomHorizontalFlip2D(p=0.5),
            Normalize2D(mean=[0.5]*3, std=[0.5]*3)
        ]

        # Differentiable augmentation pipeline 2 (weaker blur + solarize)
        augmentation2 = [
            RandomResizedCrop2D(224, scale=(0.08, 1.)),
            RandomApply2D([
                ColorJitter2D(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
            ], p=0.8),
            RandomGrayscale2D(p=0.2),
            RandomApply2D([
                GaussianBlur2D(kernel_size=23, sigma_range=(0.1, 2.0))
            ], p=0.1),
            RandomApply2D([
                Solarize2D(threshold=0.5)
            ], p=0.2),
            RandomHorizontalFlip2D(p=0.5),
            Normalize2D(mean=[0.5]*3, std=[0.5]*3)
        ]

        self.augm1 = Compose2D(augmentation1)
        self.augm2 = Compose2D(augmentation2)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input 3D volume [B, 3, H, W, D]
        Returns:
            torch.Tensor or tuple: If training, returns (view1, view2) augmented 2D images.
                                  Otherwise returns single 2D image [B, 3, H', W']
        """
        # Convert 3D to 2D: [B, 3, H, W, D] -> [B, 3, H', W']
        x_2d = self.encoder_3d_to_2d(x)

        if self.is_train:
            view1 = self.augm1(x_2d)
            view2 = self.augm2(x_2d)
            return view1, view2
        else:
            x_2d = F.interpolate(x_2d, size=(224, 224),
                              mode='bilinear', align_corners=False)
            return x_2d


class BYOLOnlineNetwork(nn.Module):
    """Online network with 3D encoder, 2D encoder, projector, and predictor."""
    
    def __init__(self, config, projection_dim=256, hidden_dim=4096, freeze_2d_encoder=True, is_train=True, fine_tune=True):
        """
        Args:
            config (dict): Configuration for encoder
            projection_dim (int): Output dimension of projector
            hidden_dim (int): Hidden dimension for MLP layers
            freeze_2d_encoder (bool): Whether to freeze the pretrained 2D encoder
            is_train (bool): Whether to train the model
            fine_tune (bool): Whether to fine-tune the 2D encoder
        """
        super().__init__()
        
        # 3D to 2D encoder
        self.encoder_3d = BYOLEncoder(config, is_train)
        
        # 2D encoder (ResNet50 pretrained)
        self.encoder_2d = timm.create_model("hf_hub:1aurent/resnet50.medmae_in1k_byol", pretrained=freeze_2d_encoder)

        total_params = sum(p.numel() for p in self.encoder_2d.parameters())
        print(f"Total Encoder_2D parameters: {total_params}") 
        
        if not fine_tune:
            for param in self.encoder_2d.parameters():
                param.requires_grad = False
            self.encoder_2d.eval()
        
        self.projector = MLP(
            input_dim=2048,
            hidden_dim=hidden_dim,
            output_dim=projection_dim
        )
        self.predictor = MLP(
            input_dim=projection_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim
        )
    
    def forward(self, x, return_projection=False):
        """
        Args:
            x (torch.Tensor): Input 2D image [B, 3, H, W] (already augmented)
            return_projection (bool): If True, return (projection, prediction)
                                     If False, return only prediction
        Returns:
            torch.Tensor or tuple: Prediction or (projection, prediction)
        """
        # Encode with 2D encoder
        features = self.encoder_2d(x)
        
        # Project
        projection = self.projector(features)
        
        # Predict
        prediction = self.predictor(projection)
        
        if return_projection:
            return projection, prediction
        return prediction


class BYOLTargetNetwork(nn.Module):
    """Target network with 3D encoder, 2D encoder, and projector (no predictor)."""
    
    def __init__(self, config, projection_dim=256, hidden_dim=4096, freeze_2d_encoder=True, is_train=True, fine_tune=True):
        """
        Args:
            config (dict): Configuration for encoder
            projection_dim (int): Output dimension of projector
            hidden_dim (int): Hidden dimension for MLP layers
            freeze_2d_encoder (bool): Whether to freeze the pretrained 2D encoder
            is_train (bool): Whether to train the model
            fine_tune (bool): Whether to fine-tune the 2D encoder
        """
        super().__init__()
        
        # 3D to 2D encoder
        self.encoder_3d = BYOLEncoder(config, is_train)
        
        # 2D encoder (ResNet50 pretrained)
        self.encoder_2d = timm.create_model("hf_hub:1aurent/resnet50.medmae_in1k_byol", pretrained=freeze_2d_encoder)
        
        if not fine_tune:
            for param in self.encoder_2d.parameters():
                param.requires_grad = False
            self.encoder_2d.eval()
        
        self.projector = MLP(
            input_dim=2048,
            hidden_dim=hidden_dim,
            output_dim=projection_dim
        )
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input 2D image [B, 3, H, W] (already augmented)
        Returns:
            torch.Tensor: Projected features [B, projection_dim]
        """
        # Encode with 2D encoder
        features = self.encoder_2d(x)
        
        # Project
        projection = self.projector(features)
        
        return projection


class BYOL(nn.Module):
    """
    Complete BYOL model with online and target networks.
    
    The model maintains two networks:
    - Online network: Updated via gradient descent
    - Target network: Updated via exponential moving average (EMA) of online network
    """
    
    def __init__(self, config, projection_dim=256, hidden_dim=4096, 
                 moving_average_decay=0.996, freeze_2d_encoder=True, is_train=True, fine_tune=True):
        """
        Args:
            config (dict): Configuration for encoder
            projection_dim (int): Dimension of projection space
            hidden_dim (int): Hidden dimension for MLP layers
            moving_average_decay (float): EMA decay rate for target network (tau)
            freeze_2d_encoder (bool): Whether to freeze the pretrained 2D encoder
            is_train (bool): Whether to train the model
            fine_tune (bool): Whether to fine-tune the 2D encoder
        """
        super().__init__()
        
        self.online_network = BYOLOnlineNetwork(
            config, projection_dim, hidden_dim, freeze_2d_encoder, is_train, fine_tune
        )
        self.target_network = BYOLTargetNetwork(
            config, projection_dim, hidden_dim, freeze_2d_encoder, is_train, fine_tune
        )
        
        # Initialize target network with online network parameters
        self._initialize_target_network()
        
        # Target network parameters are not updated via gradient descent
        for param in self.target_network.parameters():
            param.requires_grad = False
        
        self.moving_average_decay = moving_average_decay
    
    def _initialize_target_network(self):
        """Initialize target network with online network parameters."""
        # Copy 3D encoder parameters
        for online_params, target_params in zip(
            self.online_network.encoder_3d.parameters(),
            self.target_network.encoder_3d.parameters()
        ):
            target_params.data.copy_(online_params.data)
        
        # Copy 2D encoder parameters
        for online_params, target_params in zip(
            self.online_network.encoder_2d.parameters(),
            self.target_network.encoder_2d.parameters()
        ):
            target_params.data.copy_(online_params.data)
        
        # Copy projector parameters
        for online_params, target_params in zip(
            self.online_network.projector.parameters(),
            self.target_network.projector.parameters()
        ):
            target_params.data.copy_(online_params.data)
    
    @torch.no_grad()
    def update_target_network(self, momentum=None):
        """
        Update target network parameters using exponential moving average.
        
        target_params = momentum * target_params + (1 - momentum) * online_params
        
        Args:
            momentum (float, optional): If provided, use this momentum value.
                                       Otherwise use self.moving_average_decay
        """
        if momentum is None:
            momentum = self.moving_average_decay
        
        # Update 3D encoder
        for online_params, target_params in zip(
            self.online_network.encoder_3d.parameters(),
            self.target_network.encoder_3d.parameters()
        ):
            target_params.data = (
                momentum * target_params.data + 
                (1 - momentum) * online_params.data
            )
        
        # Update 2D encoder
        for online_params, target_params in zip(
            self.online_network.encoder_2d.parameters(),
            self.target_network.encoder_2d.parameters()
        ):
            target_params.data = (
                momentum * target_params.data + 
                (1 - momentum) * online_params.data
            )
        
        # Update projector
        for online_params, target_params in zip(
            self.online_network.projector.parameters(),
            self.target_network.projector.parameters()
        ):
            target_params.data = (
                momentum * target_params.data + 
                (1 - momentum) * online_params.data
            )
    
    def forward(self, x):
        """
        Forward pass for BYOL training.
        
        Args:
            x (torch.Tensor): Input 3D volume [B, 3, H, W, D]
        
        Returns:
            dict: Dictionary containing:
                - 'pred1': Online network prediction from view1
                - 'pred2': Online network prediction from view2
                - 'proj1': Target network projection from view1
                - 'proj2': Target network projection from view2
        """
        # Get two augmented views from 3D encoder
        view1, view2 = self.online_network.encoder_3d(x)
        
        # Online network predictions
        pred1 = self.online_network(view1)
        pred2 = self.online_network(view2)
        
        # Target network projections (no gradients)
        with torch.no_grad():
            # Use target network's 3D encoder for consistency
            view1_target, view2_target = self.target_network.encoder_3d(x)
            
            proj1 = self.target_network(view1_target)
            proj2 = self.target_network(view2_target)
        
        return {
            'pred1': pred1,
            'pred2': pred2,
            'proj1': proj1,
            'proj2': proj2
        }

    @torch.no_grad()
    def get_representations(
        self,
        x,
        use_target=False,
        from_projector=False,
        normalize=True
    ):
        """
        Extract representations for downstream tasks.
        
        Args:
            x (torch.Tensor): Input 3D volume [B, 3, H, W, D]
            use_target (bool): If True, use the EMA target network; otherwise use online.
            from_projector (bool): If True, return projector output (projection space).
            normalize (bool): If True, apply L2 normalization to the output.
        
        Returns:
            torch.Tensor: Representations [B, D] where D is encoder or projector dim.
        """
        network = self.target_network if use_target else self.online_network
        
        # Convert 3D to 2D
        x_2d = network.encoder_3d(x)
        
        # Encode with 2D encoder
        reps = network.encoder_2d(x_2d)
        
        # if from_projector:
        #     reps = network.projector(reps)
        
        # if normalize:
        #     reps = F.normalize(reps, dim=-1, p=2)
        
        return reps


class Timm_BYOL(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = Enhanced3DTo2DEncoder(config)

        self.model = timm.create_model("hf_hub:1aurent/resnet50.medmae_in1k_byol", pretrained=True)

        self.projector = nn.Linear(2048, 384)

        for params in self.model.parameters():
            params.requires_grad = False

        self.model.eval()

        summary(self.model, input_size=(1, 3, 224, 224))

    def forward(self, view1, view2):
        enc1 = self.encoder(view1)
        enc2 = self.encoder(view2)

        enc1 = self.model(enc1)
        enc2 = self.model(enc2)

        

        return enc1, enc2

    @torch.no_grad()
    def get_representations(self, X):
        enc = self.encoder(enc)





def byol_loss_fn(pred, proj):
    """
    BYOL loss function: normalized mean squared error.
    
    Loss = 2 - 2 * <pred_norm, proj_norm>
         = 2 - 2 * cosine_similarity(pred, proj)
    
    Args:
        pred (torch.Tensor): Prediction from online network [B, D]
        proj (torch.Tensor): Projection from target network [B, D]
    
    Returns:
        torch.Tensor: Scalar loss value
    """
    # Normalize
    pred = F.normalize(pred, dim=-1, p=2)
    proj = F.normalize(proj, dim=-1, p=2)
    
    # Compute loss: 2 - 2 * cosine_similarity
    loss = 2 - 2 * (pred * proj).sum(dim=-1)
    
    return loss.mean()


def compute_byol_loss(outputs):
    """
    Compute symmetrized BYOL loss.
    
    Loss = loss(pred1, proj2) + loss(pred2, proj1)
    
    Args:
        outputs (dict): Output dictionary from BYOL.forward()
    
    Returns:
        torch.Tensor: Scalar loss value
    """
    # First direction: predict view2 from view1
    loss1 = byol_loss_fn(outputs['pred1'], outputs['proj2'])
    
    # Second direction: predict view1 from view2
    loss2 = byol_loss_fn(outputs['pred2'], outputs['proj1'])
    
    # Symmetrized loss
    total_loss = loss1 + loss2
    
    return total_loss


if __name__ == "__main__":
    # Test the model
    # print("Testing BYOL model...")
    
    # # Create a dummy config
    config = {
        'encoder_3d': {
            'in_channels': 3,
            'block_out_channels': [16, 32, 64],
            'block_out_types': ['down', 'down', 'same'],
            'num_groups': 16,
            'dropout': 0.01,
            'scales_w': [2, 2],
            'scales_h': [2, 2],
            'scales_d': [2, 2]
        }
    }
    
    # # Create model
    # model = BYOL(config, projection_dim=256, hidden_dim=4096)
    
    # # Create dummy input (two augmented views)
    # batch_size = 4
    # view1 = torch.randn(batch_size, 3, 64, 64, 32)  # [B, 3, H, W, D]
    # view2 = torch.randn(batch_size, 3, 64, 64, 32)
    
    # # Forward pass
    # outputs = model(view1, view2)
    
    # # Compute loss
    # loss = compute_byol_loss(outputs)
    
    # print(f"Input shape: {view1.shape}")
    # print(f"Prediction shape: {outputs['pred1'].shape}")
    # print(f"Projection shape: {outputs['proj1'].shape}")
    # print(f"Loss: {loss.item():.4f}")
    # print("\n✓ Model test passed!")

    model = Timm_BYOL(config=config)
