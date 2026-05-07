"""
3D Image Augmentation for BYOL Training.

Provides augmentation transforms and utilities to convert standard 3D datasets
to BYOL format (with two augmented views).
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional, Union, Tuple
import random


class RandomCrop3D:
    """Random crop of 3D volume."""
    
    def __init__(self, output_size: Union[int, Tuple[int, int, int]]):
        """
        Args:
            output_size: Desired output size. If int, cubic crop is made.
                        If tuple, (H, W, D) crop is made.
        """
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            self.output_size = output_size
    
    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: Input volume [C, H, W, D]
        Returns:
            Cropped volume [C, H', W', D']
        """
        C, H, W, D = volume.shape
        target_h, target_w, target_d = self.output_size
        
        # If volume is smaller than target, pad it first
        if H < target_h or W < target_w or D < target_d:
            pad_h = max(0, target_h - H)
            pad_w = max(0, target_w - W)
            pad_d = max(0, target_d - D)
            volume = F.pad(volume, (0, pad_d, 0, pad_w, 0, pad_h))
            C, H, W, D = volume.shape
        
        # Random crop
        h_start = random.randint(0, H - target_h)
        w_start = random.randint(0, W - target_w)
        d_start = random.randint(0, D - target_d)
        
        return volume[
            :,
            h_start:h_start + target_h,
            w_start:w_start + target_w,
            d_start:d_start + target_d
        ]


class RandomFlip3D:
    """Random flip along spatial axes."""
    
    def __init__(self, p_h: float = 0.5, p_w: float = 0.5, p_d: float = 0.5):
        """
        Args:
            p_h: Probability of flipping along height axis
            p_w: Probability of flipping along width axis
            p_d: Probability of flipping along depth axis
        """
        self.p_h = p_h
        self.p_w = p_w
        self.p_d = p_d
    
    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: Input volume [C, H, W, D]
        Returns:
            Flipped volume [C, H, W, D]
        """
        # Flip height (axis 1)
        if random.random() < self.p_h:
            volume = torch.flip(volume, [1])
        
        # Flip width (axis 2)
        if random.random() < self.p_w:
            volume = torch.flip(volume, [2])
        
        # Flip depth (axis 3)
        if random.random() < self.p_d:
            volume = torch.flip(volume, [3])
        
        return volume


class GaussianNoise3D:
    """Add Gaussian noise to volume."""
    
    def __init__(self, mean: float = 0.0, std: float = 0.1):
        """
        Args:
            mean: Mean of Gaussian noise
            std: Standard deviation of Gaussian noise
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: Input volume [C, H, W, D]
        Returns:
            Noisy volume [C, H, W, D]
        """
        noise = torch.randn_like(volume) * self.std + self.mean
        return volume + noise


class RandomIntensityScale3D:
    """Randomly scale intensity values."""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.9, 1.1)):
        """
        Args:
            scale_range: Range of scaling factors (min, max)
        """
        self.scale_range = scale_range
    
    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: Input volume [C, H, W, D]
        Returns:
            Scaled volume [C, H, W, D]
        """
        scale = random.uniform(*self.scale_range)
        return volume * scale


class RandomIntensityShift3D:
    """Randomly shift intensity values."""
    
    def __init__(self, shift_range: Tuple[float, float] = (-0.1, 0.1)):
        """
        Args:
            shift_range: Range of shift values (min, max)
        """
        self.shift_range = shift_range
    
    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: Input volume [C, H, W, D]
        Returns:
            Shifted volume [C, H, W, D]
        """
        shift = random.uniform(*self.shift_range)
        return volume + shift


class ClampValues:
    """Clamp values to a specified range."""
    
    def __init__(self, min_val: float = -1.0, max_val: float = 1.0):
        """
        Args:
            min_val: Minimum value
            max_val: Maximum value
        """
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: Input volume
        Returns:
            Clamped volume
        """
        return torch.clamp(volume, self.min_val, self.max_val)


class RandomRotation90_3D:
    """Random 90-degree rotation in 3D space."""
    
    def __init__(self, axes: Tuple[int, int] = (2, 3), p: float = 0.5):
        """
        Args:
            axes: Axes to rotate around (default: width and depth)
            p: Probability of applying rotation
        """
        self.axes = axes
        self.p = p
    
    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: Input volume [C, H, W, D]
        Returns:
            Rotated volume [C, H, W, D]
        """
        if random.random() < self.p:
            k = random.randint(0, 3)  # Number of 90-degree rotations
            volume = torch.rot90(volume, k, self.axes)
        return volume


class Compose:
    """Compose multiple transforms together."""
    
    def __init__(self, transforms: list):
        """
        Args:
            transforms: List of transform functions
        """
        self.transforms = transforms
    
    def __call__(self, volume: torch.Tensor) -> torch.Tensor:
        """
        Args:
            volume: Input volume
        Returns:
            Transformed volume
        """
        for t in self.transforms:
            volume = t(volume)
        return volume


def get_byol_augmentation(
    crop_size: Optional[Union[int, Tuple[int, int, int]]] = None,
    flip_prob: float = 0.5,
    noise_std: float = 0.1,
    intensity_scale_range: Tuple[float, float] = (0.9, 1.1),
    intensity_shift_range: Tuple[float, float] = (-0.1, 0.1),
    rotation_prob: float = 0.3,
    clamp_range: Tuple[float, float] = (-1.0, 1.0)
) -> Compose:
    """
    Get the default BYOL augmentation pipeline for 3D volumes.
    
    This pipeline includes:
    1. Random cropping (if crop_size specified)
    2. Random flipping along all axes
    3. Random 90-degree rotation
    4. Gaussian noise
    5. Intensity scaling
    6. Intensity shifting
    7. Value clamping
    
    Args:
        crop_size: Size for random cropping. If None, no cropping is applied.
        flip_prob: Probability of flipping along each axis
        noise_std: Standard deviation of Gaussian noise
        intensity_scale_range: Range for intensity scaling
        intensity_shift_range: Range for intensity shifting
        rotation_prob: Probability of random 90-degree rotation
        clamp_range: Range to clamp values to
    
    Returns:
        Composed augmentation pipeline
    """
    transforms = []
    
    # Random crop
    if crop_size is not None:
        transforms.append(RandomCrop3D(crop_size))
    
    # Random flip
    transforms.append(RandomFlip3D(p_h=flip_prob, p_w=flip_prob, p_d=flip_prob))
    
    # Random rotation
    transforms.append(RandomRotation90_3D(p=rotation_prob))
    
    # Gaussian noise
    transforms.append(GaussianNoise3D(mean=0.0, std=noise_std))
    
    # Intensity transformations
    transforms.append(RandomIntensityScale3D(scale_range=intensity_scale_range))
    transforms.append(RandomIntensityShift3D(shift_range=intensity_shift_range))
    
    # Clamp values
    transforms.append(ClampValues(min_val=clamp_range[0], max_val=clamp_range[1]))
    
    return Compose(transforms)


# Example augmentation configurations

def get_light_augmentation(crop_size: Optional[Union[int, Tuple[int, int, int]]] = None):
    """Get light augmentation (minimal transformations)."""
    return get_byol_augmentation(
        crop_size=crop_size,
        flip_prob=0.5,
        noise_std=0.05,
        intensity_scale_range=(0.95, 1.05),
        intensity_shift_range=(-0.05, 0.05),
        rotation_prob=0.0,
        clamp_range=(-1.0, 1.0)
    )


def get_medium_augmentation(crop_size: Optional[Union[int, Tuple[int, int, int]]] = None):
    """Get medium augmentation (balanced transformations)."""
    return get_byol_augmentation(
        crop_size=crop_size,
        flip_prob=0.5,
        noise_std=0.1,
        intensity_scale_range=(0.9, 1.1),
        intensity_shift_range=(-0.1, 0.1),
        rotation_prob=0.3,
        clamp_range=(-1.0, 1.0)
    )


def get_strong_augmentation(crop_size: Optional[Union[int, Tuple[int, int, int]]] = None):
    """Get strong augmentation (aggressive transformations)."""
    return get_byol_augmentation(
        crop_size=crop_size,
        flip_prob=0.7,
        noise_std=0.15,
        intensity_scale_range=(0.8, 1.2),
        intensity_shift_range=(-0.15, 0.15),
        rotation_prob=0.5,
        clamp_range=(-1.0, 1.0)
    )


# ============================================================================
# 2D Differentiable Augmentations for MoCo (after 3D-to-2D encoder)
# ============================================================================

class RandomResizedCrop2D:
    """Differentiable random resized crop for 2D images."""
    
    def __init__(self, size: Union[int, Tuple[int, int]], scale: Tuple[float, float] = (0.08, 1.0)):
        """
        Args:
            size: Target output size
            scale: Range of size of the origin size cropped
        """
        self.size = (size, size) if isinstance(size, int) else size
        self.scale = scale
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Input image [B, C, H, W] or [C, H, W]
        Returns:
            Cropped and resized image
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        B, C, H, W = img.shape
        area = H * W
        
        # Random scale
        target_area = area * random.uniform(self.scale[0], self.scale[1])
        aspect_ratio = random.uniform(3/4, 4/3)
        
        h = int(round((target_area * aspect_ratio) ** 0.5))
        w = int(round((target_area / aspect_ratio) ** 0.5))
        
        # Ensure crop fits
        if h > H or w > W:
            h, w = H, W
        
        # Random position
        i = random.randint(0, H - h) if H > h else 0
        j = random.randint(0, W - w) if W > w else 0
        
        # Crop
        img = img[:, :, i:i+h, j:j+w]
        
        # Resize
        img = F.interpolate(img, size=self.size, mode='bilinear', align_corners=False)
        
        return img.squeeze(0) if squeeze else img


class ColorJitter2D:
    """Differentiable color jitter for 2D images."""
    
    def __init__(self, brightness: float = 0.4, contrast: float = 0.4, 
                 saturation: float = 0.2, hue: float = 0.1):
        """
        Args:
            brightness: How much to jitter brightness
            contrast: How much to jitter contrast
            saturation: How much to jitter saturation
            hue: How much to jitter hue
        """
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Input image [B, C, H, W] or [C, H, W]
        Returns:
            Jittered image
        """
        # Brightness
        if self.brightness > 0:
            brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            img = img * brightness_factor
        
        # Contrast
        if self.contrast > 0:
            contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            mean = img.mean(dim=(-2, -1), keepdim=True)
            img = (img - mean) * contrast_factor + mean
        
        # Saturation (for RGB images)
        if self.saturation > 0 and img.shape[-3] == 3:
            saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            gray = img.mean(dim=-3, keepdim=True)
            img = gray + (img - gray) * saturation_factor
        
        return img


class RandomGrayscale2D:
    """Differentiable random grayscale for 2D images."""
    
    def __init__(self, p: float = 0.2):
        """
        Args:
            p: Probability of converting to grayscale
        """
        self.p = p
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Input image [B, 3, H, W] or [3, H, W]
        Returns:
            Grayscaled image with same shape
        """
        if random.random() < self.p and img.shape[-3] == 3:
            # RGB to grayscale weights
            gray = 0.299 * img[..., 0:1, :, :] + 0.587 * img[..., 1:2, :, :] + 0.114 * img[..., 2:3, :, :]
            img = gray.repeat_interleave(3, dim=-3)
        return img


class GaussianBlur2D:
    """Differentiable Gaussian blur for 2D images."""
    
    def __init__(self, kernel_size: int = 23, sigma_range: Tuple[float, float] = (0.1, 2.0)):
        """
        Args:
            kernel_size: Size of the Gaussian kernel
            sigma_range: Range for random sigma
        """
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range
    
    def _get_gaussian_kernel(self, sigma: float, device: torch.device) -> torch.Tensor:
        """Create a 2D Gaussian kernel."""
        kernel_size = self.kernel_size
        x = torch.arange(kernel_size, dtype=torch.float32, device=device) - kernel_size // 2
        gauss = torch.exp(-x.pow(2) / (2 * sigma ** 2))
        kernel = gauss.view(1, -1) * gauss.view(-1, 1)
        kernel = kernel / kernel.sum()
        return kernel
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Input image [B, C, H, W] or [C, H, W]
        Returns:
            Blurred image
        """
        if img.dim() == 3:
            img = img.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        kernel = self._get_gaussian_kernel(sigma, img.device)
        
        B, C, H, W = img.shape
        # Expand kernel for all channels
        kernel = kernel.unsqueeze(0).unsqueeze(0).expand(C, 1, -1, -1)
        
        # Apply padding
        padding = self.kernel_size // 2
        img = F.pad(img, (padding, padding, padding, padding), mode='reflect')
        
        # Apply convolution
        img = F.conv2d(img, kernel, groups=C)
        
        return img.squeeze(0) if squeeze else img


class RandomHorizontalFlip2D:
    """Differentiable random horizontal flip for 2D images."""
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of flipping
        """
        self.p = p
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Input image [B, C, H, W] or [C, H, W]
        Returns:
            Flipped image
        """
        if random.random() < self.p:
            img = torch.flip(img, dims=[-1])
        return img


class Solarize2D:
    """Differentiable solarize for 2D images."""
    
    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Threshold for solarization
        """
        self.threshold = threshold
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Input image [B, C, H, W] or [C, H, W]
        Returns:
            Solarized image
        """
        # Invert pixels above threshold
        return torch.where(img < self.threshold, img, 1.0 - img)


class RandomApply2D:
    """Apply a transform with probability p."""
    
    def __init__(self, transforms: list, p: float = 0.5):
        """
        Args:
            transforms: List of transforms to apply
            p: Probability of applying the transforms
        """
        self.transforms = transforms
        self.p = p
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Input image
        Returns:
            Transformed image
        """
        if random.random() < self.p:
            for t in self.transforms:
                img = t(img)
        return img


class Normalize2D:
    """Differentiable normalization for 2D images."""
    
    def __init__(self, mean: list, std: list):
        """
        Args:
            mean: Mean for each channel
            std: Std for each channel
        """
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Input image [B, C, H, W] or [C, H, W]
        Returns:
            Normalized image
        """
        if img.dim() == 3:
            squeeze = True
            img = img.unsqueeze(0)
        else:
            squeeze = False
        
        self.mean = self.mean.to(img.device)
        self.std = self.std.to(img.device)
        
        img = (img - self.mean) / self.std
        
        return img.squeeze(0) if squeeze else img


class Compose2D:
    """Compose multiple 2D transforms together."""
    
    def __init__(self, transforms: list):
        """
        Args:
            transforms: List of transform functions
        """
        self.transforms = transforms
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Input image
        Returns:
            Transformed image
        """
        for t in self.transforms:
            img = t(img)
        return img


if __name__ == "__main__":
    # Test augmentation pipeline
    print("Testing 3D augmentation pipeline...")
    
    # Create a dummy 3D volume
    volume = torch.randn(3, 64, 64, 32)
    print(f"Original volume shape: {volume.shape}")
    print(f"Original volume range: [{volume.min():.3f}, {volume.max():.3f}]")
    
    # Test different augmentation levels
    aug_light = get_light_augmentation(crop_size=(48, 48, 24))
    aug_medium = get_medium_augmentation(crop_size=(48, 48, 24))
    aug_strong = get_strong_augmentation(crop_size=(48, 48, 24))
    
    # Apply augmentations
    vol_light = aug_light(volume.clone())
    vol_medium = aug_medium(volume.clone())
    vol_strong = aug_strong(volume.clone())
    
    print(f"\nLight augmentation shape: {vol_light.shape}")
    print(f"Light augmentation range: [{vol_light.min():.3f}, {vol_light.max():.3f}]")
    
    print(f"\nMedium augmentation shape: {vol_medium.shape}")
    print(f"Medium augmentation range: [{vol_medium.min():.3f}, {vol_medium.max():.3f}]")
    
    print(f"\nStrong augmentation shape: {vol_strong.shape}")
    print(f"Strong augmentation range: [{vol_strong.min():.3f}, {vol_strong.max():.3f}]")
    
    print("\n✓ Augmentation pipeline test passed!")

