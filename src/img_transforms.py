import torch
import torch.nn.functional as F
import random


class TensorTransform:
    '''
    Used for DINOv2 supervised learning
    '''
    def __init__(self, config):
        self.target_size = (config['data']['transforms']['target_size'], 
                          config['data']['transforms']['target_size'])
        num_channels = config['data']['input_channels']
        self.mean = torch.tensor(config['data']['transforms']['mean'] if config['data']['transforms']['mean'] is not None 
                               else [0.5] * num_channels).view(-1, 1, 1)
        self.std = torch.tensor(config['data']['transforms']['std'] if config['data']['transforms']['std'] is not None 
                              else [0.5] * num_channels).view(-1, 1, 1)

    def __call__(self, x):
        x = F.interpolate(x.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False)
        x = x.squeeze(0)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        return x


class MultiCropTensorTransform:
    '''
    Used for DINOv2 self-supervised learning
    '''
    def __init__(self, config):
        transforms_config = config['data']['transforms']
        self.global_size = transforms_config['global_crop_size']
        self.local_size = transforms_config['local_crop_size']
        num_channels = config['data']['input_channels']
        self.mean = torch.tensor(transforms_config['mean'] if transforms_config['mean'] is not None 
                               else [0.5] * num_channels).view(-1, 1, 1)
        self.std = torch.tensor(transforms_config['std'] if transforms_config['std'] is not None 
                              else [0.5] * num_channels).view(-1, 1, 1)
        
        # Augmentation parameters
        aug_config = transforms_config['augmentation']
        self.brightness = aug_config['color_jitter']['brightness']
        self.contrast = aug_config['color_jitter']['contrast']
        self.scale_range = aug_config['random_crop']['scale_range']
        self.aspect_ratio_range = aug_config['random_crop']['aspect_ratio_range']
        self.blur_kernel_size = aug_config['gaussian_blur']['kernel_size']
        self.flip_prob = aug_config['flip_prob']

    def random_resized_crop(self, img, size):
        C, H, W = img.shape
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        target_area = scale * H * W
        aspect_ratio = random.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1])
        new_w = int(round((target_area * aspect_ratio) ** 0.5))
        new_h = int(round((target_area / aspect_ratio) ** 0.5))

        if new_w > W or new_h > H:
            new_w = min(new_w, W)
            new_h = min(new_h, H)

        top = random.randint(0, H - new_h)
        left = random.randint(0, W - new_w)
        cropped = img[:, top:top + new_h, left:left + new_w]
        resized = F.interpolate(cropped.unsqueeze(0), size=(size, size), mode='bilinear', align_corners=False).squeeze(0)
        return resized

    def random_horizontal_flip(self, img):
        return img.flip(-1) if random.random() < self.flip_prob else img

    def gaussian_blur(self, img):
        padding = self.blur_kernel_size // 2
        channels = img.shape[0]
        kernel = torch.tensor([1., 2., 1.], dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :]
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, self.blur_kernel_size, self.blur_kernel_size).repeat(channels, 1, 1, 1)

        img = img.unsqueeze(0)
        blurred = F.conv2d(img, kernel.to(img.device), padding=padding, groups=channels)
        return blurred.squeeze(0)

    def color_jitter(self, img):
        # Apply brightness and contrast jitter
        if self.brightness > 0:
            factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            img = img * factor
        if self.contrast > 0:
            mean = img.mean(dim=(1, 2), keepdim=True)
            factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            img = (img - mean) * factor + mean
        return img

    def normalize(self, img):
        return (img - self.mean.to(img.device)) / self.std.to(img.device)

    def transform(self, img, size):
        img = self.random_resized_crop(img, size)
        img = self.random_horizontal_flip(img)
        img = self.color_jitter(img)
        img = self.gaussian_blur(img)
        img = self.normalize(img)
        return img

    def __call__(self, img):
        return [
            self.transform(img, self.global_size),
            self.transform(img, self.global_size),
            self.transform(img, self.local_size),
            self.transform(img, self.local_size),
        ]


class ThreeD_to_TwoD_Transform:
    """Transform that converts 3D geological data to 2D using SimpleEncoder3D, then applies 2D transforms."""
    
    def __init__(self, config):
        from encoder import SimpleEncoder3D
        
        # Initialize the 3D encoder
        encoder_config = config.get('encoder_3d', {})
        self.encoder_3d = SimpleEncoder3D(
            in_channels=encoder_config.get('in_channels', 3),
            out_channels=encoder_config.get('out_channels', 1), 
            block_out_channels=encoder_config.get('block_out_channels', [16, 32, 64]),
            block_out_types=encoder_config.get('block_out_types', ['down', 'down', 'same']),
            num_groups=encoder_config.get('num_groups', 16),
            dropout=encoder_config.get('dropout', 0.01),
            scales_w=encoder_config.get('scales_w', [2, 2]),
            scales_h=encoder_config.get('scales_h', [2, 2]),
            scales_d=encoder_config.get('scales_d', [2, 2])
        )
        
        # Set encoder to eval mode (no training during transform)
        self.encoder_3d.eval()
        
        # Initialize 2D multi-crop transform
        self.multi_crop_transform = MultiCropTensorTransform(config)
        
    def encode_3d_to_2d(self, x):
        """Convert 3D tensor to 2D by encoding and taking middle slice."""
        with torch.no_grad():
            # Add batch dimension: [3, H, W, D] -> [1, 3, H, W, D]
            x_batch = x.unsqueeze(0)
            
            # Encode: [1, 3, H, W, D] -> [1, 1, H', W', D']
            encoded = self.encoder_3d(x_batch)
            
            # Take middle slice from depth dimension: [1, 1, H', W', D'] -> [1, H', W']
            middle_idx = encoded.shape[-1] // 2
            slice_2d = encoded[0, 0, :, :, middle_idx]
            
            # Convert to 3-channel for compatibility with DINO: [H', W'] -> [3, H', W']
            slice_2d = slice_2d.unsqueeze(0).repeat(3, 1, 1)
            
        return slice_2d
        
    def __call__(self, x):
        """
        Args:
            x: 3D tensor [3, H, W, D] - 3 channels (facies, poro, perm)
        Returns:
            List of 2D cropped tensors [3, crop_size, crop_size]
        """
        # Step 1: Encode 3D to 2D
        x_2d = self.encode_3d_to_2d(x)
        
        # Step 2: Apply 2D multi-crop transform
        crops = self.multi_crop_transform(x_2d)
        
        return crops


class Simple3DTransform:
    """Simple transform for 3D data that just normalizes without cropping."""
    
    def __init__(self, config):
        from encoder import SimpleEncoder3D
        
        # Initialize the 3D encoder  
        encoder_config = config.get('encoder_3d', {})
        self.encoder_3d = SimpleEncoder3D(
            in_channels=encoder_config.get('in_channels', 3),
            out_channels=encoder_config.get('out_channels', 1),
            block_out_channels=encoder_config.get('block_out_channels', [16, 32, 64]),
            block_out_types=encoder_config.get('block_out_types', ['down', 'down', 'same']),
            num_groups=encoder_config.get('num_groups', 16),
            dropout=encoder_config.get('dropout', 0.01),
            scales_w=encoder_config.get('scales_w', [2, 2]),
            scales_h=encoder_config.get('scales_h', [2, 2]),
            scales_d=encoder_config.get('scales_d', [2, 2])
        )
        
        # Set encoder to eval mode
        self.encoder_3d.eval()
        
    def __call__(self, x):
        """
        Args:
            x: 3D tensor [3, H, W, D] - 3 channels (facies, poro, perm)
        Returns:
            2D tensor [3, H', W'] - encoded and converted to 2D
        """
        with torch.no_grad():
            # Add batch dimension: [3, H, W, D] -> [1, 3, H, W, D]
            x_batch = x.unsqueeze(0)
            
            # Encode: [1, 3, H, W, D] -> [1, 1, H', W', D']
            encoded = self.encoder_3d(x_batch)
            
            # Take middle slice from depth dimension: [1, 1, H', W', D'] -> [1, H', W']
            middle_idx = encoded.shape[-1] // 2
            slice_2d = encoded[0, 0, :, :, middle_idx]
            
            # Convert to 3-channel for compatibility: [H', W'] -> [3, H', W']
            slice_2d = slice_2d.unsqueeze(0).repeat(3, 1, 1)
            
        return slice_2d


class Transform3D:
    """Basic transform for 3D data - normalization only."""
    
    def __init__(self, config):
        self.config = config
    
    def __call__(self, x):
        # Basic normalization for 3D geological data
        # Normalize each channel separately
        for i in range(x.shape[0]):  # For each channel
            x[i] = (x[i] - x[i].mean()) / (x[i].std() + 1e-8)
        return x


class MultiCrop3DTransform:
    """Multi-crop transform for 3D DINO self-supervised learning with trainable encoder."""
    
    def __init__(self, config):
        self.config = config
        
        # Get crop configuration
        data_aug = config.get('data_augmentation', {})
        self.crop_sizes = data_aug.get('crop_sizes', [64, 32])
        self.num_crops = data_aug.get('num_crops', [2, 6])
        
    def __call__(self, x):
        """
        Args:
            x: 3D tensor [3, H, W, D] - 3 channels (facies, poro, perm)
        Returns:
            List of cropped 3D tensors - will be processed by model's trainable encoder
        """
        crops = []
        C, H, W, D = x.shape
        
        for i, crop_size in enumerate(self.crop_sizes):
            num_crops_for_size = (self.num_crops[i] 
                                  if i < len(self.num_crops) 
                                  else self.num_crops[0])
            for _ in range(num_crops_for_size):
                # Random crop coordinates ensuring we don't go out of bounds
                h_start = torch.randint(0, max(1, H - crop_size + 1), (1,)).item()
                w_start = torch.randint(0, max(1, W - crop_size + 1), (1,)).item()
                d_start = torch.randint(0, max(1, D - crop_size + 1), (1,)).item()
                
                # Ensure we don't exceed boundaries
                h_end = min(h_start + crop_size, H)
                w_end = min(w_start + crop_size, W)
                d_end = min(d_start + crop_size, D)
                
                # Extract crop
                crop = x[:, 
                        h_start:h_end,
                        w_start:w_end, 
                        d_start:d_end]
                
                # Normalize crop
                for j in range(crop.shape[0]):
                    crop[j] = (crop[j] - crop[j].mean()) / (crop[j].std() + 1e-8)
                    
                crops.append(crop)
                
        return crops
