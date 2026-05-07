import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import SimpleEncoder3D

class DINOV2Analog(nn.Module):
    def __init__(self, config):
        super().__init__()
        backbone = config['model']['backbone']
        proj = config['model']['input_projection']
        proj_head = config['model']['projection_head']
        init = config['model']['initialization']
        
        self.patch_size = backbone['patch_size']
        self.embed_dim = backbone['embed_dim']

        # 0. Input conv 20 -> 3 channels
        self.input_proj = nn.Conv2d(
            proj['in_channels'], 
            proj['out_channels'], 
            kernel_size=proj['kernel_size']
        )

        # Calculate base number of patches
        self.base_num_patches = (backbone['base_img_size'] // self.patch_size) ** 2

        # 1. Patch Embedding
        self.patch_embed = nn.Conv2d(
            backbone['in_channels'], 
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # 2. CLS token and positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.base_num_patches + 1, self.embed_dim)  # +1 for CLS
        )

        # 3. Transformer Encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(
                self.embed_dim, 
                backbone['num_heads'], 
                mlp_ratio=backbone['mlp_ratio'], 
                qkv_bias=backbone['qkv_bias']
            )
            for _ in range(backbone['num_layers'])
        ])

        # 4. Projection Head
        self.proj_head = nn.Sequential(
            nn.Linear(self.embed_dim, proj_head['hidden_dim']),
            nn.GELU(),
            nn.Linear(proj_head['hidden_dim'], proj_head['output_dim']),
            nn.LayerNorm(proj_head['output_dim'])
        )

        self.norm = nn.LayerNorm(self.embed_dim)
        self.init_weights(init['pos_embed_std'], init['cls_token_std'])

    def init_weights(self, pos_embed_std, cls_token_std):
        nn.init.trunc_normal_(self.pos_embed, std=pos_embed_std)
        nn.init.trunc_normal_(self.cls_token, std=cls_token_std)

    def forward(self, x):
        # Input conv 20 -> 3 channels
        x = self.input_proj(x)

        B, C, H, W = x.shape

        # Calculate number of patches
        P_H, P_W = H // self.patch_size, W // self.patch_size
        # num_patches = P_H * P_W

        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # Positional embedding with interpolation
        pos_embed = self.interpolate_pos_encoding(P_H, P_W)
        x = x + pos_embed

        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return {
            'cls_token': x[:, 0],
            'all_tokens': x[:, 1:],
            'projection': self.proj_head(x[:, 0])
        }

    def interpolate_pos_encoding(self, pH, pW):
        current_num_patches = pH * pW

        if current_num_patches == self.base_num_patches:
            return self.pos_embed

        # Split CLS and patch embeddings
        cls_pos = self.pos_embed[:, :1, :]
        patch_pos = self.pos_embed[:, 1:, :]

        # Reshape to spatial dimensions
        patch_pos = patch_pos.transpose(1, 2).reshape(
            1, self.embed_dim,
            int(self.base_num_patches ** 0.5),
            int(self.base_num_patches ** 0.5)
        )

        # Interpolate spatial embeddings
        patch_pos = F.interpolate(
            patch_pos,
            size=(pH, pW),
            mode="bilinear",
            align_corners=False,
        )

        # Re-flatten and combine with CLS
        patch_pos = patch_pos.flatten(2).transpose(1, 2)
        return torch.cat([cls_pos, patch_pos], dim=1)


class DINOV2Analog3D(nn.Module):
    """3D DINO model that includes trainable SimpleEncoder3D for end-to-end learning."""
    
    def __init__(self, config):
        super().__init__()
        backbone = config['model']['backbone']
        proj_head = config['model']['projection_head']
        init = config['model']['initialization']
        
        self.patch_size = backbone['patch_size']
        self.embed_dim = backbone['embed_dim']

        # Trainable 3D Encoder: 3-channel 3D -> 1-channel 3D (reduced size)
        from encoder import SimpleEncoder3D
        encoder_config = config['model'].get('encoder_3d', {})
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

        # Calculate base number of patches for 2D
        self.base_num_patches = (backbone['base_img_size'] // self.patch_size) ** 2

        # 2D Patch embedding (applied to 2D slices from 3D encoder)
        self.patch_embed = nn.Conv2d(
            3,  # 3 channels after replication for DINO compatibility
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

        # CLS token and positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.base_num_patches + 1, self.embed_dim)
        )

        # Transformer Encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(
                self.embed_dim, 
                backbone['num_heads'], 
                mlp_ratio=backbone['mlp_ratio'], 
                qkv_bias=backbone['qkv_bias']
            )
            for _ in range(backbone['num_layers'])
        ])

        # Projection Head
        self.proj_head = nn.Sequential(
            nn.Linear(self.embed_dim, proj_head['hidden_dim']),
            nn.GELU(),
            nn.Linear(proj_head['hidden_dim'], proj_head['output_dim']),
            nn.LayerNorm(proj_head['output_dim'])
        )

        self.norm = nn.LayerNorm(self.embed_dim)
        self.init_weights(init['pos_embed_std'], init['cls_token_std'])

    def init_weights(self, pos_embed_std, cls_token_std):
        nn.init.trunc_normal_(self.pos_embed, std=pos_embed_std)
        nn.init.trunc_normal_(self.cls_token, std=cls_token_std)

    def encode_3d_to_2d(self, x):
        """Convert 3D tensor to 2D by encoding and taking middle slice."""
        # Encode: [B, 3, H, W, D] -> [B, 1, H', W', D']
        encoded = self.encoder_3d(x)
        
        # Take middle slice from depth dimension: [B, 1, H', W', D'] -> [B, 1, H', W']
        middle_idx = encoded.shape[-1] // 2
        slice_2d = encoded[:, :, :, :, middle_idx]
        
        # Convert to 3-channel for compatibility with DINO: [B, 1, H', W'] -> [B, 3, H', W']
        slice_2d = slice_2d.repeat(1, 3, 1, 1)
        
        return slice_2d

    def forward(self, x):
        """
        Args:
            x: 3D tensor [B, 3, H, W, D] - 3 channels (facies, poro, perm)
        """
        B = x.shape[0]
        
        # Step 1: 3D encoding to 2D (trainable)
        x_2d = self.encode_3d_to_2d(x)  # [B, 3, H', W']
        
        C, H, W = x_2d.shape[1], x_2d.shape[2], x_2d.shape[3]
        
        # Step 2: Apply 2D patch embedding
        x_patches = self.patch_embed(x_2d)  # [B, embed_dim, pH, pW]
        x_patches = x_patches.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        
        P_H, P_W = H // self.patch_size, W // self.patch_size

        # Step 3: Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x_patches), dim=1)

        # Step 4: Positional embedding with interpolation
        pos_embed = self.interpolate_pos_encoding(P_H, P_W)
        x = x + pos_embed

        # Step 5: Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return {
            'cls_token': x[:, 0],
            'all_tokens': x[:, 1:],
            'projection': self.proj_head(x[:, 0])
        }

    def interpolate_pos_encoding(self, pH, pW):
        current_num_patches = pH * pW

        if current_num_patches == self.base_num_patches:
            return self.pos_embed

        # Split CLS and patch embeddings
        cls_pos = self.pos_embed[:, :1, :]
        patch_pos = self.pos_embed[:, 1:, :]

        # Reshape to spatial dimensions
        patch_pos = patch_pos.transpose(1, 2).reshape(
            1, self.embed_dim,
            int(self.base_num_patches ** 0.5),
            int(self.base_num_patches ** 0.5)
        )

        # Interpolate spatial embeddings
        patch_pos = F.interpolate(
            patch_pos,
            size=(pH, pW),
            mode="bilinear",
            align_corners=False,
        )

        # Re-flatten and combine with CLS
        patch_pos = patch_pos.flatten(2).transpose(1, 2)
        return torch.cat([cls_pos, patch_pos], dim=1)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class DinoClassifier(nn.Module):
    def __init__(self, config, backbone = None):
        super(DinoClassifier, self).__init__()
        proj = config['model']['input_projection']
        clf = config['model']['classifier_head']
        
        # Input projection
        self.input_proj = nn.Conv2d(
            proj['in_channels'], 
            proj['out_channels'], 
            kernel_size=proj['kernel_size']
        )
        
        # Load pretrained backbone
        self.transformer = torch.hub.load('facebookresearch/dinov2', config['model']['backbone']['name'])
        
        # Freeze backbone if specified
        if config['model']['freeze_backbone']:
            self.transformer.eval()
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(config['model']['backbone']['embed_dim'], clf['hidden_dim']),
            nn.ReLU(),
            nn.Linear(clf['hidden_dim'], clf['num_classes'])
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        repr = self.transformer.norm(x)
        x = self.classifier(repr)
        return x, repr

    def repr(self, x):
        repr = self.transformer.norm(x)
        return repr


class DinoRegressor(nn.Module):
    def __init__(self, config, backbone = None):
        super(DinoRegressor, self).__init__()
        proj = config['model']['input_projection']
        clf = config['model']['classifier_head']
        
        # Input projection
        self.input_proj = nn.Conv2d(
            proj['in_channels'], 
            proj['out_channels'], 
            kernel_size=proj['kernel_size']
        )
        
        # Load pretrained backbone
        self.transformer = torch.hub.load('facebookresearch/dinov2', config['model']['backbone']['name'])
        
        # Freeze backbone if specified
        if config['model']['freeze_backbone']:
            self.transformer.eval()
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(config['model']['backbone']['embed_dim'], clf['hidden_dim']),
            nn.ReLU(),
            nn.Linear(clf['hidden_dim'], clf['context_dim'])
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        repr = self.transformer.norm(x)
        x = self.regressor(repr)
        return x, repr

    def repr(self, x):
        repr = self.transformer.norm(x)
        return repr


class DinoClassifier3D(nn.Module):
    """3D DINO classifier with trainable 3D encoder."""
    
    def __init__(self, config, backbone=None):
        super(DinoClassifier3D, self).__init__()
        clf = config['model']['classifier_head']
        
        # 3D DINO backbone with trainable encoder
        self.backbone = DINOV2Analog3D(config)
        
        # Freeze backbone if specified
        if config['model']['freeze_backbone']:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(config['model']['backbone']['embed_dim'], clf['hidden_dim']),
            nn.ReLU(),
            nn.Linear(clf['hidden_dim'], clf['num_classes'])
        )

    def forward(self, x):
        """
        Args:
            x: 3D tensor [B, 3, H, W, D]
        """
        backbone_out = self.backbone(x)
        repr_vec = backbone_out['cls_token']
        logits = self.classifier(repr_vec)
        return logits, repr_vec

    def repr(self, x):
        backbone_out = self.backbone(x)
        return backbone_out['cls_token']


class DinoRegressor3D(nn.Module):
    """3D DINO regressor with trainable 3D encoder."""
    
    def __init__(self, config, backbone=None):
        super(DinoRegressor3D, self).__init__()
        clf = config['model']['classifier_head']
        
        # 3D DINO backbone with trainable encoder
        self.backbone = DINOV2Analog3D(config)
        
        # Load pretrained SSL weights if specified
        if config['model'].get('pretrained_ssl_path'):
            self._load_pretrained_ssl_weights(config['model']['pretrained_ssl_path'])
        
        # Freeze backbone if specified
        if config['model']['freeze_backbone']:
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(config['model']['backbone']['embed_dim'], clf['hidden_dim']),
            nn.ReLU(),
            nn.Linear(clf['hidden_dim'], clf['context_dim'])
        )

    def forward(self, x):
        """
        Args:
            x: 3D tensor [B, 3, H, W, D]
        """
        backbone_out = self.backbone(x)
        repr_vec = backbone_out['cls_token']
        output = self.regressor(repr_vec)
        return output, repr_vec

    def repr(self, x):
        backbone_out = self.backbone(x)
        return backbone_out['cls_token']
    
    def _load_pretrained_ssl_weights(self, checkpoint_path):
        """Load pretrained SSL weights into the backbone."""
        try:
            print(f"Loading pretrained SSL weights from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'student_state_dict' in checkpoint:
                state_dict = checkpoint['student_state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load only backbone weights (excluding classifier/regressor head)
            backbone_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('backbone.'):
                    backbone_state_dict[key[9:]] = value  # Remove 'backbone.' prefix
                elif not key.startswith(('regressor.', 'classifier.', 'proj_head.')):
                    backbone_state_dict[key] = value
            
            # Load weights with strict=False to allow missing keys
            missing_keys, unexpected_keys = self.backbone.load_state_dict(backbone_state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys in backbone: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys in checkpoint: {unexpected_keys}")
                
            print("✅ Successfully loaded pretrained SSL weights")
            
        except Exception as e:
            print(f"❌ Failed to load pretrained SSL weights: {e}")
            print("Continuing with random initialization...")


class DinoRegressor3DPretrained(nn.Module):
    """3D supervised regressor using pretrained DINOv2 backbone."""
    
    def __init__(self, config, backbone=None):
        super(DinoRegressor3DPretrained, self).__init__()
        clf = config['model']['classifier_head']
        
        # Pretrained DINOv2 3D backbone
        self.backbone = DINOV2Analog3DPretrained(config)
        
        # Load pretrained SSL weights if specified
        if config['model'].get('pretrained_ssl_path'):
            self._load_pretrained_ssl_weights(config['model']['pretrained_ssl_path'])
        
        # Freeze backbone if specified
        if config['model'].get('freeze_backbone', False):
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(config['model']['backbone']['embed_dim'], clf['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(clf['hidden_dim'], clf['context_dim'])
        )

    def forward(self, x):
        """
        Args:
            x: 3D tensor [B, 3, H, W, D]
        Returns:
            predictions: [B, context_dim]
            features: [B, embed_dim]
        """
        backbone_out = self.backbone(x)
        features = backbone_out['cls_token']
        predictions = self.regressor(features)
        return predictions, features

    def repr(self, x):
        """Get representation from the model."""
        backbone_out = self.backbone(x)
        return backbone_out['cls_token']
    
    def _load_pretrained_ssl_weights(self, checkpoint_path):
        """Load pretrained SSL weights into the backbone."""
        try:
            print(f"Loading pretrained SSL weights from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'student_state_dict' in checkpoint:
                state_dict = checkpoint['student_state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load only backbone weights (excluding classifier/regressor head)
            backbone_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('backbone.'):
                    backbone_state_dict[key[9:]] = value  # Remove 'backbone.' prefix
                elif not key.startswith(('regressor.', 'classifier.', 'proj_head.')):
                    backbone_state_dict[key] = value
            
            # Load weights with strict=False to allow missing keys
            missing_keys, unexpected_keys = self.backbone.load_state_dict(backbone_state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys in backbone: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys in checkpoint: {unexpected_keys}")
                
            print("✅ Successfully loaded pretrained SSL weights")
            
        except Exception as e:
            print(f"❌ Failed to load pretrained SSL weights: {e}")
            print("Continuing with random initialization...")


class DINOV2FineTune(nn.Module):
    """Fine-tuning DINOv2 pretrained model for self-supervised learning."""
    
    def __init__(self, config):
        super().__init__()
        backbone = config['model']['backbone']
        proj = config['model']['input_projection']
        proj_head = config['model']['projection_head']
        
        # Input projection to convert 20 channels to 3 channels
        self.input_proj = nn.Conv2d(
            proj['in_channels'], 
            proj['out_channels'], 
            kernel_size=proj['kernel_size']
        )
        
        # Load pretrained DINOv2 backbone
        self.transformer = torch.hub.load('facebookresearch/dinov2', backbone['name'])
        
        # Handle layer freezing for fine-tuning
        if config['model']['fine_tune'].get('freeze_early_layers', False):
            frozen_layers = config['model']['fine_tune'].get('frozen_layers', 6)
            self._freeze_early_layers(frozen_layers)
        
        # Custom projection head for DINO loss
        self.proj_head = nn.Sequential(
            nn.Linear(backbone['embed_dim'], proj_head['hidden_dim']),
            nn.GELU(),
            nn.Linear(proj_head['hidden_dim'], proj_head['output_dim']),
            nn.LayerNorm(proj_head['output_dim'])
        )
        
        # Normalization layer - DINOv2 already has norm, so just use the output directly
        # self.norm = nn.LayerNorm(backbone['embed_dim'])
        
    def _freeze_early_layers(self, frozen_layers):
        """Freeze the first N layers of the transformer."""
        # Freeze patch embedding
        for param in self.transformer.patch_embed.parameters():
            param.requires_grad = False
        
        # Freeze positional embeddings
        if hasattr(self.transformer, 'pos_embed'):
            self.transformer.pos_embed.requires_grad = False
        if hasattr(self.transformer, 'cls_token'):
            self.transformer.cls_token.requires_grad = False
        
        # Freeze first N transformer blocks
        for i, block in enumerate(self.transformer.blocks):
            if i < frozen_layers:
                for param in block.parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        
        # Forward through pretrained DINOv2
        x = self.transformer(x)
        
        # DINOv2 already applies normalization, so use output directly
        # Apply projection head
        projection = self.proj_head(x)
        
        return {
            'cls_token': x,
            'projection': projection
        }


class DinoClassifierFromScratch(nn.Module):
    """Supervised classifier built from scratch without pretrained weights."""
    
    def __init__(self, config):
        super().__init__()
        backbone = config['model']['backbone']
        proj = config['model']['input_projection']
        clf = config['model']['classifier_head']
        init = config['model']['initialization']
        
        self.patch_size = backbone['patch_size']
        self.embed_dim = backbone['embed_dim']
        
        # Input projection
        self.input_proj = nn.Conv2d(
            proj['in_channels'], 
            proj['out_channels'], 
            kernel_size=proj['kernel_size']
        )
        
        # Calculate base number of patches
        self.base_num_patches = (backbone['base_img_size'] // self.patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            backbone['in_channels'], 
            self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )
        
        # CLS token and positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.base_num_patches + 1, self.embed_dim)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                self.embed_dim, 
                backbone['num_heads'], 
                mlp_ratio=backbone['mlp_ratio'], 
                qkv_bias=backbone['qkv_bias']
            )
            for _ in range(backbone['num_layers'])
        ])
        
        # Normalization
        self.norm = nn.LayerNorm(self.embed_dim)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, clf['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(clf['hidden_dim'], clf['num_classes'])
        )
        
        # Initialize weights
        self.init_weights(init['pos_embed_std'], init['cls_token_std'])
        
    def init_weights(self, pos_embed_std, cls_token_std):
        nn.init.trunc_normal_(self.pos_embed, std=pos_embed_std)
        nn.init.trunc_normal_(self.cls_token, std=cls_token_std)
        
    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        
        B, C, H, W = x.shape
        
        # Calculate number of patches
        P_H, P_W = H // self.patch_size, W // self.patch_size
        
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Positional embedding with interpolation
        pos_embed = self.interpolate_pos_encoding(P_H, P_W)
        x = x + pos_embed
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        # Classification
        cls_token = x[:, 0]
        logits = self.classifier(cls_token)
        
        return logits, cls_token
        
    def interpolate_pos_encoding(self, pH, pW):
        current_num_patches = pH * pW
        
        if current_num_patches == self.base_num_patches:
            return self.pos_embed
        
        # Split CLS and patch embeddings
        cls_pos = self.pos_embed[:, :1, :]
        patch_pos = self.pos_embed[:, 1:, :]
        
        # Reshape to spatial dimensions
        patch_pos = patch_pos.transpose(1, 2).reshape(
            1, self.embed_dim,
            int(self.base_num_patches ** 0.5),
            int(self.base_num_patches ** 0.5)
        )
        
        # Interpolate spatial embeddings
        patch_pos = F.interpolate(
            patch_pos,
            size=(pH, pW),
            mode="bilinear",
            align_corners=False,
        )
        
        # Re-flatten and combine with CLS
        patch_pos = patch_pos.flatten(2).transpose(1, 2)
        return torch.cat([cls_pos, patch_pos], dim=1)
        
    def repr(self, x):
        """Get representation from the model."""
        # Input projection
        x = self.input_proj(x)
        
        B, C, H, W = x.shape
        P_H, P_W = H // self.patch_size, W // self.patch_size
        
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Positional embedding
        pos_embed = self.interpolate_pos_encoding(P_H, P_W)
        x = x + pos_embed
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        return x[:, 0]  # Return CLS token representation


class DINOV2Analog3DPretrained(nn.Module):
    """3D DINO model using pretrained DINOv2 backbone with trainable 3D encoder."""
    
    def __init__(self, config):
        super().__init__()
        backbone = config['model']['backbone']
        proj_head = config['model']['projection_head']
        init = config['model']['initialization']
        
        self.embed_dim = backbone['embed_dim']

        # Trainable 3D Encoder: 3-channel 3D -> 1-channel 3D (reduced size)
        from encoder import SimpleEncoder3D
        encoder_config = config['model'].get('encoder_3d', {})
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

        # Load pretrained DINOv2 backbone
        self.transformer = torch.hub.load('facebookresearch/dinov2', backbone['name'])
        
        # Freeze backbone if specified
        if config['model'].get('freeze_backbone', False):
            for param in self.transformer.parameters():
                param.requires_grad = False

        # Custom projection head for DINO loss
        self.proj_head = nn.Sequential(
            nn.Linear(self.embed_dim, proj_head['hidden_dim']),
            nn.GELU(),
            nn.Linear(proj_head['hidden_dim'], proj_head['output_dim']),
            nn.LayerNorm(proj_head['output_dim'])
        )

    def encode_3d_to_2d(self, x):
        """Convert 3D tensor to 2D by encoding and taking middle slice."""
        # Encode: [B, 3, H, W, D] -> [B, 1, H', W', D']
        encoded = self.encoder_3d(x)
        
        # Take middle slice from depth dimension: [B, 1, H', W', D'] -> [B, 1, H', W']
        middle_idx = encoded.shape[-1] // 2
        slice_2d = encoded[:, :, :, :, middle_idx]
        
        # Convert to 3-channel for compatibility with DINOv2: [B, 1, H', W'] -> [B, 3, H', W']
        slice_2d = slice_2d.repeat(1, 3, 1, 1)
        
        # Resize to DINOv2's expected input size (224x224)
        slice_2d = F.interpolate(slice_2d, size=(224, 224), mode='bilinear', align_corners=False)
        
        return slice_2d

    def forward(self, x):
        """
        Args:
            x: 3D tensor [B, 3, H, W, D] - 3 channels (facies, poro, perm)
        """
        # Step 1: 3D encoding to 2D (trainable)
        x_2d = self.encode_3d_to_2d(x)  # [B, 3, 224, 224]
        
        # Step 2: Forward through pretrained DINOv2 (returns dict with 'x_norm_clstoken', 'x_norm_patchtokens', etc.)
        dino_output = self.transformer(x_2d, is_training=True)
        
        # Extract CLS token (should be first element or in 'x_norm_clstoken')
        if isinstance(dino_output, dict):
            if 'x_norm_clstoken' in dino_output:
                cls_token = dino_output['x_norm_clstoken']
            elif 'cls_token' in dino_output:
                cls_token = dino_output['cls_token']
            else:
                # Fallback: try to get the first key that looks like tokens
                cls_token = list(dino_output.values())[0]
        else:
            # If it's just a tensor, assume it's the cls token
            cls_token = dino_output

        # Apply custom projection head
        projection = self.proj_head(cls_token)

        return {
            'cls_token': cls_token,
            'projection': projection
        }


class DinoClassifier3DPretrained(nn.Module):
    """3D supervised classifier using pretrained DINOv2 backbone."""
    
    def __init__(self, config, backbone=None):
        super(DinoClassifier3DPretrained, self).__init__()
        clf = config['model']['classifier_head']
        
        # Pretrained DINOv2 3D backbone
        self.backbone = DINOV2Analog3DPretrained(config)
        
        # Freeze backbone if specified
        if config['model'].get('freeze_backbone', False):
            self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(config['model']['backbone']['embed_dim'], clf['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(clf['hidden_dim'], clf['num_classes'])
        )

    def forward(self, x):
        """
        Args:
            x: 3D tensor [B, 3, H, W, D]
        Returns:
            logits: [B, num_classes]
            features: [B, embed_dim]
        """
        backbone_out = self.backbone(x)
        features = backbone_out['cls_token']
        logits = self.classifier(features)
        return logits, features

    def repr(self, x):
        """Get representation from the model."""
        backbone_out = self.backbone(x)
        return backbone_out['cls_token']
