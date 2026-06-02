"""
Neural network architectures for VegeSSL.

This module implements the ResNet-18 based pixel-level encoder
for contrastive learning of vegetation features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class PixelContrastiveEncoder(nn.Module):
    """
    ResNet-18 based pixel-level encoder for contrastive learning.
    
    Architecture:
        - ResNet-18 backbone (ImageNet pretrained)
        - Output stride: 16
        - 1x1 conv projection head for dimensionality reduction
        - L2 normalization for cosine similarity computation
    
    Args:
        embedding_dim: Dimension of output embeddings (default: 128)
        pretrained: Whether to use ImageNet pretrained weights (default: True)
    
    Example:
        >>> encoder = PixelContrastiveEncoder(embedding_dim=128)
        >>> x = torch.randn(4, 3, 512, 512)
        >>> embeddings = encoder(x, upsample=True)  # (4, 128, 512, 512)
    """
    
    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super().__init__()
        
        # Load pretrained ResNet-18 backbone
        weights = 'IMAGENET1K_V1' if pretrained else None
        resnet = models.resnet18(weights=weights)
        
        # Extract backbone layers (remove FC and avgpool)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # Projection head: 512 -> embedding_dim
        self.projection = nn.Conv2d(512, embedding_dim, kernel_size=1, bias=False)
        
        self.embedding_dim = embedding_dim
    
    def forward(self, x: torch.Tensor, upsample: bool = False) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            upsample: If True, bilinear upsample output to input resolution
            
        Returns:
            L2-normalized embeddings of shape:
                - (B, embedding_dim, H/16, W/16) if upsample=False
                - (B, embedding_dim, H, W) if upsample=True
        """
        input_size = x.shape[2:]
        
        # ResNet backbone forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Project to embedding space
        x = self.projection(x)
        
        # L2 normalize for cosine similarity
        x = F.normalize(x, p=2, dim=1)
        
        # Optionally upsample to input resolution
        if upsample:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        return x
    
    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dim
    
    def freeze_backbone(self):
        """Freeze backbone weights for fine-tuning only the projection head."""
        for name, param in self.named_parameters():
            if 'projection' not in name:
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all weights."""
        for param in self.parameters():
            param.requires_grad = True
