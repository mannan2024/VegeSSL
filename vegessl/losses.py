"""
Loss functions for VegeSSL contrastive learning.

This module implements:
- PixelInfoNCELoss: Standard pixel-wise InfoNCE loss
- HardNegativeInfoNCELoss: InfoNCE with emphasis on hard negatives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelInfoNCELoss(nn.Module):
    """
    Pixel-wise InfoNCE loss for contrastive learning.
    
    For each pixel position (i, j):
        - Positive: Same position in the augmented view
        - Negatives: All other pixels in the batch
    
    The loss is computed only on vegetation pixels to focus learning
    on the classes of interest.
    
    Args:
        temperature: Softmax temperature for similarity scaling (default: 0.07)
    
    Reference:
        Chen et al., "A Simple Framework for Contrastive Learning
        of Visual Representations", ICML 2020
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        veg_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute pixel-wise InfoNCE loss.
        
        Args:
            z1: Embeddings from view 1, shape (B, D, H, W)
            z2: Embeddings from view 2, shape (B, D, H, W)
            veg_mask: Vegetation mask, shape (B, H, W), True for vegetation
            
        Returns:
            Scalar loss value
        """
        B, D, H, W = z1.shape
        
        # Flatten spatial dimensions: (B*H*W, D)
        z1_flat = z1.permute(0, 2, 3, 1).reshape(-1, D)
        z2_flat = z2.permute(0, 2, 3, 1).reshape(-1, D)
        veg_mask_flat = veg_mask.reshape(-1)
        
        # Select only vegetation pixels
        veg_indices = torch.where(veg_mask_flat)[0]
        
        if len(veg_indices) == 0:
            return z1.sum() * 0.0  # Return zero loss with gradient
        
        # Subsample for memory efficiency
        max_samples = min(4096, len(veg_indices))
        if len(veg_indices) > max_samples:
            perm = torch.randperm(len(veg_indices), device=z1.device)[:max_samples]
            veg_indices = veg_indices[perm]
        
        # Get embeddings for selected pixels
        z1_veg = z1_flat[veg_indices]
        z2_veg = z2_flat[veg_indices]
        
        # Compute similarity matrix: (M, M)
        # Positive pairs are on the diagonal
        sim_matrix = torch.mm(z1_veg, z2_veg.t()) / self.temperature
        
        # Cross-entropy loss with diagonal as positive
        labels = torch.arange(len(veg_indices), device=z1.device)
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class HardNegativeInfoNCELoss(nn.Module):
    """
    InfoNCE loss with emphasis on hard negative pairs.
    
    Combines standard pixel InfoNCE with an additional term that
    pushes apart embeddings of hard negatives (pixels that are
    similar in embedding space but have different labels).
    
    Args:
        temperature: Softmax temperature (default: 0.07)
        lambda_hard: Weight for hard negative loss term (default: 0.8)
    """
    
    def __init__(self, temperature: float = 0.07, lambda_hard: float = 0.8):
        super().__init__()
        self.temperature = temperature
        self.lambda_hard = lambda_hard
        self.base_loss = PixelInfoNCELoss(temperature)
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        veg_mask: torch.Tensor,
        hard_neg_embeddings: torch.Tensor = None,
        hard_neg_labels: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute combined loss with hard negative emphasis.
        
        Args:
            z1, z2: Embeddings from two views
            veg_mask: Vegetation mask
            hard_neg_embeddings: Pre-computed hard negative embeddings (optional)
            hard_neg_labels: Labels for hard negatives (optional)
            
        Returns:
            Combined loss value
        """
        base_loss = self.base_loss(z1, z2, veg_mask)
        
        if hard_neg_embeddings is None:
            return base_loss
        
        B, D, H, W = z1.shape
        z1_flat = z1.permute(0, 2, 3, 1).reshape(-1, D)
        veg_mask_flat = veg_mask.reshape(-1)
        
        veg_indices = torch.where(veg_mask_flat)[0]
        if len(veg_indices) == 0:
            return base_loss
        
        # Sample subset of vegetation pixels
        max_samples = min(1024, len(veg_indices))
        perm = torch.randperm(len(veg_indices), device=z1.device)[:max_samples]
        z1_veg = z1_flat[veg_indices[perm]]
        
        # Compute similarity with hard negatives
        if isinstance(hard_neg_embeddings, torch.Tensor):
            hard_neg_tensor = hard_neg_embeddings.to(z1.device)
        else:
            hard_neg_tensor = torch.from_numpy(hard_neg_embeddings).to(z1.device)
        
        sim_matrix = torch.mm(z1_veg, hard_neg_tensor.t()) / self.temperature
        
        # Push away from hard negatives (minimize similarity)
        hard_loss = torch.logsumexp(sim_matrix, dim=1).mean()
        
        return base_loss + self.lambda_hard * hard_loss
