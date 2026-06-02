"""
Dataset classes for VegeSSL contrastive learning.

This module implements data loading with augmentations for
the two-view contrastive learning paradigm.
"""

import random
from typing import List, Dict, Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from .utils import rgb_to_class_index, get_vegetation_mask


class ContrastiveCropDataset(Dataset):
    """
    Dataset for pixel-level contrastive learning.
    
    Extracts random crops from large images and applies stochastic
    augmentations to create two views of each crop for contrastive learning.
    
    Args:
        image_paths: List of paths to RGB images
        label_paths: List of paths to corresponding label images
        crop_size: Size of random crops (default: 512)
        crops_per_image: Number of crops per image per epoch (default: 2)
        cache_size: Number of images to cache in memory (default: 50)
    
    Returns:
        Dictionary with keys:
            - view1, view2: Augmented crop tensors (3, H, W)
            - mask: Class index mask (H, W)
            - veg_mask: Boolean vegetation mask (H, W)
            - img_idx: Source image index
    """
    
    def __init__(
        self,
        image_paths: List[str],
        label_paths: List[str],
        crop_size: int = 512,
        crops_per_image: int = 2,
        cache_size: int = 50
    ):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        
        # LRU-style image cache
        self.cache_size = cache_size
        self.image_cache: Dict[int, np.ndarray] = {}
        self.mask_cache: Dict[int, np.ndarray] = {}
        
        # Augmentation transforms
        self.color_jitter = T.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        )
        self.gaussian_blur = T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        
        # ImageNet normalization
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self) -> int:
        return len(self.image_paths) * self.crops_per_image
    
    def _load_cached(self, img_idx: int) -> tuple:
        """Load image and mask with caching."""
        if img_idx in self.image_cache:
            return self.image_cache[img_idx], self.mask_cache[img_idx]
        
        image = np.array(Image.open(self.image_paths[img_idx]))
        label_rgb = np.array(Image.open(self.label_paths[img_idx]))
        class_mask = rgb_to_class_index(label_rgb)
        
        # Add to cache if space available
        if len(self.image_cache) < self.cache_size:
            self.image_cache[img_idx] = image
            self.mask_cache[img_idx] = class_mask
        
        return image, class_mask
    
    def _apply_augmentation(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to create a view."""
        # Random horizontal flip
        if random.random() > 0.5:
            image_tensor = torch.flip(image_tensor, dims=[2])
        
        # Random vertical flip
        if random.random() > 0.5:
            image_tensor = torch.flip(image_tensor, dims=[1])
        
        # Color jitter
        image_pil = T.ToPILImage()(image_tensor)
        image_pil = self.color_jitter(image_pil)
        image_tensor = T.ToTensor()(image_pil)
        
        # Gaussian blur (50% probability)
        if random.random() > 0.5:
            image_tensor = self.gaussian_blur(image_tensor)
        
        # Normalize
        image_tensor = self.normalize(image_tensor)
        
        return image_tensor
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_idx = idx // self.crops_per_image
        
        image, class_mask = self._load_cached(img_idx)
        
        # Random crop location
        h, w = image.shape[:2]
        y = random.randint(0, h - self.crop_size)
        x = random.randint(0, w - self.crop_size)
        
        # Extract crop
        image_crop = image[y:y+self.crop_size, x:x+self.crop_size]
        mask_crop = class_mask[y:y+self.crop_size, x:x+self.crop_size]
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image_crop).permute(2, 0, 1).float() / 255.0
        
        # Create masks
        veg_mask = get_vegetation_mask(mask_crop)
        veg_mask_tensor = torch.from_numpy(veg_mask).bool()
        mask_tensor = torch.from_numpy(mask_crop).long()
        
        # Create two augmented views
        view1 = self._apply_augmentation(image_tensor.clone())
        view2 = self._apply_augmentation(image_tensor.clone())
        
        return {
            'view1': view1,
            'view2': view2,
            'mask': mask_tensor,
            'veg_mask': veg_mask_tensor,
            'img_idx': img_idx
        }
    
    def clear_cache(self):
        """Clear the image cache to free memory."""
        self.image_cache.clear()
        self.mask_cache.clear()


class InferenceDataset(Dataset):
    """
    Dataset for inference/embedding extraction.
    
    Returns full images without augmentation for extracting
    segment-level embeddings.
    
    Args:
        image_paths: List of paths to RGB images
        label_paths: List of paths to corresponding label images
    """
    
    def __init__(self, image_paths: List[str], label_paths: List[str]):
        self.image_paths = image_paths
        self.label_paths = label_paths
        
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image = np.array(Image.open(self.image_paths[idx]))
        label_rgb = np.array(Image.open(self.label_paths[idx]))
        class_mask = rgb_to_class_index(label_rgb)
        
        # Normalize image
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        image_tensor = self.normalize(image_tensor)
        
        mask_tensor = torch.from_numpy(class_mask).long()
        veg_mask = get_vegetation_mask(class_mask)
        veg_mask_tensor = torch.from_numpy(veg_mask).bool()
        
        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'veg_mask': veg_mask_tensor,
            'image_path': self.image_paths[idx],
            'label_path': self.label_paths[idx]
        }
