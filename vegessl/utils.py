"""
Utility functions for VegeSSL.

This module provides utilities for:
- Checkpoint saving/loading
- Label encoding/decoding
- Segment extraction
- Reproducibility (seed setting)
"""

import os
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import torch
import cv2


# Default class configuration for vegetation dataset
DEFAULT_CLASS_COLORS = np.array([
    [0, 0, 0],        # Invalid
    [255, 255, 255],  # Other
    [151, 219, 242],  # Water
    [38, 115, 0],     # Abandoned land
    [255, 234, 190],  # Bare land
    [245, 245, 122],  # Farmland
    [233, 255, 190],  # Herbaceous
    [114, 137, 68],   # Broadleaf forest
    [115, 178, 115],  # Shrubland
    [240, 176, 207]   # Orchard
], dtype=np.uint8)

DEFAULT_CLASS_NAMES = [
    'Invalid', 'Other', 'Water', 'Abandoned land', 'Bare land',
    'Farmland', 'Herbaceous', 'Broadleaf forest', 'Shrubland', 'Orchard'
]

# Vegetation classes (classes of interest for mislabel detection)
VEGETATION_CLASSES = [3, 5, 6, 7, 8, 9]

# Module-level configuration (can be updated via set_class_config)
_class_colors = DEFAULT_CLASS_COLORS.copy()
_class_names = DEFAULT_CLASS_NAMES.copy()
_vegetation_classes = VEGETATION_CLASSES.copy()


def set_class_config(
    class_colors: np.ndarray = None,
    class_names: List[str] = None,
    vegetation_classes: List[int] = None
):
    """
    Set the class configuration for label encoding.
    
    Args:
        class_colors: RGB colors for each class, shape (N, 3)
        class_names: Names for each class
        vegetation_classes: Indices of vegetation classes
    """
    global _class_colors, _class_names, _vegetation_classes
    
    if class_colors is not None:
        _class_colors = np.array(class_colors, dtype=np.uint8)
    if class_names is not None:
        _class_names = list(class_names)
    if vegetation_classes is not None:
        _vegetation_classes = list(vegetation_classes)


def get_class_config() -> Dict[str, Any]:
    """Return the current class configuration."""
    return {
        'class_colors': _class_colors,
        'class_names': _class_names,
        'vegetation_classes': _vegetation_classes,
        'num_classes': len(_class_names)
    }


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(
    data: Dict[str, Any],
    name: str,
    checkpoint_dir: str = "./checkpoints"
):
    """
    Save a checkpoint to disk.
    
    Args:
        data: Dictionary containing model state, optimizer state, etc.
        name: Checkpoint name (without extension)
        checkpoint_dir: Directory for checkpoints
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{name}.pt")
    torch.save(data, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(
    name: str,
    checkpoint_dir: str = "./checkpoints",
    device: torch.device = None
) -> Optional[Dict[str, Any]]:
    """
    Load a checkpoint from disk.
    
    Args:
        name: Checkpoint name (without extension)
        checkpoint_dir: Directory containing checkpoints
        device: Device to map tensors to
        
    Returns:
        Checkpoint dictionary or None if not found
    """
    path = os.path.join(checkpoint_dir, f"{name}.pt")
    
    if not os.path.exists(path):
        return None
    
    map_location = device if device else 'cpu'
    checkpoint = torch.load(path, map_location=map_location)
    print(f"Loaded checkpoint: {path}")
    return checkpoint


def save_numpy_checkpoint(
    data: Any,
    name: str,
    checkpoint_dir: str = "./checkpoints"
):
    """Save numpy data to disk."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{name}.npy")
    np.save(path, data)
    print(f"Saved numpy checkpoint: {path}")


def load_numpy_checkpoint(
    name: str,
    checkpoint_dir: str = "./checkpoints"
) -> Optional[np.ndarray]:
    """Load numpy data from disk."""
    path = os.path.join(checkpoint_dir, f"{name}.npy")
    
    if not os.path.exists(path):
        return None
    
    data = np.load(path, allow_pickle=True)
    print(f"Loaded numpy checkpoint: {path}")
    return data


def rgb_to_class_index(label_rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB label image to class index mask.
    
    Args:
        label_rgb: RGB label image, shape (H, W, 3)
        
    Returns:
        Class index mask, shape (H, W)
    """
    h, w = label_rgb.shape[:2]
    pixels = label_rgb.reshape(-1, 3)
    
    # Encode colors as single integers for fast comparison
    encoded = (
        pixels[:, 0].astype(np.int32) * 256 * 256 +
        pixels[:, 1].astype(np.int32) * 256 +
        pixels[:, 2].astype(np.int32)
    )
    
    class_mask = np.zeros(h * w, dtype=np.uint8)
    
    for idx, color in enumerate(_class_colors):
        encoded_color = int(color[0]) * 256 * 256 + int(color[1]) * 256 + int(color[2])
        class_mask[encoded == encoded_color] = idx
    
    return class_mask.reshape(h, w)


def class_index_to_rgb(class_mask: np.ndarray) -> np.ndarray:
    """
    Convert class index mask to RGB label image.
    
    Args:
        class_mask: Class index mask, shape (H, W)
        
    Returns:
        RGB label image, shape (H, W, 3)
    """
    h, w = class_mask.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_idx, color in enumerate(_class_colors):
        mask = class_mask == class_idx
        rgb_image[mask] = color
    
    return rgb_image


def get_vegetation_mask(class_mask: np.ndarray) -> np.ndarray:
    """
    Create binary mask for vegetation classes.
    
    Args:
        class_mask: Class index mask, shape (H, W)
        
    Returns:
        Boolean mask, True for vegetation pixels
    """
    veg_mask = np.zeros_like(class_mask, dtype=bool)
    for veg_class in _vegetation_classes:
        veg_mask |= (class_mask == veg_class)
    return veg_mask


def extract_segments(
    class_mask: np.ndarray,
    min_segment_size: int = 10
) -> List[Dict[str, Any]]:
    """
    Extract connected component segments from vegetation mask.
    
    Args:
        class_mask: Class index mask, shape (H, W)
        min_segment_size: Minimum pixels for a valid segment
        
    Returns:
        List of segment dictionaries with keys:
            - label: Segment ID
            - class_idx: Vegetation class index
            - class_name: Vegetation class name
            - pixel_coords: (N, 2) array of (y, x) coordinates
            - bbox: (y_min, y_max, x_min, x_max)
            - area: Number of pixels
    """
    segments = []
    segment_id = 0
    
    for veg_class in _vegetation_classes:
        # Binary mask for this class
        class_binary = (class_mask == veg_class).astype(np.uint8)
        
        # Find connected components
        num_labels, labeled_mask = cv2.connectedComponents(
            class_binary, connectivity=8
        )
        
        # Extract each segment
        for label_id in range(1, num_labels):
            segment_mask = labeled_mask == label_id
            pixel_coords = np.argwhere(segment_mask)
            
            if len(pixel_coords) < min_segment_size:
                continue
            
            y_coords = pixel_coords[:, 0]
            x_coords = pixel_coords[:, 1]
            
            segments.append({
                'label': segment_id,
                'class_idx': veg_class,
                'class_name': _class_names[veg_class],
                'pixel_coords': pixel_coords,
                'bbox': (y_coords.min(), y_coords.max(), x_coords.min(), x_coords.max()),
                'area': len(pixel_coords)
            })
            segment_id += 1
    
    return segments


def get_size_bin(area: int, size_bins: Dict[str, Dict] = None) -> str:
    """
    Categorize segment by size.
    
    Args:
        area: Segment area in pixels
        size_bins: Size bin definitions
        
    Returns:
        Size bin name
    """
    if size_bins is None:
        size_bins = {
            'small': {'min': 0, 'max': 100},
            'medium': {'min': 100, 'max': 5000},
            'large': {'min': 5000, 'max': float('inf')}
        }
    
    for bin_name, bounds in size_bins.items():
        if bounds['min'] <= area < bounds['max']:
            return bin_name
    
    return 'unknown'


def compute_class_weights(class_mask: np.ndarray) -> torch.Tensor:
    """
    Compute inverse frequency weights for class balancing.
    
    Args:
        class_mask: Class index mask
        
    Returns:
        Class weights tensor
    """
    unique, counts = np.unique(class_mask, return_counts=True)
    total = counts.sum()
    
    weights = np.ones(len(_class_names), dtype=np.float32)
    for cls_idx, count in zip(unique, counts):
        if count > 0:
            weights[cls_idx] = total / (len(unique) * count)
    
    return torch.from_numpy(weights)
