"""
VegeSSL: Semi-Supervised Contrastive Learning for Vegetation Mislabel Detection

A multistage semi-supervised contrastive learning framework for identifying
noisy labels in remote sensing vegetation classification data.
"""

from .models import PixelContrastiveEncoder
from .losses import PixelInfoNCELoss, HardNegativeInfoNCELoss
from .datasets import ContrastiveCropDataset
from .utils import (
    save_checkpoint,
    load_checkpoint,
    rgb_to_class_index,
    class_index_to_rgb,
    get_vegetation_mask,
    extract_segments,
    set_seed,
)

__version__ = "1.0.0"
__author__ = "VegeSSL Authors"

__all__ = [
    "PixelContrastiveEncoder",
    "PixelInfoNCELoss",
    "HardNegativeInfoNCELoss",
    "ContrastiveCropDataset",
    "save_checkpoint",
    "load_checkpoint",
    "rgb_to_class_index",
    "class_index_to_rgb",
    "get_vegetation_mask",
    "extract_segments",
    "set_seed",
]
