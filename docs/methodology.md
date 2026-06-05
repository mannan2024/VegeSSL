# VegeSSL Methodology

## Overview

VegeSSL is a multistage semi-supervised contrastive learning framework for detecting mislabeled segments in vegetation classification data from remote sensing imagery.

## Pipeline Stages

### Stage 1: Pixel-Level Unsupervised Contrastive Learning

**Architecture:**
- ResNet-18 encoder (11M parameters, ImageNet pretrained)
- Remove final FC layers, keep convolutional layers
- Output stride: 16
- 1×1 Conv projection head: 512 → 128 dimensions
- Output: H/16 × W/16 × 128 feature map (bilinear upsampled during inference)

**Training Strategy:**
- Extract random 512×512 crops from large images
- Apply augmentations: ColorJitter, RandomFlip, GaussianBlur
- For each pixel at position (i,j):
  - Positive: same pixel position in augmented view
  - Negatives: all other pixels in the batch
- Use mixed precision training (AMP)

**Loss Function:**
Pixel-wise InfoNCE loss computed only on vegetation pixels:

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(z_i \cdot z_j^+ / \tau)}{\sum_{k} \exp(z_i \cdot z_k / \tau)}$$

where $\tau$ is the temperature parameter (default: 0.05).

### Stage 2: Hard Negative Mining

**Process:**
1. Extract embeddings using Stage 1 encoder
2. Sample ~300K vegetation pixels randomly
3. Build KNN index (FAISS) on sampled pixels

**Hard Negative Selection:**
For each pixel with label A, find k=50 nearest neighbors and identify hard negatives as neighbors with different labels at distance percentile 40-70.

**Refined Training:**
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{InfoNCE}} + \lambda \cdot \mathcal{L}_{\text{hard}}$$

where $\lambda = 1.0$ controls the hard negative emphasis.

### Stage 3: Segment-Level Mislabel Detection

**Segment Extraction:**
1. Find connected components in vegetation mask (8-connectivity)
2. Each segment = contiguous pixels with same vegetation label
3. Skip segments smaller than 10 pixels

**Segment Embedding:**
Mean pooling of all pixel embeddings within each segment, followed by L2 normalization.

**Isolation Score Computation:**
For each segment with label A:
1. Find k=50 nearest segment neighbors in embedding space
2. Count neighbors with different labels

$$\text{isolation\_score} = \frac{\text{count}(\text{neighbor\_label} \neq A)}{k}$$

**Threshold Levels:**
| Level | Name | Threshold | Description |
|-------|------|-----------|-------------|
| L1 | Low | 0.3 | Many detections |
| L2 | Medium | 0.5 | Balanced |
| L3 | High | 0.7 | Fewer false positives |
| L4 | Very High | 0.85 | Only strong anomalies |

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch size | 4 | Training batch size |
| Crop size | 512 | Random crop dimensions |
| Embedding dim | 128 | Feature vector size |
| Temperature | 0.05 | InfoNCE temperature |
| Learning rate | 5e-5 | AdamW learning rate |
| Weight decay | 1e-4 | L2 regularization |
| Stage 1 epochs | 100 | Initial training |
| Stage 2 epochs | 100 | Refinement training |
| Early stopping | 20 | Patience epochs |

## Class Configuration

**Vegetation Classes (focus classes):**
- Abandoned land (3)
- Farmland (5)
- Herbaceous (6)
- Broadleaf forest (7)
- Shrubland (8)
- Orchard (9)

**Non-vegetation Classes (ignored):**
- Invalid (0)
- Other (1)
- Water (2)
- Bare land (4)

## References

1. Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020
2. He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
3. Robinson et al., "Contrastive Learning with Hard Negative Samples", ICLR 2021
