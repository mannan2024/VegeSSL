"""
Stage 3: Segment-Level Mislabel Detection

Extracts segment embeddings and computes isolation scores
to identify potentially mislabeled segments.
"""

import os
import sys
import gc
import argparse
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vegessl import PixelContrastiveEncoder
from vegessl.utils import (
    load_checkpoint, save_numpy_checkpoint, load_numpy_checkpoint,
    rgb_to_class_index, get_vegetation_mask, extract_segments, set_seed
)
from configs import load_config


def extract_segment_embedding(model, image, segment, device, tile_size=512):
    """Extract mean embedding for a segment."""
    model.eval()
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    pixel_coords = segment['pixel_coords']
    y_min, y_max, x_min, x_max = segment['bbox']
    
    H, W = image.shape[:2]
    D = model.get_embedding_dim()
    
    all_embeddings = []
    processed = set()
    
    # Process tiles overlapping the segment
    for tile_y in range(max(0, y_min - tile_size), min(H, y_max + 1), tile_size):
        for tile_x in range(max(0, x_min - tile_size), min(W, x_max + 1), tile_size):
            y_end = min(tile_y + tile_size, H)
            x_end = min(tile_x + tile_size, W)
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)
            
            # Check for segment pixels in tile
            in_tile = (
                (pixel_coords[:, 0] >= y_start) & (pixel_coords[:, 0] < y_end) &
                (pixel_coords[:, 1] >= x_start) & (pixel_coords[:, 1] < x_end)
            )
            
            if not in_tile.any():
                continue
            
            tile = image[y_start:y_end, x_start:x_end]
            tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
            tile_tensor = normalize(tile_tensor).unsqueeze(0).to(device)
            
            with torch.no_grad():
                with autocast():
                    emb = model(tile_tensor, upsample=True)
                emb = emb.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            for idx in np.where(in_tile)[0]:
                gy, gx = pixel_coords[idx]
                if (gy, gx) not in processed:
                    processed.add((gy, gx))
                    ly, lx = gy - y_start, gx - x_start
                    all_embeddings.append(emb[ly, lx])
            
            del emb, tile_tensor
    
    if not all_embeddings:
        return np.zeros(D, dtype=np.float32)
    
    # Mean pooling
    embedding = np.mean(all_embeddings, axis=0)
    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
    
    return embedding.astype(np.float32)


def extract_all_segments(model, image_paths, label_paths, device, config):
    """Extract segments and embeddings for all images."""
    all_segments = []
    embeddings = []
    labels = []
    
    print(f"Processing {len(image_paths)} images...")
    
    for img_idx, (img_path, lbl_path) in enumerate(tqdm(zip(image_paths, label_paths))):
        try:
            image = np.array(Image.open(img_path))
            label_rgb = np.array(Image.open(lbl_path))
            class_mask = rgb_to_class_index(label_rgb)
            
            segments = extract_segments(
                class_mask,
                min_segment_size=config.detection.min_segment_size
            )
            
            for seg in segments:
                emb = extract_segment_embedding(model, image, seg, device)
                
                seg['embedding'] = emb
                seg['image_idx'] = img_idx
                seg['image_path'] = img_path
                seg['label_path'] = lbl_path
                
                all_segments.append(seg)
                embeddings.append(emb)
                labels.append(seg['class_idx'])
            
            if (img_idx + 1) % 20 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    embeddings = np.array(embeddings, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    
    print(f"Extracted {len(all_segments):,} segments")
    return all_segments, embeddings, labels


def compute_isolation_scores(embeddings, labels, k=50):
    """Compute isolation scores based on neighbor label consistency."""
    print(f"Computing isolation scores with k={k}...")
    
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(embeddings)), metric='cosine', n_jobs=-1)
    nn.fit(embeddings)
    
    distances, indices = nn.kneighbors(embeddings)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    
    isolation_scores = []
    neighbor_info = []
    
    for i in range(len(embeddings)):
        anchor_label = labels[i]
        neighbor_labels = labels[indices[i]]
        
        # Fraction of neighbors with different label
        different = (neighbor_labels != anchor_label).sum()
        score = different / len(neighbor_labels)
        
        isolation_scores.append(score)
        neighbor_info.append({
            'anchor_label': anchor_label,
            'neighbor_labels': neighbor_labels,
            'neighbor_distances': distances[i],
            'neighbor_indices': indices[i]
        })
    
    isolation_scores = np.array(isolation_scores, dtype=np.float32)
    
    print(f"Isolation scores - Mean: {isolation_scores.mean():.4f}, Max: {isolation_scores.max():.4f}")
    return isolation_scores, neighbor_info


def main(args):
    config = load_config(args.config)
    set_seed(config.training.random_seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(config.output.tables_dir, exist_ok=True)
    os.makedirs(config.output.errors_dir, exist_ok=True)
    
    # Load refined model
    checkpoint = load_checkpoint("refined_pixel_encoder", config.output.checkpoints_dir, device)
    if checkpoint is None:
        checkpoint = load_checkpoint("pixel_encoder_stage1", config.output.checkpoints_dir, device)
        if checkpoint is None:
            raise RuntimeError("No trained model found. Run training first.")
        print("Using Stage 1 model (Stage 2 not found)")
    else:
        print("Using refined Stage 2 model")
    
    model = PixelContrastiveEncoder(embedding_dim=config.training.embedding_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    test_images = sorted(glob(os.path.join(config.data.test_image_path, "*.png")))
    test_labels = sorted(glob(os.path.join(config.data.test_label_path, "*.png")))
    
    print(f"Found {len(test_images)} test images")
    
    # Check for cached segments
    cached = load_numpy_checkpoint("segment_embeddings", config.output.checkpoints_dir)
    
    if cached is not None:
        print("Loading cached segment data...")
        data = cached.item()
        all_segments = data['segments']
        embeddings = data['embeddings']
        labels = data['labels']
    else:
        all_segments, embeddings, labels = extract_all_segments(
            model, test_images, test_labels, device, config
        )
        
        save_numpy_checkpoint({
            'segments': all_segments,
            'embeddings': embeddings,
            'labels': labels
        }, "segment_embeddings", config.output.checkpoints_dir)
    
    # Compute isolation scores
    isolation_scores, neighbor_info = compute_isolation_scores(
        embeddings, labels, k=config.detection.knn_neighbors
    )
    
    # Add scores to segments
    for i, seg in enumerate(all_segments):
        seg['isolation_score'] = isolation_scores[i]
        seg['neighbor_info'] = neighbor_info[i]
    
    # Export results
    results = []
    for seg in all_segments:
        results.append({
            'image': os.path.basename(seg['image_path']),
            'segment_id': seg['label'],
            'class_idx': seg['class_idx'],
            'class_name': seg['class_name'],
            'isolation_score': seg['isolation_score'],
            'area_pixels': seg['area'],
            'bbox_y_min': seg['bbox'][0],
            'bbox_y_max': seg['bbox'][1],
            'bbox_x_min': seg['bbox'][2],
            'bbox_x_max': seg['bbox'][3]
        })
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(config.output.tables_dir, "all_segments_analysis.csv"), index=False)
    
    # Export suspicious segments at each threshold
    thresholds = config.detection.thresholds
    for level_name in ['level1', 'level2', 'level3', 'level4']:
        level = getattr(thresholds, level_name)
        threshold = level.value
        
        suspicious = df[df['isolation_score'] >= threshold].sort_values(
            'isolation_score', ascending=False
        )
        
        output_path = os.path.join(config.output.tables_dir, f"suspicious_{level.name}.csv")
        suspicious.to_csv(output_path, index=False)
        
        print(f"Level {level.name} (≥{threshold}): {len(suspicious):,} suspicious segments")
    
    print("\n" + "=" * 60)
    print("Mislabel detection complete!")
    print(f"Results saved to: {config.output.tables_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Mislabel Detection")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    main(args)
