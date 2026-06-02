"""
Stage 2: Hard Negative Mining and Encoder Refinement

This script refines the encoder by focusing on hard negative pairs -
pixels that are similar in embedding space but have different labels.
"""

import os
import sys
import gc
import argparse
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    from sklearn.neighbors import NearestNeighbors
    FAISS_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vegessl import (
    PixelContrastiveEncoder,
    PixelInfoNCELoss,
    HardNegativeInfoNCELoss,
    ContrastiveCropDataset
)
from vegessl.utils import (
    set_seed, save_checkpoint, load_checkpoint,
    save_numpy_checkpoint, load_numpy_checkpoint,
    rgb_to_class_index, get_vegetation_mask
)
from configs import load_config


def extract_vegetation_embeddings(model, image_path, label_path, device, tile_size=512):
    """Extract embeddings only for vegetation pixels."""
    model.eval()
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    image = np.array(Image.open(image_path))
    label_rgb = np.array(Image.open(label_path))
    class_mask = rgb_to_class_index(label_rgb)
    veg_mask = get_vegetation_mask(class_mask)
    
    H, W = image.shape[:2]
    D = model.get_embedding_dim()
    
    veg_y, veg_x = np.where(veg_mask)
    if len(veg_y) == 0:
        return np.zeros((0, D), dtype=np.float32), np.zeros(0, dtype=np.int32)
    
    all_embeddings = []
    all_labels = []
    processed = set()
    
    with torch.no_grad():
        for y in range(0, H, tile_size):
            for x in range(0, W, tile_size):
                y_end = min(y + tile_size, H)
                x_end = min(x + tile_size, W)
                
                tile_veg = veg_mask[y:y_end, x:x_end]
                if not tile_veg.any():
                    continue
                
                tile = image[y:y_end, x:x_end]
                tile_tensor = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
                tile_tensor = normalize(tile_tensor).unsqueeze(0).to(device)
                
                with autocast():
                    emb = model(tile_tensor, upsample=True)
                emb = emb.squeeze(0).permute(1, 2, 0).cpu().numpy()
                
                local_y, local_x = np.where(tile_veg)
                for ly, lx in zip(local_y, local_x):
                    gy, gx = y + ly, x + lx
                    if (gy, gx) not in processed:
                        processed.add((gy, gx))
                        all_embeddings.append(emb[ly, lx])
                        all_labels.append(class_mask[gy, gx])
                
                del emb, tile_tensor
        
        torch.cuda.empty_cache()
    
    if not all_embeddings:
        return np.zeros((0, D), dtype=np.float32), np.zeros(0, dtype=np.int32)
    
    embeddings = np.array(all_embeddings, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int32)
    
    # L2 normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-8)
    
    return embeddings, labels


def sample_vegetation_pixels(model, image_paths, label_paths, device, num_samples, max_images=50):
    """Sample vegetation pixels for hard negative mining."""
    all_embeddings = []
    all_labels = []
    
    paths = list(zip(image_paths[:max_images], label_paths[:max_images]))
    samples_per_image = max(1000, num_samples // len(paths))
    
    print(f"Sampling from {len(paths)} images...")
    
    for img_path, lbl_path in tqdm(paths):
        try:
            emb, lbl = extract_vegetation_embeddings(model, img_path, lbl_path, device)
            
            if len(emb) == 0:
                continue
            
            n_sample = min(len(emb), samples_per_image)
            indices = np.random.choice(len(emb), size=n_sample, replace=False)
            
            all_embeddings.extend(emb[indices])
            all_labels.extend(lbl[indices])
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    embeddings = np.array(all_embeddings, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int32)
    
    # Subsample to target
    if len(embeddings) > num_samples:
        indices = np.random.choice(len(embeddings), size=num_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
    
    print(f"Sampled {len(embeddings):,} vegetation pixels")
    return embeddings, labels


def find_hard_negatives(embeddings, labels, k=50, percentile_low=40, percentile_high=70):
    """Find hard negative pairs using KNN."""
    print(f"Building KNN index for {len(embeddings):,} samples...")
    
    if FAISS_AVAILABLE:
        d = embeddings.shape[1]
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)
        distances, indices = index.search(embeddings, k + 1)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
    else:
        nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine', n_jobs=-1)
        nn.fit(embeddings)
        distances, indices = nn.kneighbors(embeddings)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
    
    print("Finding hard negative pairs...")
    hard_neg_pairs = []
    
    for i in tqdm(range(len(embeddings))):
        anchor_label = labels[i]
        neighbor_labels = labels[indices[i]]
        neighbor_dists = distances[i]
        
        diff_mask = neighbor_labels != anchor_label
        if not np.any(diff_mask):
            continue
        
        diff_dists = neighbor_dists[diff_mask]
        diff_indices = indices[i][diff_mask]
        
        if len(diff_dists) > 0:
            p_low = np.percentile(neighbor_dists, percentile_low)
            p_high = np.percentile(neighbor_dists, percentile_high)
            valid = (diff_dists >= p_low) & (diff_dists <= p_high)
            
            for neg_idx in diff_indices[valid]:
                hard_neg_pairs.append((i, neg_idx))
    
    print(f"Found {len(hard_neg_pairs):,} hard negative pairs")
    return hard_neg_pairs


def train_stage2_epoch(model, dataloader, optimizer, loss_fn, scaler, device,
                       hard_neg_embeddings, hard_neg_labels):
    """Train one epoch with hard negative emphasis."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training Stage 2")
    for batch in pbar:
        view1 = batch['view1'].to(device)
        view2 = batch['view2'].to(device)
        veg_mask = batch['veg_mask'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            z1 = model(view1, upsample=False)
            z2 = model(view2, upsample=False)
            
            veg_mask_down = F.interpolate(
                veg_mask.float().unsqueeze(1),
                size=z1.shape[2:],
                mode='nearest'
            ).squeeze(1).bool()
            
            loss = loss_fn(z1, z2, veg_mask_down, hard_neg_embeddings, hard_neg_labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def main(args):
    config = load_config(args.config)
    set_seed(config.training.random_seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    os.makedirs(config.output.checkpoints_dir, exist_ok=True)
    
    # Load Stage 1 model
    checkpoint = load_checkpoint("pixel_encoder_stage1", config.output.checkpoints_dir, device)
    if checkpoint is None:
        raise RuntimeError("Stage 1 checkpoint not found. Run train_stage1.py first.")
    
    model = PixelContrastiveEncoder(embedding_dim=config.training.embedding_dim).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Loaded Stage 1 encoder")
    
    # Check for existing Stage 2 checkpoint
    stage2_ckpt = load_checkpoint("refined_pixel_encoder", config.output.checkpoints_dir, device)
    
    if stage2_ckpt is not None:
        print("Stage 2 already complete. Loading refined model...")
        model.load_state_dict(stage2_ckpt['model_state_dict'])
        return
    
    # Load data
    train_images = sorted(glob(os.path.join(config.data.train_image_path, "*.png")))
    train_labels = sorted(glob(os.path.join(config.data.train_label_path, "*.png")))
    
    num_val = int(len(train_images) * config.training.validation_split)
    train_images = train_images[num_val:]
    train_labels = train_labels[num_val:]
    
    # Sample embeddings or load from cache
    sampled_data = load_numpy_checkpoint("sampled_embeddings", config.output.checkpoints_dir)
    
    if sampled_data is not None:
        print("Loading cached embeddings...")
        data = sampled_data.item()
        sampled_embeddings = data['embeddings']
        sampled_labels = data['labels']
    else:
        print("Extracting and sampling vegetation pixels...")
        gc.collect()
        torch.cuda.empty_cache()
        
        sampled_embeddings, sampled_labels = sample_vegetation_pixels(
            model, train_images, train_labels, device,
            num_samples=config.training.stage2.num_sample_pixels,
            max_images=50
        )
        
        save_numpy_checkpoint({
            'embeddings': sampled_embeddings,
            'labels': sampled_labels
        }, "sampled_embeddings", config.output.checkpoints_dir)
    
    # Find hard negatives
    hard_neg_data = load_numpy_checkpoint("hard_negatives", config.output.checkpoints_dir)
    
    if hard_neg_data is not None:
        hard_neg_pairs = hard_neg_data.tolist()
    else:
        hard_neg_pairs = find_hard_negatives(
            sampled_embeddings, sampled_labels,
            k=config.training.stage2.knn_neighbors,
            percentile_low=config.training.stage2.percentile_low,
            percentile_high=config.training.stage2.percentile_high
        )
        save_numpy_checkpoint(np.array(hard_neg_pairs), "hard_negatives", config.output.checkpoints_dir)
    
    neg_indices = list(set([p[1] for p in hard_neg_pairs]))
    hard_neg_embeddings = sampled_embeddings[neg_indices]
    hard_neg_labels = sampled_labels[neg_indices]
    print(f"Unique hard negatives: {len(hard_neg_embeddings):,}")
    
    # Create dataloader
    train_dataset = ContrastiveCropDataset(
        image_paths=train_images,
        label_paths=train_labels,
        crop_size=config.training.crop_size,
        crops_per_image=2
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    loss_fn = HardNegativeInfoNCELoss(
        temperature=config.training.temperature,
        lambda_hard=config.training.stage2.lambda_hard_neg
    )
    scaler = GradScaler()
    
    val_loss_fn = PixelInfoNCELoss(config.training.temperature)
    
    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    
    num_epochs = config.training.stage2.epochs
    patience = config.training.stage2.patience
    
    print(f"\nStarting Stage 2 training for {num_epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss = train_stage2_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device,
            hard_neg_embeddings, hard_neg_labels
        )
        train_losses.append(train_loss)
        
        print(f"Train Loss: {train_loss:.4f}")
        
        if train_loss < best_loss:
            best_loss = train_loss
            patience_counter = 0
            
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'best_loss': best_loss
            }, "refined_pixel_encoder", config.output.checkpoints_dir)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
    
    print("\n" + "=" * 60)
    print("Stage 2 training complete!")
    print(f"Best training loss: {best_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Hard Negative Mining")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    main(args)
