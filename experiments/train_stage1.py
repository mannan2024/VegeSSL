"""
Stage 1: Pixel-Level Contrastive Learning

This script trains the pixel-level encoder using contrastive learning
on vegetation pixels from remote sensing images.
"""

import os
import sys
import argparse
from glob import glob

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vegessl import PixelContrastiveEncoder, PixelInfoNCELoss, ContrastiveCropDataset
from vegessl.utils import set_seed, save_checkpoint, load_checkpoint
from configs import load_config


def train_epoch(model, dataloader, optimizer, loss_fn, scaler, device):
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        view1 = batch['view1'].to(device)
        view2 = batch['view2'].to(device)
        veg_mask = batch['veg_mask'].to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            z1 = model(view1, upsample=False)
            z2 = model(view2, upsample=False)
            
            # Downsample mask to match embedding resolution
            veg_mask_down = torch.nn.functional.interpolate(
                veg_mask.float().unsqueeze(1),
                size=z1.shape[2:],
                mode='nearest'
            ).squeeze(1).bool()
            
            loss = loss_fn(z1, z2, veg_mask_down)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def validate_epoch(model, dataloader, loss_fn, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            view1 = batch['view1'].to(device)
            view2 = batch['view2'].to(device)
            veg_mask = batch['veg_mask'].to(device)
            
            with torch.cuda.amp.autocast():
                z1 = model(view1, upsample=False)
                z2 = model(view2, upsample=False)
                
                veg_mask_down = torch.nn.functional.interpolate(
                    veg_mask.float().unsqueeze(1),
                    size=z1.shape[2:],
                    mode='nearest'
                ).squeeze(1).bool()
                
                loss = loss_fn(z1, z2, veg_mask_down)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    set_seed(config.training.random_seed)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(config.output.checkpoints_dir, exist_ok=True)
    
    # Load data paths
    train_images = sorted(glob(os.path.join(config.data.train_image_path, "*.png")))
    train_labels = sorted(glob(os.path.join(config.data.train_label_path, "*.png")))
    
    print(f"Found {len(train_images)} training images")
    
    # Split into train/validation
    num_val = int(len(train_images) * config.training.validation_split)
    val_images = train_images[:num_val]
    val_labels = train_labels[:num_val]
    train_images = train_images[num_val:]
    train_labels = train_labels[num_val:]
    
    print(f"Training: {len(train_images)} | Validation: {len(val_images)}")
    
    # Create datasets
    train_dataset = ContrastiveCropDataset(
        image_paths=train_images,
        label_paths=train_labels,
        crop_size=config.training.crop_size,
        crops_per_image=2
    )
    
    val_dataset = ContrastiveCropDataset(
        image_paths=val_images,
        label_paths=val_labels,
        crop_size=config.training.crop_size,
        crops_per_image=1
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    # Check for existing checkpoint
    checkpoint = load_checkpoint(
        "pixel_encoder_stage1",
        config.output.checkpoints_dir,
        device
    )
    
    # Initialize model
    model = PixelContrastiveEncoder(
        embedding_dim=config.training.embedding_dim
    ).to(device)
    
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Resumed from epoch {checkpoint['epoch']}")
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
    else:
        start_epoch = 0
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
    
    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    loss_fn = PixelInfoNCELoss(temperature=config.training.temperature)
    scaler = GradScaler()
    
    patience_counter = 0
    patience = config.training.stage1.patience
    num_epochs = config.training.stage1.epochs
    
    print(f"\nStarting Stage 1 training for {num_epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, scaler, device)
        train_losses.append(train_loss)
        
        val_loss = validate_epoch(model, val_loader, loss_fn, device)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'best_val_loss': best_val_loss
            }, "pixel_encoder_stage1", config.output.checkpoints_dir)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break
    
    print("\n" + "=" * 60)
    print("Stage 1 training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Pixel-Level Contrastive Learning")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    main(args)
