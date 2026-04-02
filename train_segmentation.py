"""
Segmentation Training Script
Converted from train_mask.ipynb
Trains a segmentation head on top of DINOv2 backbone

CHANGES vs original:
  - n_epochs set to 2
  - Backbone upgraded: small -> base (vitb14_reg) for richer features
  - Deeper SegmentationHead with 3 ConvNeXt blocks + channel expansion (128->256)
  - Optimizer: SGD -> AdamW (faster convergence, better generalisation)
  - Scheduler: cosine annealing LR over all steps (smooth decay)
  - Loss: CrossEntropyLoss with label_smoothing=0.1
  - Mixed-precision training via torch.amp (faster GPU utilisation)
  - Best model checkpoint saved at peak val IoU

BUGFIX:
  - GradScaler: torch.cuda.GradScaler -> version-safe torch.amp.GradScaler
    (torch.cuda.GradScaler was deprecated in PyTorch 2.3 and removed in 2.4+)
"""

import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import torchvision
from tqdm import tqdm

plt.switch_backend('Agg')


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration class for training hyperparameters."""
    def __init__(self):
        self.n_epochs = 2
        self.batch_size = 2
        self.patch_width = int(((960 / 2) // 14) * 14)   # 672
        self.patch_height = int(((540 / 2) // 14) * 14)  # 378
        self.lr = 3e-4
        self.weight_decay = 1e-4
        self.label_smoothing = 0.1
        self.grad_clip_norm = 1.0
        self.scheduler_eta_min = 1e-6


# ============================================================================
# Utility Functions
# ============================================================================

def save_image(img, filename):
    """Save an image tensor to file after denormalizing."""
    img  = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = np.moveaxis(img, 0, -1)
    img  = (img * std + mean) * 255
    cv2.imwrite(filename, img[:, :, ::-1])


# ============================================================================
# Mask Conversion
# ============================================================================

value_map = {
    0: 0,        # background
    100: 1,      # Trees
    200: 2,      # Lush Bushes
    300: 3,      # Dry Grass
    500: 4,      # Dry Bushes
    550: 5,      # Ground Clutter
    700: 6,      # Logs
    800: 7,      # Rocks
    7100: 8,     # Landscape
    10000: 9     # Sky
}
n_classes = len(value_map)


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr     = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir      = os.path.join(data_dir, 'Color_Images')
        self.masks_dir      = os.path.join(data_dir, 'Segmentation')
        self.transform      = transform
        self.mask_transform = mask_transform
        self.data_ids       = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        image   = Image.open(os.path.join(self.image_dir, data_id)).convert("RGB")
        mask    = Image.open(os.path.join(self.masks_dir, data_id))
        mask    = convert_mask(mask)
        if self.transform:
            image = self.transform(image)
            mask  = self.mask_transform(mask) * 255
        return image, mask


# ============================================================================
# Model: Deeper Segmentation Head (ConvNeXt-style, 3 blocks, 256 channels)
# ============================================================================

class ConvNeXtBlock(nn.Module):
    """Single depthwise-separable ConvNeXt block."""
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        padding   = kernel_size // 2
        self.dw   = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                              padding=padding, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.pw1  = nn.Linear(channels, channels * 4)
        self.act  = nn.GELU()
        self.pw2  = nn.Linear(channels * 4, channels)

    def forward(self, x):
        residual = x
        x = self.dw(x)                   # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)       # (B, H, W, C)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = x.permute(0, 3, 1, 2)       # (B, C, H, W)
        return x + residual


class SegmentationHeadConvNeXt(nn.Module):
    """
    Deeper segmentation head:
      - stem projects backbone dim -> 256
      - 3 ConvNeXt blocks at 256 channels
      - 1x1 classifier
    """
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        hidden = 256  # increased from 128

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.GELU()
        )
        self.blocks = nn.Sequential(
            ConvNeXtBlock(hidden, kernel_size=7),
            ConvNeXtBlock(hidden, kernel_size=7),
            ConvNeXtBlock(hidden, kernel_size=7),
        )
        self.classifier = nn.Conv2d(hidden, out_channels, kernel_size=1)

    def forward(self, x):
        B, N, C = x.shape
        x = x.reshape(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.stem(x)
        x = self.blocks(x)
        return self.classifier(x)


# ============================================================================
# Metrics
# ============================================================================

def compute_iou(pred, target, num_classes=10, ignore_index=255):
    """Compute IoU for each class and return mean IoU."""
    pred         = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)
    iou_per_class = []
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue
        pred_inds    = pred == class_id
        target_inds  = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        union        = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())
    return np.nanmean(iou_per_class)


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    """Compute Dice coefficient (F1 Score) per class and return mean Dice Score."""
    pred         = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)
    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds    = pred == class_id
        target_inds  = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        dice_score   = (2. * intersection + smooth) / (
            pred_inds.sum().float() + target_inds.sum().float() + smooth)
        dice_per_class.append(dice_score.cpu().numpy())
    return np.mean(dice_per_class)


def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    return (torch.argmax(pred, dim=1) == target).float().mean().cpu().numpy()


def evaluate_metrics(model, backbone, data_loader, device, num_classes=10, show_progress=True):
    """Evaluate all metrics on a dataset (with AMP for speed)."""
    iou_scores, dice_scores, pixel_accuracies = [], [], []
    model.eval()
    loader = tqdm(data_loader, desc="Evaluating", leave=False, unit="batch") if show_progress else data_loader
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                output  = backbone.forward_features(imgs)["x_norm_patchtokens"]
                logits  = model(output)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
            labels = labels.squeeze(dim=1).long()
            iou_scores.append(compute_iou(outputs, labels, num_classes=num_classes))
            dice_scores.append(compute_dice(outputs, labels, num_classes=num_classes))
            pixel_accuracies.append(compute_pixel_accuracy(outputs, labels))
    model.train()
    return np.mean(iou_scores), np.mean(dice_scores), np.mean(pixel_accuracies)


# ============================================================================
# Plotting Functions
# ============================================================================

def save_training_plots(history, output_dir):
    """Save all training metric plots to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: Loss + Pixel Accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'],   label='val')
    plt.title('Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history['train_pixel_acc'], label='train')
    plt.plot(history['val_pixel_acc'],   label='val')
    plt.title('Pixel Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()
    print(f"Saved training curves to '{output_dir}/training_curves.png'")

    # Plot 2: IoU
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.title('Train IoU vs Epoch'); plt.xlabel('Epoch'); plt.ylabel('IoU'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history['val_iou'], label='Val IoU')
    plt.title('Validation IoU vs Epoch'); plt.xlabel('Epoch'); plt.ylabel('IoU'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'iou_curves.png'))
    plt.close()
    print(f"Saved IoU curves to '{output_dir}/iou_curves.png'")

    # Plot 3: Dice
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.title('Train Dice vs Epoch'); plt.xlabel('Epoch'); plt.ylabel('Dice Score'); plt.legend(); plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history['val_dice'], label='Val Dice')
    plt.title('Validation Dice vs Epoch'); plt.xlabel('Epoch'); plt.ylabel('Dice Score'); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_curves.png'))
    plt.close()
    print(f"Saved Dice curves to '{output_dir}/dice_curves.png'")

    # Plot 4: Combined
    plt.figure(figsize=(12, 10))
    for i, (train_key, val_key, title, ylabel) in enumerate([
        ('train_loss',      'val_loss',      'Loss vs Epoch',           'Loss'),
        ('train_iou',       'val_iou',       'IoU vs Epoch',            'IoU'),
        ('train_dice',      'val_dice',      'Dice Score vs Epoch',     'Dice Score'),
        ('train_pixel_acc', 'val_pixel_acc', 'Pixel Accuracy vs Epoch', 'Pixel Accuracy'),
    ]):
        plt.subplot(2, 2, i + 1)
        plt.plot(history[train_key], label='train')
        plt.plot(history[val_key],   label='val')
        plt.title(title); plt.xlabel('Epoch'); plt.ylabel(ylabel); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_curves.png'))
    plt.close()
    print(f"Saved combined metrics curves to '{output_dir}/all_metrics_curves.png'")


def save_history_to_file(history, output_dir):
    """Save training history to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(filepath, 'w') as f:
        f.write("TRAINING RESULTS\n" + "=" * 50 + "\n\n")
        f.write("Final Metrics:\n")
        for label, key in [
            ("Train Loss",     "train_loss"),     ("Val Loss",     "val_loss"),
            ("Train IoU",      "train_iou"),      ("Val IoU",      "val_iou"),
            ("Train Dice",     "train_dice"),     ("Val Dice",     "val_dice"),
            ("Train Accuracy", "train_pixel_acc"),("Val Accuracy", "val_pixel_acc"),
        ]:
            f.write(f"  Final {label:<16}: {history[key][-1]:.4f}\n")
        f.write("=" * 50 + "\n\nBest Results:\n")
        f.write(f"  Best Val IoU:      {max(history['val_iou']):.4f} (Epoch {np.argmax(history['val_iou'])+1})\n")
        f.write(f"  Best Val Dice:     {max(history['val_dice']):.4f} (Epoch {np.argmax(history['val_dice'])+1})\n")
        f.write(f"  Best Val Accuracy: {max(history['val_pixel_acc']):.4f} (Epoch {np.argmax(history['val_pixel_acc'])+1})\n")
        f.write(f"  Lowest Val Loss:   {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss'])+1})\n")
        f.write("=" * 50 + "\n\nPer-Epoch History:\n" + "-" * 100 + "\n")
        headers = ['Epoch', 'Train Loss', 'Val Loss', 'Train IoU', 'Val IoU',
                   'Train Dice', 'Val Dice', 'Train Acc', 'Val Acc']
        f.write("{:<8} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(*headers))
        f.write("-" * 100 + "\n")
        for i in range(len(history['train_loss'])):
            f.write("{:<8} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}\n".format(
                i + 1,
                history['train_loss'][i],      history['val_loss'][i],
                history['train_iou'][i],       history['val_iou'][i],
                history['train_dice'][i],      history['val_dice'][i],
                history['train_pixel_acc'][i], history['val_pixel_acc'][i]
            ))
    print(f"Saved evaluation metrics to {filepath}")


# ============================================================================
# Training and Validation Functions
# ============================================================================

def train_one_epoch(classifier, backbone_model, train_loader, loss_fct, optimizer, scheduler, scaler, device, use_amp):
    """Train the model for one epoch."""
    classifier.train()
    train_losses = []
    train_pbar = tqdm(train_loader, desc="Training", leave=False, unit="batch")
    
    for imgs, labels in train_pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        
        with torch.autocast(device_type=device.type, enabled=use_amp):
            with torch.no_grad():
                output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
            logits = classifier(output)
            outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
            loss = loss_fct(outputs, labels.squeeze(1).long())
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=config.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        
        train_losses.append(loss.item())
        train_pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    
    return np.mean(train_losses)


def validate_one_epoch(classifier, backbone_model, val_loader, loss_fct, device, use_amp):
    """Validate the model for one epoch."""
    classifier.eval()
    val_losses = []
    val_pbar = tqdm(val_loader, desc="Validating", leave=False, unit="batch")
    
    with torch.no_grad():
        for imgs, labels in val_pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                output = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
                logits = classifier(output)
                outputs = F.interpolate(logits, size=imgs.shape[2:], mode="bilinear", align_corners=False)
                loss = loss_fct(outputs, labels.squeeze(1).long())
            val_losses.append(loss.item())
            val_pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    return np.mean(val_losses)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Configuration ────────────────────────────────────────────────────────
    config = Config()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'train_stats')
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((config.patch_height, config.patch_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((config.patch_height, config.patch_width)),
        transforms.ToTensor(),
    ])

    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--data-root', type=str, default=script_dir,
                        help='Root folder containing train/ and val/ directories')
    parser.add_argument('--train-dir', type=str, default='train',
                        help='Train data folder name or absolute path')
    parser.add_argument('--val-dir',   type=str, default='val',
                        help='Validation data folder name or absolute path')
    args = parser.parse_args()

    def resolve_path(base, path):
        return path if os.path.isabs(path) else os.path.abspath(os.path.join(base, path))

    data_dir = resolve_path(args.data_root, args.train_dir)
    val_dir  = resolve_path(args.data_root, args.val_dir)
    print(f"Using train data directory: {data_dir}")
    print(f"Using val data directory:   {val_dir}")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Train directory not found: {data_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Validation directory not found: {val_dir}")

    trainset     = MaskDataset(data_dir=data_dir, transform=transform, mask_transform=mask_transform)
    train_loader = DataLoader(trainset, batch_size=config.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    valset       = MaskDataset(data_dir=val_dir, transform=transform, mask_transform=mask_transform)
    val_loader   = DataLoader(valset, batch_size=config.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)
    print(f"Training samples: {len(trainset)}  |  Validation samples: {len(valset)}")

    # ── Backbone ──────────────────────────────────────────────────────────────
    print("Loading DINOv2 backbone (base)...")
    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vitb14_reg")
    backbone_model.eval()
    backbone_model.to(device)
    print("Backbone loaded successfully!")

    imgs, _ = next(iter(train_loader))
    with torch.no_grad():
        output = backbone_model.forward_features(imgs.to(device))["x_norm_patchtokens"]
    n_embedding = output.shape[2]
    print(f"Embedding dimension: {n_embedding}  |  Patch tokens: {output.shape}")

    # ── Segmentation head ─────────────────────────────────────────────────────
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=config.patch_width // 14,
        tokenH=config.patch_height // 14
    ).to(device)

    # ── Loss / optimiser / scheduler ─────────────────────────────────────────
    loss_fct    = torch.nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer   = optim.AdamW(classifier.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    total_steps = config.n_epochs * len(train_loader)
    scheduler   = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=config.scheduler_eta_min)

    # ── BUGFIX: version-safe GradScaler ──────────────────────────────────────
    # torch.cuda.GradScaler was deprecated in PyTorch 2.3 and removed in 2.4+.
    # torch.amp.GradScaler (with explicit device string) is the correct API.
    use_amp = device.type == "cuda"
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)   # PyTorch >= 2.3
    else:
        scaler = torch.cuda.GradScaler(enabled=use_amp)           # PyTorch <  2.3

    # ── History & checkpointing ───────────────────────────────────────────────
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou':  [], 'val_iou':  [],
        'train_dice': [], 'val_dice': [],
        'train_pixel_acc': [], 'val_pixel_acc': []
    }
    best_val_iou    = -1.0
    best_model_path = os.path.join(script_dir, "segmentation_head_best.pth")

    print("\nStarting training...")
    print(f"  Epochs:     {config.n_epochs}  |  Backbone: dinov2_vitb14_reg  |  Head: 256ch x 3 blocks")
    print(f"  Optimizer:  AdamW lr={config.lr} wd={config.weight_decay}  |  Scheduler: CosineAnnealingLR  |  AMP: {use_amp}")
    print("=" * 80)

    epoch_pbar = tqdm(range(config.n_epochs), desc="Training", unit="epoch")
    for epoch in epoch_pbar:

        # ── Training phase ────────────────────────────────────────────────────
        epoch_train_loss = train_one_epoch(
            classifier, backbone_model, train_loader, loss_fct, 
            optimizer, scheduler, scaler, device, use_amp
        )

        # ── Validation phase ──────────────────────────────────────────────────
        epoch_val_loss = validate_one_epoch(
            classifier, backbone_model, val_loader, loss_fct, device, use_amp
        )

        # ── Metrics ───────────────────────────────────────────────────────────
        train_iou, train_dice, train_pixel_acc = evaluate_metrics(
            classifier, backbone_model, train_loader, device, num_classes=n_classes)
        val_iou, val_dice, val_pixel_acc = evaluate_metrics(
            classifier, backbone_model, val_loader,   device, num_classes=n_classes)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['train_iou'].append(train_iou);        history['val_iou'].append(val_iou)
        history['train_dice'].append(train_dice);      history['val_dice'].append(val_dice)
        history['train_pixel_acc'].append(train_pixel_acc)
        history['val_pixel_acc'].append(val_pixel_acc)

        epoch_pbar.set_postfix(
            train_loss=f"{epoch_train_loss:.3f}", val_loss=f"{epoch_val_loss:.3f}",
            val_iou=f"{val_iou:.3f}",             val_acc=f"{val_pixel_acc:.3f}"
        )

        # ── Best checkpoint ───────────────────────────────────────────────────
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(classifier.state_dict(), best_model_path)
            print(f"\n  ✓ New best val IoU: {best_val_iou:.4f} — saved to '{best_model_path}'")

    # ── Save artefacts ────────────────────────────────────────────────────────
    print("\nSaving training curves...")
    save_training_plots(history, output_dir)
    save_history_to_file(history, output_dir)

    model_path = os.path.join(script_dir, "segmentation_head.pth")
    torch.save(classifier.state_dict(), model_path)
    print(f"Saved final model to '{model_path}'")
    print(f"Saved best  model to '{best_model_path}'")

    print("\nFinal evaluation results:")
    print(f"  Final Val Loss:     {history['val_loss'][-1]:.4f}")
    print(f"  Final Val IoU:      {history['val_iou'][-1]:.4f}")
    print(f"  Final Val Dice:     {history['val_dice'][-1]:.4f}")
    print(f"  Final Val Accuracy: {history['val_pixel_acc'][-1]:.4f}")
    print(f"  Best  Val IoU:      {best_val_iou:.4f}")
    print("\nTraining complete!")


if __name__ == "__main__":
    main()