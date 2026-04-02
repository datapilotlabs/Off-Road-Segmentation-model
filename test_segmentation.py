"""
Segmentation Validation Script
Converted from val_mask.ipynb
Evaluates a trained segmentation head on validation data and saves predictions

CHANGES vs original (must mirror train_mask.py changes):
  - Backbone upgraded: small (vits14) -> base (vitb14_reg)  [matches training]
  - SegmentationHeadConvNeXt replaced with deeper version:
      * ConvNeXtBlock class (proper depthwise + LayerNorm + MLP)
      * 3 stacked blocks at 256 channels instead of 1 block at 128
      * stem now uses kernel_size=1 projection (matches training head)
  - AMP (torch.autocast) added to forward passes for speed
  - Best-model checkpoint flag: --use_best_model loads segmentation_head_best.pth
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import cv2
import os
import argparse
from tqdm import tqdm

# Set matplotlib to non-interactive backend
plt.switch_backend('Agg')


# ============================================================================
# Utility Functions
# ============================================================================

def save_image(img, filename):
    """Save an image tensor to file after denormalizing."""
    img = np.array(img)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = (img * std + mean) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(filename, img[:, :, ::-1])


# ============================================================================
# Mask Conversion
# ============================================================================

# Mapping from raw pixel values to new class IDs
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

# Class names for visualization
class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

n_classes = len(value_map)

# Color palette for visualization (10 distinct colors)
color_palette = np.array([
    [0, 0, 0],        # Background - black
    [34, 139, 34],    # Trees - forest green
    [0, 255, 0],      # Lush Bushes - lime
    [210, 180, 140],  # Dry Grass - tan
    [139, 90, 43],    # Dry Bushes - brown
    [128, 128, 0],    # Ground Clutter - olive
    [139, 69, 19],    # Logs - saddle brown
    [128, 128, 128],  # Rocks - gray
    [160, 82, 45],    # Landscape - sienna
    [135, 206, 235],  # Sky - sky blue
], dtype=np.uint8)


def convert_mask(mask):
    """Convert raw mask values to class IDs."""
    arr = np.array(mask)
    new_arr = np.zeros_like(arr, dtype=np.uint8)
    for raw_value, new_value in value_map.items():
        new_arr[arr == raw_value] = new_value
    return Image.fromarray(new_arr)


def mask_to_color(mask):
    """Convert a class mask to a colored RGB image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(n_classes):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask


# ============================================================================
# Dataset
# ============================================================================

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, mask_transform=None):
        self.image_dir = os.path.join(data_dir, 'Color_Images')
        self.masks_dir = os.path.join(data_dir, 'Segmentation')
        self.transform = transform
        self.mask_transform = mask_transform
        self.data_ids = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        data_id = self.data_ids[idx]
        img_path = os.path.join(self.image_dir, data_id)
        mask_path = os.path.join(self.masks_dir, data_id)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = convert_mask(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.mask_transform(mask) * 255

        return image, mask, data_id


# ============================================================================
# Model: Deeper Segmentation Head — MUST match train_mask.py exactly
# ============================================================================

class ConvNeXtBlock(nn.Module):
    """Single depthwise-separable ConvNeXt block."""
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.dw   = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                              padding=padding, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.pw1  = nn.Linear(channels, channels * 4)
        self.act  = nn.GELU()
        self.pw2  = nn.Linear(channels * 4, channels)

    def forward(self, x):
        residual = x
        x = self.dw(x)                       # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)           # (B, H, W, C)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = x.permute(0, 3, 1, 2)           # (B, C, H, W)
        return x + residual


class SegmentationHeadConvNeXt(nn.Module):
    """
    Deeper segmentation head (mirrors train_mask.py):
      - stem: 1x1 conv projects backbone dim -> 256
      - 3 ConvNeXt blocks at 256 channels
      - 1x1 classifier
    """
    def __init__(self, in_channels, out_channels, tokenW, tokenH):
        super().__init__()
        self.H, self.W = tokenH, tokenW
        hidden = 256  # matches training

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
    """Compute IoU for each class and return (mean IoU, per-class list)."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    iou_per_class = []
    for class_id in range(num_classes):
        if class_id == ignore_index:
            continue
        pred_inds   = pred == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        union        = (pred_inds | target_inds).sum().float()
        if union == 0:
            iou_per_class.append(float('nan'))
        else:
            iou_per_class.append((intersection / union).cpu().numpy())

    return np.nanmean(iou_per_class), iou_per_class


def compute_dice(pred, target, num_classes=10, smooth=1e-6):
    """Compute Dice coefficient per class and return (mean Dice, per-class list)."""
    pred = torch.argmax(pred, dim=1)
    pred, target = pred.view(-1), target.view(-1)

    dice_per_class = []
    for class_id in range(num_classes):
        pred_inds   = pred == class_id
        target_inds = target == class_id
        intersection = (pred_inds & target_inds).sum().float()
        dice_score   = (2. * intersection + smooth) / (
            pred_inds.sum().float() + target_inds.sum().float() + smooth
        )
        dice_per_class.append(dice_score.cpu().numpy())

    return np.mean(dice_per_class), dice_per_class


def compute_pixel_accuracy(pred, target):
    """Compute pixel accuracy."""
    pred_classes = torch.argmax(pred, dim=1)
    return (pred_classes == target).float().mean().cpu().numpy()


# ============================================================================
# Visualization Functions
# ============================================================================

def save_prediction_comparison(img_tensor, gt_mask, pred_mask, output_path, data_id):
    """Save a side-by-side comparison of input, ground truth, and prediction."""
    img = img_tensor.cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img = np.moveaxis(img, 0, -1)
    img = img * std + mean
    img = np.clip(img, 0, 1)

    gt_color   = mask_to_color(gt_mask.cpu().numpy().astype(np.uint8))
    pred_color = mask_to_color(pred_mask.cpu().numpy().astype(np.uint8))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img);       axes[0].set_title('Input Image');    axes[0].axis('off')
    axes[1].imshow(gt_color);  axes[1].set_title('Ground Truth');   axes[1].axis('off')
    axes[2].imshow(pred_color);axes[2].set_title('Prediction');     axes[2].axis('off')

    plt.suptitle(f'Sample: {data_id}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_metrics_summary(results, output_dir):
    """Save metrics summary to a text file and create bar chart."""
    os.makedirs(output_dir, exist_ok=True)

    filepath = os.path.join(output_dir, 'evaluation_metrics.txt')
    with open(filepath, 'w') as f:
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Mean IoU:          {results['mean_iou']:.4f}\n")
        f.write("=" * 50 + "\n\n")
        f.write("Per-Class IoU:\n")
        f.write("-" * 40 + "\n")
        for name, iou in zip(class_names, results['class_iou']):
            iou_str = f"{iou:.4f}" if not np.isnan(iou) else "N/A"
            f.write(f"  {name:<20}: {iou_str}\n")
    print(f"\nSaved evaluation metrics to {filepath}")

    # Per-class IoU bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    valid_iou = [iou if not np.isnan(iou) else 0 for iou in results['class_iou']]
    ax.bar(range(n_classes), valid_iou,
           color=[color_palette[i] / 255 for i in range(n_classes)],
           edgecolor='black')
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_ylabel('IoU')
    ax.set_title(f'Per-Class IoU (Mean: {results["mean_iou"]:.4f})')
    ax.set_ylim(0, 1)
    ax.axhline(y=results['mean_iou'], color='red', linestyle='--', label='Mean IoU')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved per-class metrics chart to '{output_dir}/per_class_metrics.png'")


# ============================================================================
# Main Validation Function
# ============================================================================

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description='Segmentation prediction/inference script')
    parser.add_argument('--model_path', type=str,
                        default=os.path.join(script_dir, 'segmentation_head.pth'),
                        help='Path to trained model weights')
    # ── CHANGED: --use_best_model flag loads segmentation_head_best.pth ───────
    parser.add_argument('--use_best_model', action='store_true',
                        help='Load segmentation_head_best.pth (best val IoU checkpoint) '
                             'instead of the final segmentation_head.pth')
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(script_dir, 'Offroad_Segmentation_testImages'),
                        help='Path to validation dataset')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                        help='Directory to save prediction visualizations')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for validation')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of comparison visualizations to save '
                             '(predictions saved for ALL images)')
    args = parser.parse_args()

    # ── CHANGED: honour --use_best_model flag ─────────────────────────────────
    if args.use_best_model:
        args.model_path = os.path.join(script_dir, 'segmentation_head_best.pth')
        print("Using best-checkpoint model: segmentation_head_best.pth")

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == "cuda"
    print(f"Using device: {device}  |  AMP: {use_amp}")

    # Image dimensions (must match training)
    w = int(((960 / 2) // 14) * 14)
    h = int(((540 / 2) // 14) * 14)

    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
    ])

    print(f"Loading dataset from {args.data_dir}...")
    valset     = MaskDataset(data_dir=args.data_dir, transform=transform, mask_transform=mask_transform)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True)
    print(f"Loaded {len(valset)} samples")

    # ── CHANGED: backbone upgraded to base (vitb14_reg) ───────────────────────
    print("Loading DINOv2 backbone (base)...")
    BACKBONE_SIZE  = "base"
    backbone_archs = {
        "small": "vits14",
        "base":  "vitb14_reg",
        "large": "vitl14_reg",
        "giant": "vitg14_reg",
    }
    backbone_name  = f"dinov2_{backbone_archs[BACKBONE_SIZE]}"
    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()
    backbone_model.to(device)
    print("Backbone loaded successfully!")

    # Get embedding dimension
    sample_img, _, _ = valset[0]
    sample_img = sample_img.unsqueeze(0).to(device)
    with torch.no_grad():
        output = backbone_model.forward_features(sample_img)["x_norm_patchtokens"]
    n_embedding = output.shape[2]
    print(f"Embedding dimension: {n_embedding}")

    # ── CHANGED: load deeper head (256-channel, 3 blocks) ─────────────────────
    print(f"Loading model from {args.model_path}...")
    classifier = SegmentationHeadConvNeXt(
        in_channels=n_embedding,
        out_channels=n_classes,
        tokenW=w // 14,
        tokenH=h // 14
    )
    classifier.load_state_dict(torch.load(args.model_path, map_location=device))
    classifier = classifier.to(device)
    classifier.eval()
    print("Model loaded successfully!")

    # Output sub-directories
    masks_dir       = os.path.join(args.output_dir, 'masks')
    masks_color_dir = os.path.join(args.output_dir, 'masks_color')
    comparisons_dir = os.path.join(args.output_dir, 'comparisons')
    for d in (masks_dir, masks_color_dir, comparisons_dir):
        os.makedirs(d, exist_ok=True)

    print(f"\nRunning evaluation and saving predictions for all {len(valset)} images...")

    iou_scores, dice_scores, pixel_accuracies = [], [], []
    all_class_iou, all_class_dice = [], []
    sample_count = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Processing", unit="batch")
        for imgs, labels, data_ids in pbar:
            imgs, labels = imgs.to(device), labels.to(device)

            # ── CHANGED: AMP autocast for faster inference ─────────────────────
            with torch.autocast(device_type=device.type, enabled=use_amp):
                output  = backbone_model.forward_features(imgs)["x_norm_patchtokens"]
                logits  = classifier(output.to(device))
                outputs = F.interpolate(logits, size=imgs.shape[2:],
                                        mode="bilinear", align_corners=False)

            labels_squeezed = labels.squeeze(dim=1).long()
            predicted_masks = torch.argmax(outputs, dim=1)

            iou,  class_iou  = compute_iou(outputs, labels_squeezed, num_classes=n_classes)
            dice, class_dice = compute_dice(outputs, labels_squeezed, num_classes=n_classes)
            pixel_acc        = compute_pixel_accuracy(outputs, labels_squeezed)

            iou_scores.append(iou)
            dice_scores.append(dice)
            pixel_accuracies.append(pixel_acc)
            all_class_iou.append(class_iou)
            all_class_dice.append(class_dice)

            # Save per-image outputs
            for i in range(imgs.shape[0]):
                data_id   = data_ids[i]
                base_name = os.path.splitext(data_id)[0]

                # Raw class-ID mask
                pred_mask = predicted_masks[i].cpu().numpy().astype(np.uint8)
                Image.fromarray(pred_mask).save(
                    os.path.join(masks_dir, f'{base_name}_pred.png'))

                # Coloured mask
                pred_color = mask_to_color(pred_mask)
                cv2.imwrite(
                    os.path.join(masks_color_dir, f'{base_name}_pred_color.png'),
                    cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))

                # Side-by-side comparison (first N samples only)
                if sample_count < args.num_samples:
                    save_prediction_comparison(
                        imgs[i], labels_squeezed[i], predicted_masks[i],
                        os.path.join(comparisons_dir,
                                     f'sample_{sample_count}_comparison.png'),
                        data_id
                    )
                sample_count += 1

            pbar.set_postfix(iou=f"{iou:.3f}")

    # Aggregate
    mean_iou      = np.nanmean(iou_scores)
    mean_dice     = np.nanmean(dice_scores)
    mean_pixel_acc = np.mean(pixel_accuracies)
    avg_class_iou  = np.nanmean(all_class_iou, axis=0)

    results = {'mean_iou': mean_iou, 'class_iou': avg_class_iou}

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Mean IoU:          {mean_iou:.4f}")
    print(f"Mean Dice:         {mean_dice:.4f}")
    print(f"Pixel Accuracy:    {mean_pixel_acc:.4f}")
    print("=" * 50)

    save_metrics_summary(results, args.output_dir)

    print(f"\nPrediction complete! Processed {len(valset)} images.")
    print(f"\nOutputs saved to {args.output_dir}/")
    print(f"  - masks/               : Raw prediction masks (class IDs 0-9)")
    print(f"  - masks_color/         : Colored prediction masks (RGB)")
    print(f"  - comparisons/         : Side-by-side comparisons ({args.num_samples} samples)")
    print(f"  - evaluation_metrics.txt")
    print(f"  - per_class_metrics.png")


if __name__ == "__main__":
    main()