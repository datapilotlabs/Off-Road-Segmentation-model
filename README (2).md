# 🚀 Offroad Semantic Scene Segmentation

## 📥 Dataset
Download the dataset here:  
https://falcon.duality.ai/secure/documentation/hackathon-segmentation-desert?utm_source=hackathon&utm_medium=instructions&utm_campaign=HacktheNight  

---

# 🌿 Off-Road Scene Segmentation

A deep learning pipeline for **semantic segmentation of off-road environments**, built on a frozen [DINOv2](https://github.com/facebookresearch/dinov2) ViT-B/14 backbone with a custom ConvNeXt-style segmentation head. Includes training, evaluation, and a Flask web UI for real-time image and video inference.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Segmentation Classes](#segmentation-classes)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Running the Web App (localhost)](#running-the-web-app-localhost)
- [API Reference](#api-reference)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Outputs](#outputs)

---

## Overview

This project segments off-road scenes into 10 semantic classes using:

- **Backbone:** `dinov2_vitb14_reg` (frozen, loaded from `facebookresearch/dinov2` via `torch.hub`)
- **Segmentation Head:** `SegmentationHeadConvNeXt` — 3 ConvNeXt blocks at 256 channels
- **Loss:** CrossEntropyLoss with label smoothing (0.1)
- **Optimizer:** AdamW with cosine annealing LR schedule
- **Training:** Mixed-precision (AMP) with gradient clipping; best checkpoint saved at peak val IoU

---

## Segmentation Classes

| ID | Class | Color |
|----|-------|-------|
| 0 | Background | Black |
| 1 | Trees | Forest Green |
| 2 | Lush Bushes | Lime |
| 3 | Dry Grass | Tan |
| 4 | Dry Bushes | Brown |
| 5 | Ground Clutter | Olive |
| 6 | Logs | Saddle Brown |
| 7 | Rocks | Gray |
| 8 | Landscape | Sienna |
| 9 | Sky | Sky Blue |

---

## Project Structure

```
├── train_segmentation.py       # Training script
├── test_segmentation.py        # Evaluation / inference script
├── app.py                      # Flask web application
├── segmentation_head_best.pth  # Best model checkpoint (saved after training)
├── segmentation_head.pth       # Final model checkpoint (saved after training)
├── train_stats/                # Training curves and metric logs (auto-created)
├── predictions/                # Evaluation outputs (auto-created)
│   ├── masks/                  # Raw class-ID prediction masks
│   ├── masks_color/            # Colorized prediction masks
│   └── comparisons/            # Side-by-side comparison images
├── train/                      # Training dataset (see Dataset section)
│   ├── Color_Images/
│   └── Segmentation/
└── val/                        # Validation dataset
    ├── Color_Images/
    └── Segmentation/
```

---

## Dataset

This project uses an **off-road segmentation dataset** with paired RGB images and segmentation masks.

### Expected Structure

Each split (`train/`, `val/`, test dir) must follow this layout:

```
<split>/
├── Color_Images/     # RGB images (.png or .jpg)
└── Segmentation/     # Grayscale mask images with the same filenames
```

### Raw Mask Value Mapping

The dataset masks use sparse pixel values that are remapped to class IDs at load time:

| Raw Value | Class ID | Label |
|-----------|----------|-------|
| 0 | 0 | Background |
| 100 | 1 | Trees |
| 200 | 2 | Lush Bushes |
| 300 | 3 | Dry Grass |
| 500 | 4 | Dry Bushes |
| 550 | 5 | Ground Clutter |
| 700 | 6 | Logs |
| 800 | 7 | Rocks |
| 7100 | 8 | Landscape |
| 10000 | 9 | Sky |

> ⚠️ All images are resized to **672 × 378** pixels (multiples of the DINOv2 patch size of 14) during preprocessing.

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (strongly recommended; CPU inference is supported but slow)
- PyTorch ≥ 2.0

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Install dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install flask numpy pillow opencv-python tqdm matplotlib
```

Or install all at once with a requirements file:

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**

```
torch>=2.0.0
torchvision>=0.15.0
flask>=2.3.0
numpy>=1.24.0
pillow>=9.5.0
opencv-python>=4.7.0
tqdm>=4.65.0
matplotlib>=3.7.0
```

### 3. DINOv2 Backbone

The backbone (`dinov2_vitb14_reg`) is downloaded automatically from `facebookresearch/dinov2` via `torch.hub` on first run. No manual download required — ensure you have an internet connection on first launch.

---

## Training

### Basic usage (default paths)

```bash
python train_segmentation.py
```

This expects `train/` and `val/` directories in the same folder as the script.

### Custom data paths

```bash
python train_segmentation.py \
  --data-root /path/to/dataset \
  --train-dir train \
  --val-dir val
```

You can also pass absolute paths:

```bash
python train_segmentation.py \
  --train-dir /absolute/path/to/train \
  --val-dir /absolute/path/to/val
```

### What gets saved

| File | Description |
|------|-------------|
| `segmentation_head_best.pth` | Best checkpoint (highest val IoU) |
| `segmentation_head.pth` | Final epoch checkpoint |
| `train_stats/` | Loss & metric plots, history CSV |

### Training configuration

Key hyperparameters are in the `Config` class inside `train_segmentation.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_epochs` | 2 | Number of training epochs |
| `batch_size` | 2 | Batch size |
| `lr` | 3e-4 | AdamW learning rate |
| `weight_decay` | 1e-4 | AdamW weight decay |
| `label_smoothing` | 0.1 | CrossEntropyLoss smoothing |
| `grad_clip_norm` | 1.0 | Gradient clipping max norm |
| `patch_width` | 672 | Input width (must be divisible by 14) |
| `patch_height` | 378 | Input height (must be divisible by 14) |

---

## Evaluation

Run the evaluation script against a test or validation split:

```bash
python test_segmentation.py \
  --data_dir /path/to/test/split \
  --output_dir ./predictions \
  --num_samples 5
```

### Use the best checkpoint

```bash
python test_segmentation.py \
  --data_dir /path/to/test/split \
  --use_best_model
```

### All arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `segmentation_head.pth` | Path to model weights |
| `--use_best_model` | False | Load `segmentation_head_best.pth` instead |
| `--data_dir` | `Offroad_Segmentation_testImages/` | Path to test dataset |
| `--output_dir` | `./predictions` | Where to save outputs |
| `--batch_size` | 2 | Inference batch size |
| `--num_samples` | 5 | Number of side-by-side comparisons to save |

### Evaluation outputs

```
predictions/
├── masks/                   # Raw class-ID masks (0–9)
├── masks_color/             # RGB colorized masks
├── comparisons/             # Side-by-side: original | ground truth | prediction
├── evaluation_metrics.txt   # Mean IoU, Dice, pixel accuracy
└── per_class_metrics.png    # Bar chart of per-class IoU
```

### Reported metrics

- **Mean IoU** (primary metric)
- **Mean Dice Score**
- **Pixel Accuracy**
- **Per-class IoU** (bar chart)

---

## Running the Web App (localhost)

The Flask app provides a browser-based UI for uploading images or videos and visualizing segmentation predictions in real time.

### 1. Make sure a trained model exists

Training must be run first. The app automatically loads `segmentation_head_best.pth`, falling back to `segmentation_head.pth`.

### 2. Start the server

```bash
python app.py
```

You will see:

```
Starting Flask app for segmentation prediction...
Using device: cuda
Loading DINOv2 backbone...
 * Running on http://0.0.0.0:5000
```

### 3. Open in your browser

```
http://localhost:5000
```

### Features

- **Image upload** — upload any `.jpg`, `.jpeg`, or `.png` image and receive a colorized segmentation mask side by side with the original.
- **Video upload** — upload `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, or `.flv` files. Video is processed frame-by-frame in a background thread; a progress bar is shown. The processed video can be downloaded on completion.
- **Class legend** — a color legend for all 10 classes is shown alongside every result.

### Stopping the server

Press `Ctrl+C` in the terminal.

---

## API Reference

The app also exposes a REST API for programmatic use.

### `POST /api/predict`

Run segmentation on a single image.

**Request:** `multipart/form-data` with an `image` field.

**Response (JSON):**

```json
{
  "success": true,
  "segmentation_mask": "<base64-encoded PNG>",
  "classes": ["Background", "Trees", ...],
  "colors": [[0, 0, 0], [34, 139, 34], ...]
}
```

**Example (curl):**

```bash
curl -X POST http://localhost:5000/api/predict \
  -F "image=@my_photo.jpg" | python -m json.tool
```

### `POST /process_video`

Start background video processing.

**Request:** `multipart/form-data` with a `video` field.

**Response (JSON):**

```json
{
  "success": true,
  "video_id": "17432918283",
  "message": "Video processing started"
}
```

### `GET /video_progress/<video_id>`

Poll processing status.

**Response states:** `processing`, `completed`, `error`

### `GET /download_video/<video_id>`

Download the processed (segmented) video once `status == "completed"`.

---

## Model Architecture

```
Input Image (3 × 378 × 672)
        │
        ▼
DINOv2 ViT-B/14 backbone (frozen)
  dinov2_vitb14_reg
        │  patch tokens: (B, N, 768)
        ▼
SegmentationHeadConvNeXt
  ├── stem: Conv1×1 (768 → 256) + GELU
  ├── ConvNeXtBlock ×3
  │     ├── Depthwise Conv 7×7
  │     ├── LayerNorm
  │     └── MLP: Linear(256→1024) → GELU → Linear(1024→256) + residual
  └── classifier: Conv1×1 (256 → 10)
        │  logits: (B, 10, H_token, W_token)
        ▼
Bilinear upsampling → (B, 10, 378, 672)
        │
        ▼
argmax → class mask (B, 378, 672)
```

---

## Configuration

### Patch size constraint

DINOv2 uses 14×14 patches. Input dimensions **must be divisible by 14**:

```
patch_width  = int(((960 / 2) // 14) * 14) = 672
patch_height = int(((540 / 2) // 14) * 14) = 378
```

Changing input resolution requires updating these values consistently across all three scripts.

### Device

All scripts auto-detect CUDA. Mixed-precision (AMP) is enabled automatically when a GPU is available.

---

## Outputs

### Training (`train_stats/`)

| File | Description |
|------|-------------|
| `loss_curve.png` | Train vs. val loss per epoch |
| `iou_curve.png` | Train vs. val mean IoU per epoch |
| `dice_curve.png` | Train vs. val Dice score per epoch |
| `pixel_acc_curve.png` | Train vs. val pixel accuracy per epoch |
| `history.txt` | Raw metric values per epoch |

### Evaluation (`predictions/`)

| Path | Description |
|------|-------------|
| `masks/` | Grayscale masks with class IDs 0–9 |
| `masks_color/` | RGB colorized masks |
| `comparisons/` | Side-by-side comparisons for N samples |
| `evaluation_metrics.txt` | Summary of mean IoU, Dice, pixel accuracy |
| `per_class_metrics.png` | Per-class IoU bar chart |

---

## License

This project is released under the [MIT License](LICENSE).

---

## Acknowledgements

- [DINOv2](https://github.com/facebookresearch/dinov2) — Meta AI's self-supervised ViT backbone
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) — depthwise block design inspiration
- [PyTorch](https://pytorch.org/) and [Flask](https://flask.palletsprojects.com/)
