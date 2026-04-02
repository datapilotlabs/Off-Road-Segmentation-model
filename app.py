"""
Flask Web App for Segmentation Prediction
Allows users to upload images and get segmentation predictions
"""

import os
import io
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from flask import Flask, request, render_template_string, send_file, jsonify
import base64
import cv2
import tempfile
import threading
import time
from queue import Queue

# ============================================================================
# Model Components (copied from train_segmentation.py)
# ============================================================================

class ConvNeXtBlock(nn.Module):
    """Single depthwise-separable ConvNeXt block."""
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.dw = nn.Conv2d(channels, channels, kernel_size=kernel_size,
                            padding=padding, groups=channels)
        self.norm = nn.LayerNorm(channels)
        self.pw1 = nn.Linear(channels, channels * 4)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(channels * 4, channels)

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
        hidden = 256

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
# Configuration
# ============================================================================

class Config:
    """Configuration for inference."""
    def __init__(self):
        self.patch_width = int(((960 / 2) // 14) * 14)   # 672
        self.patch_height = int(((540 / 2) // 14) * 14)  # 378
        self.n_classes = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# Class Information
# ============================================================================

class_names = [
    'Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Dry Bushes',
    'Ground Clutter', 'Logs', 'Rocks', 'Landscape', 'Sky'
]

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


def mask_to_color(mask):
    """Convert a class mask to a colored RGB image."""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id in range(len(class_names)):
        color_mask[mask == class_id] = color_palette[class_id]
    return color_mask


# ============================================================================
# Flask App
# ============================================================================

app = Flask(__name__)
config = Config()

# Global model variables
backbone_model = None
classifier = None
transform = None

def load_models():
    """Load the trained models."""
    global backbone_model, classifier, transform

    if backbone_model is None:
        print("Loading DINOv2 backbone...")
        backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model="dinov2_vitb14_reg")
        backbone_model.eval()
        backbone_model.to(config.device)

    if classifier is None:
        print("Loading segmentation head...")
        # Get embedding dimension
        dummy_input = torch.randn(1, 3, config.patch_height, config.patch_width).to(config.device)
        with torch.no_grad():
            output = backbone_model.forward_features(dummy_input)["x_norm_patchtokens"]
        n_embedding = output.shape[2]

        classifier = SegmentationHeadConvNeXt(
            in_channels=n_embedding,
            out_channels=config.n_classes,
            tokenW=config.patch_width // 14,
            tokenH=config.patch_height // 14
        )

        # Load trained weights (try best model first, fallback to regular)
        model_path = "segmentation_head_best.pth"
        if not os.path.exists(model_path):
            model_path = "segmentation_head.pth"

        if os.path.exists(model_path):
            classifier.load_state_dict(torch.load(model_path, map_location=config.device))
            print(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError("No trained model found. Please run training first.")

        classifier = classifier.to(config.device)
        classifier.eval()

    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((config.patch_height, config.patch_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    print("Models loaded successfully!")


def predict_segmentation(image):
    """Run segmentation prediction on an image."""
    load_models()

    # Preprocess image
    if isinstance(image, str):
        # If image is a file path
        image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        # If image is already a PIL Image
        image = image.convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(config.device)

    with torch.no_grad():
        with torch.autocast(device_type=config.device.type, enabled=(config.device.type == "cuda")):
            features = backbone_model.forward_features(input_tensor)["x_norm_patchtokens"]
            logits = classifier(features)
            outputs = F.interpolate(logits, size=input_tensor.shape[2:],
                                  mode="bilinear", align_corners=False)

    # Get prediction
    pred_mask = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    return pred_mask


def process_video(video_path, output_path, progress_callback=None):
    """Process video frames and create segmented output video."""
    load_models()

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))  # Side by side

    frame_count = 0
    processed_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Run segmentation
        pred_mask = predict_segmentation(pil_image)
        color_mask = mask_to_color(pred_mask)

        # Resize mask to match original frame size
        mask_resized = cv2.resize(color_mask, (width, height), interpolation=cv2.INTER_NEAREST)

        # Create side-by-side comparison
        combined = np.hstack([frame_rgb, cv2.cvtColor(mask_resized, cv2.COLOR_RGB2BGR)])
        out.write(combined)

        frame_count += 1

        # Update progress
        if progress_callback:
            progress = (frame_count / total_frames) * 100
            progress_callback(progress, f"Processing frame {frame_count}/{total_frames}")

    cap.release()
    out.release()

    return {
        'total_frames': frame_count,
        'fps': fps,
        'width': width,
        'height': height,
        'output_path': output_path
    }


# Progress tracking
video_progress = {}


def update_progress(video_id, progress, message):
    """Update video processing progress."""
    video_progress[video_id] = {
        'progress': progress,
        'message': message,
        'timestamp': time.time()
    }


# HTML template for the upload form
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Offroad Segmentation Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
            line-height: 1.6;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }

        .header h1 {
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 10px;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }

        .header p {
            font-size: 1.2rem;
            opacity: 0.9;
            max-width: 600px;
            margin: 0 auto;
        }

        .upload-section {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }

        .upload-tabs {
            display: flex;
            margin-bottom: 30px;
            background: #f8f9fa;
            border-radius: 10px;
            padding: 5px;
        }

        .tab-btn {
            flex: 1;
            padding: 12px 20px;
            border: none;
            background: transparent;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            transition: all 0.3s ease;
            color: #666;
        }

        .tab-btn.active {
            background: white;
            color: #667eea;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .upload-area {
            border: 3px dashed #e1e5e9;
            border-radius: 15px;
            padding: 60px 40px;
            text-align: center;
            transition: all 0.3s ease;
            cursor: pointer;
            background: #fafbfc;
            position: relative;
        }

        .upload-area:hover,
        .upload-area.dragover {
            border-color: #667eea;
            background: #f8f9ff;
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
        }

        .upload-area.hidden {
            display: none;
        }

        .upload-icon {
            font-size: 4rem;
            color: #667eea;
            margin-bottom: 20px;
        }

        .upload-text {
            font-size: 1.3rem;
            color: #666;
            margin-bottom: 10px;
        }

        .upload-subtext {
            color: #999;
            font-size: 0.9rem;
        }

        .file-input {
            display: none;
        }

        .upload-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 20px;
        }

        .upload-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }

        .upload-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }

        .loading.show {
            display: block;
        }

        .spinner {
            width: 50px;
            height: 50px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .results-section {
            display: none;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }

        .results-section.show {
            display: block;
        }

        .results-header {
            text-align: center;
            margin-bottom: 30px;
        }

        .results-header h2 {
            color: #333;
            font-size: 2rem;
            margin-bottom: 10px;
        }

        .image-comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }

        .image-container {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
        }

        .image-container h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.3rem;
        }

        .result-image {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        .legend {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
        }

        .legend h3 {
            text-align: center;
            color: #333;
            margin-bottom: 20px;
            font-size: 1.3rem;
        }

        .legend-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }

        .legend-item {
            display: flex;
            align-items: center;
            padding: 10px;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }

        .color-box {
            width: 30px;
            height: 30px;
            border-radius: 6px;
            margin-right: 12px;
            border: 2px solid #e1e5e9;
            flex-shrink: 0;
        }

        .legend-text {
            font-weight: 500;
            color: #555;
        }

        .error-message {
            display: none;
            background: #fee;
            color: #c33;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #c33;
            margin-bottom: 20px;
        }

        .error-message.show {
            display: block;
        }

        .try-again-btn {
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            margin-top: 10px;
        }

        .video-progress {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 30px;
            margin: 20px 0;
            text-align: center;
        }

        .progress-bar {
            width: 100%;
            height: 20px;
            background: #e1e5e9;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 15px;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            width: 0%;
            transition: width 0.3s ease;
        }

        .progress-text {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-weight: 500;
        }

        .video-info {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }

        .video-info p {
            margin: 10px 0;
            color: #555;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🗺️ Offroad Segmentation Predictor</h1>
            <p>Upload an image to get AI-powered segmentation predictions for offroad terrain classes</p>
        </div>

        <div id="error-message" class="error-message">
            <strong>Error:</strong> <span id="error-text"></span>
            <br>
            <button class="try-again-btn" onclick="resetForm()">Try Again</button>
        </div>

        <div class="upload-section">
            <div class="upload-tabs">
                <button type="button" class="tab-btn active" onclick="switchTab('image')">📸 Image</button>
                <button type="button" class="tab-btn" onclick="switchTab('video')">🎬 Video</button>
            </div>

            <form id="image-form" class="tab-content active" method="POST" enctype="multipart/form-data">
                <div id="image-upload-area" class="upload-area">
                    <div class="upload-icon">📷</div>
                    <div class="upload-text">Drop your image here or click to browse</div>
                    <div class="upload-subtext">Supports JPG, PNG, and other common image formats</div>
                    <input type="file" name="image" id="image-file-input" class="file-input" accept="image/*" required>
                </div>
                <button type="submit" id="image-upload-btn" class="upload-btn" disabled>
                    🚀 Predict Segmentation
                </button>
            </form>

            <form id="video-form" class="tab-content" method="POST" enctype="multipart/form-data">
                <div id="video-upload-area" class="upload-area">
                    <div class="upload-icon">🎬</div>
                    <div class="upload-text">Drop your video here or click to browse</div>
                    <div class="upload-subtext">Supports MP4, AVI, MOV, MKV, WMV, FLV formats</div>
                    <input type="file" name="video" id="video-file-input" class="file-input" accept="video/*" required>
                </div>
                <button type="submit" id="video-upload-btn" class="upload-btn" disabled>
                    🎬 Process Video
                </button>
            </form>

            <div id="loading" class="loading">
                <div class="spinner"></div>
                <h3 id="loading-title">Analyzing your image...</h3>
                <p id="loading-message">This may take a few seconds</p>
            </div>
        </div>

        <div id="video-results-section" class="results-section">
            <div class="results-header">
                <h2>🎬 Video Processing</h2>
                <p id="video-status">Initializing video processing...</p>
            </div>

            <div class="video-progress">
                <div class="progress-bar">
                    <div id="progress-fill" class="progress-fill" style="width: 0%"></div>
                </div>
                <div class="progress-text">
                    <span id="progress-percent">0%</span>
                    <span id="progress-message">Starting...</span>
                </div>
            </div>

            <div id="video-completed" style="display: none;">
                <div class="results-header">
                    <h2>✅ Processing Complete!</h2>
                    <p>Your segmented video is ready for download.</p>
                </div>
                <div class="video-info">
                    <p><strong>Frames processed:</strong> <span id="frames-count"></span></p>
                    <p><strong>Original FPS:</strong> <span id="video-fps"></span></p>
                    <p><strong>Resolution:</strong> <span id="video-resolution"></span></p>
                </div>
                <a id="download-btn" class="upload-btn" href="#" download>📥 Download Segmented Video</a>
            </div>
        </div>

            <div class="image-comparison">
                <div class="image-container">
                    <h3>📸 Original Image</h3>
                    <img id="original-image" class="result-image" src="" alt="Original Image">
                </div>
                <div class="image-container">
                    <h3>🎨 Segmentation Mask</h3>
                    <img id="segmentation-mask" class="result-image" src="" alt="Segmentation Mask">
                </div>
            </div>

            <div class="legend">
                <h3>📊 Class Legend</h3>
                <div class="legend-grid" id="legend-grid">
                    <!-- Legend items will be populated by JavaScript -->
                </div>
            </div>
        </div>
    </div>

    <script>
        const uploadArea = document.getElementById('image-upload-area');
        const imageFileInput = document.getElementById('image-file-input');
        const imageUploadBtn = document.getElementById('image-upload-btn');
        const imageForm = document.getElementById('image-form');

        const videoUploadArea = document.getElementById('video-upload-area');
        const videoFileInput = document.getElementById('video-file-input');
        const videoUploadBtn = document.getElementById('video-upload-btn');
        const videoForm = document.getElementById('video-form');

        const uploadForm = imageForm; // For backward compatibility
        const fileInput = imageFileInput;
        const uploadBtn = imageUploadBtn;

        const loading = document.getElementById('loading');
        const loadingTitle = document.getElementById('loading-title');
        const loadingMessage = document.getElementById('loading-message');
        const resultsSection = document.getElementById('results-section');
        const videoResultsSection = document.getElementById('video-results-section');
        const errorMessage = document.getElementById('error-message');
        const errorText = document.getElementById('error-text');

        // Tab switching functionality
        function switchTab(tabType) {
            // Update tab buttons
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            // Update tab content
            document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
            document.getElementById(tabType + '-form').classList.add('active');

            // Reset forms and states
            resetForm();
        }

        // Drag and drop functionality for images
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
            videoUploadArea.addEventListener(eventName, preventDefaults, false);
        });

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, highlight, false);
            videoUploadArea.addEventListener(eventName, highlight, false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, unhighlight, false);
            videoUploadArea.addEventListener(eventName, unhighlight, false);
        });

        function highlight(e) {
            e.target.classList.add('dragover');
        }

        function unhighlight(e) {
            e.target.classList.remove('dragover');
        }

        uploadArea.addEventListener('drop', (e) => handleDrop(e, 'image'));
        videoUploadArea.addEventListener('drop', (e) => handleDrop(e, 'video'));

        function handleDrop(e, type) {
            const dt = e.dataTransfer;
            const files = dt.files;

            if (files.length > 0) {
                const input = type === 'image' ? imageFileInput : videoFileInput;
                input.files = files;
                handleFileSelect(type);
            }
        }

        uploadArea.addEventListener('click', () => imageFileInput.click());
        videoUploadArea.addEventListener('click', () => videoFileInput.click());

        imageFileInput.addEventListener('change', () => handleFileSelect('image'));
        videoFileInput.addEventListener('change', () => handleFileSelect('video'));

        function handleFileSelect(type) {
            const input = type === 'image' ? imageFileInput : videoFileInput;
            const btn = type === 'image' ? imageUploadBtn : videoUploadBtn;

            if (input.files.length > 0) {
                btn.disabled = false;
                btn.textContent = type === 'image' ?
                    `🚀 Predict Segmentation (${input.files[0].name})` :
                    `🎬 Process Video (${input.files[0].name})`;
            } else {
                btn.disabled = true;
                btn.textContent = type === 'image' ? '🚀 Predict Segmentation' : '🎬 Process Video';
            }
        }

        // Form submission handling
        imageForm.addEventListener('submit', function(e) {
            showLoading('Analyzing your image...', 'This may take a few seconds');
        });

        videoForm.addEventListener('submit', function(e) {
            e.preventDefault(); // Prevent default form submission

            const formData = new FormData(videoForm);

            fetch('/process_video', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    startVideoProgressTracking(data.video_id);
                } else {
                    showError(data.error);
                }
            })
            .catch(error => {
                showError('Network error: ' + error.message);
            });
        });

        function showLoading(title, message) {
            if (uploadArea) uploadArea.classList.add('hidden');
            if (videoUploadArea) videoUploadArea.classList.add('hidden');
            if (imageUploadBtn) imageUploadBtn.style.display = 'none';
            if (videoUploadBtn) videoUploadBtn.style.display = 'none';
            if (loadingTitle) loadingTitle.textContent = title;
            if (loadingMessage) loadingMessage.textContent = message;
            if (loading) loading.classList.add('show');
            if (errorMessage) errorMessage.classList.remove('show');
            if (resultsSection) resultsSection.classList.remove('show');
            if (videoResultsSection) videoResultsSection.classList.remove('show');
        }

        function showError(message) {
            if (errorText) errorText.textContent = message;
            if (errorMessage) errorMessage.classList.add('show');
            resetForm();
        }

        function resetForm() {
            if (uploadArea) uploadArea.classList.remove('hidden');
            if (videoUploadArea) videoUploadArea.classList.remove('hidden');
            if (imageUploadBtn) {
                imageUploadBtn.style.display = 'inline-block';
                imageUploadBtn.disabled = true;
                imageUploadBtn.textContent = '🚀 Predict Segmentation';
            }
            if (videoUploadBtn) {
                videoUploadBtn.style.display = 'inline-block';
                videoUploadBtn.disabled = true;
                videoUploadBtn.textContent = '🎬 Process Video';
            }
            if (loading) loading.classList.remove('show');
            if (resultsSection) resultsSection.classList.remove('show');
            if (videoResultsSection) videoResultsSection.classList.remove('show');
            if (errorMessage) errorMessage.classList.remove('show');
            if (imageFileInput) imageFileInput.value = '';
            if (videoFileInput) videoFileInput.value = '';
            if (imageForm) imageForm.reset();
            if (videoForm) videoForm.reset();
        }

        // Video progress tracking
        let progressInterval;

        function startVideoProgressTracking(videoId) {
            showLoading('Processing your video...', 'This may take several minutes depending on video length.');

            // Show video results section
            if (videoResultsSection) videoResultsSection.classList.add('show');
            const videoStatus = document.getElementById('video-status');
            if (videoStatus) videoStatus.textContent = 'Processing video frames...';

            progressInterval = setInterval(() => {
                fetch(`/video_progress/${videoId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        clearInterval(progressInterval);
                        showError(data.error);
                        return;
                    }

                    const progress = data.progress || 0;
                    const message = data.message || 'Processing...';

                    document.getElementById('progress-fill').style.width = progress + '%';
                    document.getElementById('progress-percent').textContent = Math.round(progress) + '%';
                    document.getElementById('progress-message').textContent = message;

                    if (data.status === 'completed') {
                        clearInterval(progressInterval);
                        showVideoCompleted(data.result, videoId);
                    } else if (data.status === 'error') {
                        clearInterval(progressInterval);
                        showError(data.error);
                    }
                })
                .catch(error => {
                    clearInterval(progressInterval);
                    showError('Progress tracking failed: ' + error.message);
                });
            }, 1000);
        }

        function showVideoCompleted(result, videoId) {
            if (loading) loading.classList.remove('show');
            const videoStatus = document.getElementById('video-status');
            if (videoStatus) videoStatus.textContent = '✅ Video processing completed!';
            const progressFill = document.getElementById('progress-fill');
            if (progressFill) progressFill.style.width = '100%';
            const progressPercent = document.getElementById('progress-percent');
            if (progressPercent) progressPercent.textContent = '100%';
            const progressMessage = document.getElementById('progress-message');
            if (progressMessage) progressMessage.textContent = 'Complete!';

            // Show completion section
            const videoCompleted = document.getElementById('video-completed');
            if (videoCompleted) videoCompleted.style.display = 'block';
            const framesCount = document.getElementById('frames-count');
            if (framesCount) framesCount.textContent = result.total_frames;
            const videoFps = document.getElementById('video-fps');
            if (videoFps) videoFps.textContent = result.fps.toFixed(1);
            const videoResolution = document.getElementById('video-resolution');
            if (videoResolution) videoResolution.textContent = `${result.width}x${result.height}`;

            // Set download link
            const downloadBtn = document.getElementById('download-btn');
            if (downloadBtn) {
                downloadBtn.href = `/download_video/${videoId}`;
            }
        }

        // Populate legend
        const classNames = {{ class_names|tojson }};
        const colors = {{ colors|tojson }};
        const legendGrid = document.getElementById('legend-grid');

        classNames.forEach((name, index) => {
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `
                <div class="color-box" style="background-color: rgb(${colors[index][0]}, ${colors[index][1]}, ${colors[index][2]})"></div>
                <span class="legend-text">${name}</span>
            `;
            legendGrid.appendChild(item);
        });

        {% if result %}
        // Show results if they exist
        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('original-image').src = 'data:image/jpeg;base64,{{ result.original_image }}';
            document.getElementById('segmentation-mask').src = 'data:image/png;base64,{{ result.segmentation_mask }}';
            resultsSection.classList.add('show');
            uploadArea.classList.add('hidden');
            imageUploadBtn.style.display = 'none';
        });
        {% elif video_result %}
        // Show video processing if it exists
        document.addEventListener('DOMContentLoaded', function() {
            startVideoProgressTracking('{{ video_result.video_id }}');
        });
        {% elif error %}
        // Show error if it exists
        document.addEventListener('DOMContentLoaded', function() {
            document.getElementById('error-text').textContent = '{{ error }}';
            document.getElementById('error-message').classList.add('show');
        });
        {% endif %}
    </script>
</body>
</html>
"""


@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    error = None
    result = None
    video_result = None

    if request.method == 'POST':
        # Check if it's an image upload
        if 'image' in request.files and request.files['image'].filename:
            file = request.files['image']
            if file.filename == '':
                error = "No image selected"
            elif not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                error = "Unsupported file format. Please upload an image file."
            else:
                try:
                    # Read the uploaded image
                    image_data = file.read()
                    image = Image.open(io.BytesIO(image_data))

                    # Validate image
                    if image.size[0] < 32 or image.size[1] < 32:
                        error = "Image is too small. Please upload an image at least 32x32 pixels."
                    elif image.size[0] > 4096 or image.size[1] > 4096:
                        error = "Image is too large. Please upload an image smaller than 4096x4096 pixels."
                    else:
                        # Run prediction
                        pred_mask = predict_segmentation(image)

                        # Create colored mask
                        color_mask = mask_to_color(pred_mask)

                        # Convert images to base64 for display
                        # Original image
                        original_buffer = io.BytesIO()
                        image.save(original_buffer, format='JPEG', quality=95)
                        original_b64 = base64.b64encode(original_buffer.getvalue()).decode()

                        # Segmentation mask
                        mask_pil = Image.fromarray(color_mask)
                        mask_buffer = io.BytesIO()
                        mask_pil.save(mask_buffer, format='PNG')
                        mask_b64 = base64.b64encode(mask_buffer.getvalue()).decode()

                        result = {
                            'original_image': original_b64,
                            'segmentation_mask': mask_b64
                        }

                except Exception as e:
                    error = f"Error processing image: {str(e)}"

        # Check if it's a video upload
        elif 'video' in request.files and request.files['video'].filename:
            file = request.files['video']
            if file.filename == '':
                error = "No video selected"
            elif not any(file.filename.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']):
                error = "Unsupported video format. Please upload MP4, AVI, MOV, MKV, WMV, or FLV."
            else:
                try:
                    # Save uploaded video to temporary file
                    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
                    file.save(temp_video.name)
                    temp_video.close()

                    # Generate unique ID for this video processing task
                    video_id = str(time.time()).replace('.', '')

                    # Create output path
                    output_filename = f"segmented_{video_id}.mp4"
                    output_path = os.path.join(tempfile.gettempdir(), output_filename)

                    # Start video processing in background thread
                    def process_thread():
                        try:
                            result = process_video(temp_video.name, output_path,
                                                 lambda p, m: update_progress(video_id, p, m))
                            video_progress[video_id].update({
                                'status': 'completed',
                                'result': result
                            })
                        except Exception as e:
                            video_progress[video_id].update({
                                'status': 'error',
                                'error': str(e)
                            })
                        finally:
                            # Clean up temp file
                            try:
                                os.unlink(temp_video.name)
                            except:
                                pass

                    thread = threading.Thread(target=process_thread)
                    thread.daemon = True
                    thread.start()

                    video_result = {
                        'video_id': video_id,
                        'message': 'Video processing started. This may take several minutes depending on video length.'
                    }

                except Exception as e:
                    error = f"Error processing video: {str(e)}"
        else:
            error = "Please upload either an image or a video file."

    return render_template_string(HTML_TEMPLATE,
                                result=result,
                                video_result=video_result,
                                error=error,
                                class_names=class_names,
                                colors=color_palette.tolist())


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic access."""
    if 'image' not in request.files:
        return {"error": "No image uploaded"}, 400

    file = request.files['image']
    if file.filename == '':
        return {"error": "No image selected"}, 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        pred_mask = predict_segmentation(image)

        # Return as JSON with base64 encoded mask
        mask_pil = Image.fromarray(mask_to_color(pred_mask))
        mask_buffer = io.BytesIO()
        mask_pil.save(mask_buffer, format='PNG')
        mask_b64 = base64.b64encode(mask_buffer.getvalue()).decode()

        return {
            "success": True,
            "segmentation_mask": mask_b64,
            "classes": class_names,
            "colors": color_palette.tolist()
        }

    except Exception as e:
        return {"error": str(e)}, 500


@app.route('/process_video', methods=['POST'])
def process_video_route():
    """Process uploaded video for segmentation."""
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No video selected"}), 400

    # Validate video format
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        return jsonify({"error": "Unsupported video format. Please upload MP4, AVI, MOV, MKV, WMV, or FLV."}), 400

    try:
        # Save uploaded video to temporary file
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
        file.save(temp_video.name)
        temp_video.close()

        # Generate unique ID for this video processing task
        video_id = str(time.time()).replace('.', '')

        # Create output path
        output_filename = f"segmented_{video_id}.mp4"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)

        # Start video processing in background thread
        def process_thread():
            try:
                result = process_video(temp_video.name, output_path,
                                     lambda p, m: update_progress(video_id, p, m))
                video_progress[video_id].update({
                    'status': 'completed',
                    'result': result
                })
            except Exception as e:
                video_progress[video_id].update({
                    'status': 'error',
                    'error': str(e)
                })
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_video.name)
                except:
                    pass

        thread = threading.Thread(target=process_thread)
        thread.daemon = True
        thread.start()

        return jsonify({
            "success": True,
            "video_id": video_id,
            "message": "Video processing started"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/video_progress/<video_id>')
def get_video_progress(video_id):
    """Get processing progress for a video."""
    if video_id not in video_progress:
        return jsonify({"error": "Video ID not found"}), 404

    progress_data = video_progress[video_id].copy()

    # Clean up old completed/error entries after some time
    if progress_data.get('status') in ['completed', 'error']:
        if time.time() - progress_data.get('timestamp', 0) > 300:  # 5 minutes
            del video_progress[video_id]
            return jsonify({"error": "Video processing expired"}), 410

    return jsonify(progress_data)


@app.route('/download_video/<video_id>')
def download_video(video_id):
    """Download the processed video."""
    if video_id not in video_progress:
        return jsonify({"error": "Video ID not found"}), 404

    progress_data = video_progress[video_id]
    if progress_data.get('status') != 'completed':
        return jsonify({"error": "Video processing not completed"}), 400

    output_path = progress_data['result']['output_path']
    if not os.path.exists(output_path):
        return jsonify({"error": "Output file not found"}), 404

    return send_file(output_path, as_attachment=True, download_name=f"segmented_video_{video_id}.mp4")


if __name__ == '__main__':
    print("Starting Flask app for segmentation prediction...")
    print(f"Using device: {config.device}")
    app.run(debug=True, host='0.0.0.0', port=5000)