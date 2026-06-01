<div align="center">

# 🎯 YOLO26 App

**An all-in-one desktop app for annotation, training, and inference based on Ultralytics YOLO26**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/PyQt6-6.0+-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[中文](README.md) · [Features](#-features) · [Quick Start](#-quick-start) · [User Guide](#-user-guide) · [Architecture](#-architecture) · [Development](#-development)

</div>

---

## ✨ Features

### 🏷️ Data Annotation

| Feature | Description |
|---------|-------------|
| **Bounding Box** | Drag to draw detection boxes with any aspect ratio |
| **Polygon** | Click to add points, double-click to complete polygon |
| **SAM 2 Interactive Segmentation** | Click target area to auto-generate segmentation masks, supports SAM 2 models (Hiera-T/S/B+/L) |
| **YOLO Pre-annotation** | Auto-annotate with trained model + class mapping dialog |
| **Grounding DINO** | Zero-shot detection via text prompts (e.g. "person, car") |
| **Video Tracking** | Auto-track annotations across video frames, supports CSRT/KCF/MIL |
| **Batch Detection** | Background thread with progress dialog and cancel support |
| **Undo/Redo** | Ctrl+Z undo, Ctrl+Shift+Z redo, up to 50 steps |
| **Auto-persistence** | Annotations auto-saved to annotations.json, auto-restored on reopen |
| **Keyboard Shortcuts** | ↑↓ keys to quickly navigate between images |
| **Custom Experiment Name** | Customize experiment name during training to distinguish different runs |
| **Class Name Display** | Show class names (e.g. "person") instead of indices |
| **Incremental Rendering** | O(1) updates on add/select, no full redraw |

**Supported Import Methods:**
- Single image (JPG/PNG/BMP etc.)
- Video file (MP4/AVI etc., auto-extract frames)
- Entire directory (batch import)

### 📦 Dataset Export

| Feature | Description |
|---------|-------------|
| **YOLO Format** | Auto-generate images/ + labels/ + data.yaml standard directory structure |
| **Train/Val Split** | Configurable ratio (default 80%), random split |
| **Smart Conversion** | Polygons auto-convert to bounding boxes for detection tasks |
| **Data Validation** | Filter invalid annotations, skip empty images, clean old files before export |

**Export Directory Structure:**
```
output_dir/
├── images/
│   ├── train/           # Training images
│   └── val/             # Validation images
├── labels/
│   ├── train/           # Training labels (.txt)
│   └── val/             # Validation labels (.txt)
└── data.yaml            # Dataset config
```

**YOLO Label Format:**
- Bounding Box: `<class_index> <center_x> <center_y> <width> <height>` (normalized)
- Polygon: `<class_index> <x1> <y1> <x2> <y2> ... <xn> <yn>` (normalized)

### 🏋️ Model Training

| Feature | Description |
|---------|-------------|
| **4 Tasks** | detect / segment / classify / pose |
| **5 Model Sizes** | Nano / Small / Medium / Large / XLarge |
| **Background Training** | QThread async training, responsive UI |
| **Real-time Progress** | Callback-based progress and log updates via Ultralytics |
| **Early Stopping** | Configurable patience parameter |
| **GPU Auto-detection** | Status bar shows GPU/CPU status in real-time |

**Model Size Reference:**

| Model | Params | VRAM | Speed | Use Case |
|-------|--------|------|-------|----------|
| n (Nano) | 3.2M | ≥2GB | ★★★★★ | Edge devices, real-time |
| s (Small) | 11.2M | ≥4GB | ★★★★ | Mobile, lightweight |
| m (Medium) | 25.9M | ≥8GB | ★★★ | General purpose |
| l (Large) | 43.7M | ≥12GB | ★★ | High accuracy |
| x (XLarge) | 68.4M | ≥16GB | ★ | Max accuracy, server |

**Training Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Number of training epochs |
| `batch` | 16 | Batch size (auto-reduce if OOM) |
| `imgsz` | 640 | Input image size |
| `device` | auto | Device: auto/cpu/0/0,1 |
| `optimizer` | auto | Optimizer: auto/SGD/Adam/AdamW |
| `lr0` | 0.01 | Initial learning rate |
| `patience` | 100 | Early stopping patience (0=off) |

### 🔍 Inference & Testing

| Feature | Description |
|---------|-------------|
| **Multiple Sources** | Image / directory / video / USB camera / RealSense |
| **Async Inference** | Background thread, auto frame-skip when inference can't keep up |
| **Depth Display** | RealSense RGB + depth side-by-side |
| **Model Validation** | Background async mAP50/mAP50-95 validation (.pt models only, ONNX etc. will show unsupported message) |
| **Model Export** | Background async export, supports ONNX / TorchScript / OpenVINO / TensorRT, ONNX auto graph optimization + post-export validation |
| **Multi-format Loading** | .pt / .onnx / .torchscript / .xml |
| **Async Image Inference** | Image inference runs in background thread, no UI freeze when loading ONNX models |
| **ONNX Health Check** | Auto-verify ONNX output validity on load, auto-fallback to CPU if GPU fails |
| **ONNX Export Optimization** | Auto-add simplify=True for ONNX export graph optimization |
| **Post-Export Validation** | Auto-verify exported ONNX model can run inference correctly |

**Export Format Comparison:**

| Format | Extension | Use Case |
|--------|-----------|----------|
| ONNX | `.onnx` | Cross-platform deployment |
| TorchScript | `.torchscript` | Native PyTorch deployment |
| OpenVINO | `.xml` | Intel CPU/GPU optimized inference |
| TensorRT | `.engine` | NVIDIA GPU fast inference |

### 🎨 Themes

- **Catppuccin Mocha** — Dark theme, eye-friendly
- **Catppuccin Latte** — Light theme, daytime use

---

## 🚀 Quick Start

## 📦 Resources & Downloads

### Required Dependencies
| Resource | Link | Description |
|----------|------|-------------|
| PyTorch | https://pytorch.org/get-started/locally/ | Required for GPU-accelerated inference, install CUDA version |
| CUDA Toolkit | https://developer.nvidia.com/cuda-toolkit-archive | GPU support, version must match PyTorch |
| Ultralytics | https://docs.ultralytics.com/ | YOLO model framework |

### Optional Dependencies
| Resource | Link | Description |
|----------|------|-------------|
| SAM 2 | https://github.com/facebookresearch/sam2 | Interactive segmentation annotation |
| SAM 2 Model Weights | https://github.com/facebookresearch/segment-anything-2#download-checkpoints | Recommended: sam2.1_hiera_small.pt (184MB) |
| Grounding DINO | https://github.com/IDEA-Research/GroundingDINO | Text-driven zero-shot detection |
| Grounding DINO Weights | https://github.com/IDEA-Research/GroundingDINO#model-zoo | groundingdino_swint_ogc.pth |

### Requirements

- Python 3.9+
- NVIDIA GPU (recommended, CPU works but slower)

### Installation

**1. Clone Repository**

```bash
git clone https://github.com/only-one-over/yolo26_app.git
cd yolo26_app
```

**2. Install Dependencies**

```bash
pip install -r requirements.txt
```

**3. Install PyTorch (GPU Support)**

> ⚠️ **Important**: Do NOT use `pip install torch` — it installs the CPU-only version!

Check your CUDA version:
```bash
nvidia-smi
```

Then install the matching version:

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**4. Optional Dependencies**

```bash
# Intel RealSense depth camera
pip install pyrealsense2

# SAM 2 segmentation
pip install sam2

# Grounding DINO text detection
pip install groundingdino
```

**5. Launch App**

```bash
python main.py
```

### Verify GPU

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

If output shows `CUDA available: True`, the app status bar will display 🟢 GPU: [Device Name].

---

## 📖 User Guide

### Complete Workflow

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ New      │──→│ Import   │──→│ Add      │──→│ Annotate │──→│ Export   │──→│ Train    │──→│ Test     │
│ Project  │   │ Images   │   │ Classes  │   │          │   │ Dataset  │   │ Model    │   │ Inference│
└──────────┘   └──────────┘   └────┬─────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
                                    │               ↑
                                    │    ┌──────────┘
                                    └───→│ Assisted │
                                         │ Annotate │
                                         └──────────┘
```

#### Step 1: New Project

1. Menu → File → New Project
2. Enter project name and path
3. Project directory auto-created:
   ```
   project_path/
   ├── project_config.json      # Project config
   ├── annotations.json         # Annotations (auto-saved)
   ├── classes.txt              # Class list
   ├── datasets/                # Dataset directory
   ├── models/                  # Model directory
   └── runs/                    # Training runs
   ```

#### Step 2: Import Images

- **Import Image** — Select one or more images
- **Import Video** — Select video file, auto-extract frames
- **Import Directory** — Batch import all images

#### Step 3: Add Classes

Click "+" in the class panel to add annotation classes. Each class is auto-assigned a different color.

#### Step 4: Draw Annotations

| Tool | Operation | Shortcut |
|------|-----------|----------|
| Bounding Box | Drag to draw | — |
| Polygon | Click points, double-click to finish | — |
| Select | Click to select | — |
| Delete | Delete selected | Delete |
| Undo | Undo last action | Ctrl+Z |
| Redo | Redo undone action | Ctrl+Shift+Z |

#### Step 5: Assisted Annotation (Optional)

**YOLO Pre-annotation:**
1. Load a trained YOLO model in Test page
2. Click "YOLO Pre-annotate" in Annotate page
3. Class mapping dialog appears
4. Annotations auto-generated after confirmation

**SAM 2 Interactive Segmentation:**
1. Switch to SAM tool mode
2. Click target area to generate mask
3. Supports SAM 2 models: sam2.1_hiera_tiny / sam2.1_hiera_small / sam2.1_hiera_base_plus / sam2.1_hiera_large

**Grounding DINO Text Detection:**
1. Click "Text Detection"
2. Enter text prompt (e.g. "person, car, dog")
3. Auto-detect and annotate

**Video Tracking:**
1. Annotate the first frame
2. Click "Video Tracking"
3. Auto-propagate to subsequent frames

#### Step 6: Export Dataset

1. Click "Export Dataset"
2. Select output directory
3. YOLO dataset auto-generated (images/ + labels/ + data.yaml)

#### Step 7: Train Model

1. Switch to Train page
2. Select data.yaml file
3. Configure training parameters
4. Click "Start Training"
5. Real-time progress and logs
6. Best model saved at `runs/train/weights/best.pt`

#### Step 8: Test Inference

1. Switch to Test page
2. Load trained model (best.pt)
3. Select inference source:
   - Single image
   - Image directory
   - Video file
   - USB camera
   - RealSense camera
4. Optional: Validate mAP metrics
5. Optional: Export model for deployment

---

## 🏗️ Architecture

### Layered Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      UI Layer (ui/)                         │
│  ┌──────────────┬──────────────┬──────────────┐             │
│  │AnnotateWidget│  TrainWidget  │  TestWidget │             │
│  └──────────────┴──────────────┴──────────────┘             │
│                      MainWindow                              │
├─────────────────────────────────────────────────────────────┤
│                   Business Logic Layer (core/)                │
│  ┌────────────┬────────────┬────────────┬────────────┐        │
│  │Annotation  │  Trainer   │ Predictor  │ YOLO      │        │
│  │Canvas      │  (QThread) │            │ Exporter  │        │
│  └────────────┴────────────┴────────────┴────────────┘        │
│  ┌────────────┬────────────┬────────────┬────────────┐        │
│  │Label       │  Project   │Auto        │RealSense   │        │
│  │Manager     │  Manager   │Annotator   │Camera      │        │
│  └────────────┴────────────┴────────────┴────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                    Data Model Layer (config.py)               │
│  ┌──────────────┬──────────────┬──────────────┐             │
│  │  ClassItem  │ TrainConfig  │ProjectConfig│             │
│  └──────────────┴──────────────┴──────────────┘             │
├─────────────────────────────────────────────────────────────┤
│                   External Libraries                          │
│  Ultralytics YOLO26 | PyQt6 | OpenCV | PyTorch | NumPy      │
└─────────────────────────────────────────────────────────────┘
```

### Project Structure

```
yolo26_app/
├── main.py                          # App entry point
├── yolo26_app/                      # Main package
│   ├── core/                        # Core business logic
│   │   ├── config.py                # Data models
│   │   ├── project_manager.py       # Project management
│   │   ├── label_manager.py         # Class management
│   │   ├── annotation_canvas.py     # Annotation canvas + undo/redo
│   │   ├── yolo_exporter.py         # Dataset export
│   │   ├── trainer.py               # Trainer (QThread + callbacks)
│   │   ├── predictor.py             # Predictor
│   │   ├── auto_annotator.py        # Assisted annotation
│   │   └── realsense_camera.py      # RealSense depth camera
│   └── ui/                          # User interface
│       ├── main_window.py           # Main window
│       ├── annotate_widget.py       # Annotation module
│       ├── train_widget.py          # Training module
│       ├── test_widget.py           # Testing module
│       └── styles.py                # Stylesheet (Catppuccin)
├── requirements.txt                 # Dependencies
├── pyproject.toml                   # Project config
├── CODE_WIKI.md                     # Architecture docs
├── LICENSE                          # License
└── README.md                        # Chinese README
```

### Core Data Models

```python
@dataclass
class ClassItem:
    name: str = ""           # Class name
    color: str = "#FF0000"   # Hex color

@dataclass
class TrainConfig:
    task: str = "detect"           # Task type
    model_size: str = "n"          # Model size
    data: str = ""                 # Dataset path
    epochs: int = 100              # Epochs
    batch: int = 16                # Batch size
    imgsz: int = 640               # Image size
    device: str = ""               # Device
    optimizer: str = "auto"        # Optimizer
    lr0: float = 0.01              # Learning rate
    patience: int = 100            # Early stop patience

@dataclass
class ProjectConfig:
    project_name: str = ""
    project_path: str = ""
    classes: List[ClassItem] = field(...)
    train_config: TrainConfig = field(...)
    created_at: str = ""
    last_opened: str = ""

@dataclass
class AnnotationItem:
    class_index: int
    rect: QRectF = field(...)        # Bounding box
    polygon: QPolygonF = field(...)  # Polygon points
    item_type: str = "rect"          # "rect" or "polygon"
```

### Signals & Data Flow

**Cross-module Communication:**
```
MainWindow.project_config ──→ AnnotateWidget / TrainWidget / TestWidget
TestWidget.model_loaded   ──→ AnnotateWidget.set_yolo_model
```

**Training Data Flow:**
```
TrainWidget._on_start
  → YOLOTrainer (QThread)
    → model.train() + on_train_epoch_end callback
    → progress_signal / log_signal / finished_signal / error_signal
  → TrainWidget updates progress bar, logs, results
```

**Inference Data Flow:**
```
TestWidget._on_timer_timeout (video stream)
  → _InferenceWorker (QThread)
    → predictor.predict_frame()
    → result_signal
  → _on_inference_result → _display_np_image
  (auto frame-skip when inference is busy)
```

---

## 🛠️ Development

### Tech Stack

| Tech | Version | Purpose |
|------|---------|---------|
| Python | 3.9+ | Programming language |
| PyQt6 | 6.0+ | GUI framework |
| Ultralytics | 8.0+ | YOLO framework |
| PyTorch | — | Deep learning |
| OpenCV | 4.6+ | Image processing |
| NumPy | 1.20+ | Numerical computing |

### Add New Annotator

1. Create annotator class in `auto_annotator.py`
2. Implement `annotate` method returning `List[AnnotationItem]`
3. Add UI button and logic in `annotate_widget.py`

### Add New Export Format

1. Add export method in `yolo_exporter.py`
2. Implement data conversion logic
3. Add export option in UI

### Add New Inference Source

1. Add input handler in `test_widget.py`
2. Implement frame capture and inference logic
3. Add input button in UI

### Config File Format

**project_config.json:**
```json
{
  "project_name": "my_project",
  "project_path": "/path/to/project",
  "classes": [
    {"name": "person", "color": "#FF6B6B"},
    {"name": "car", "color": "#4ECDC4"}
  ],
  "train_config": {
    "task": "detect",
    "model_size": "n",
    "data": "/path/to/data.yaml",
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "device": "",
    "optimizer": "auto",
    "lr0": 0.01,
    "patience": 100
  },
  "created_at": "2024-01-01T00:00:00",
  "last_opened": "2024-01-02T00:00:00"
}
```

**data.yaml:**
```yaml
path: /path/to/exported/dataset
train: images/train
val: images/val
nc: 2
names: ['person', 'car']
```

---

## ❓ FAQ

### GPU Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Status shows 🔴 CPU | CPU-only PyTorch installed | Install CUDA PyTorch (see [Installation](#3-install-pytorch-gpu-support)) |
| Slow training | Training on CPU | Verify GPU is available, check device parameter |
| CUDA out of memory | Batch or model too large | Reduce batch_size or choose smaller model |

### Training Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `Sizes of tensors must match` | Label format mismatch | Re-export dataset (polygons auto-convert to boxes) |
| Progress stuck | Old version without callback | Use latest version with `on_train_epoch_end` callback |
| data.yaml not found | Wrong path | Verify path, ensure dataset was exported correctly |

### Annotation Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| SAM 2 load failed | SAM 2 not installed or weights missing | Install `sam2`, download SAM 2 weights (sam2.1_hiera_*.pt) |
| Annotations lost | Old version in-memory only | New version auto-persists to annotations.json |
| Canvas lag | Full redraw on many annotations | New version uses incremental rendering |

### ONNX Related

| Issue | Cause | Solution |
|-------|-------|----------|
| ONNX model returns no detections | onnxruntime-gpu version mismatch with CUDA | Install CPU version: `pip uninstall onnxruntime-gpu && pip install onnxruntime` |
| ONNX model validation fails | ONNX format doesn't support validation (val), only .pt supported | Use .pt model for mAP validation |
| App freezes after loading ONNX model | ONNX Runtime first inference initialization blocks main thread | Fixed in latest version: image inference now runs asynchronously |
| Poor inference results after ONNX export | Missing graph optimization during export | Fixed in latest version: simplify=True auto-added |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

> **Note**: This project depends on Ultralytics YOLO26, which is licensed under [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE). If you modify and distribute Ultralytics source code, you must comply with AGPL-3.0.

---

<div align="center">

**If this project helps you, please give it a ⭐ Star!**

</div>
