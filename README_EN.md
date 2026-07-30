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
| **Keypoint Annotation** | Custom keypoint count per class, click to place numbered keypoints with auto-connecting lines, double-click/Enter to finish |
| **OBB (Oriented Bounding Box)** | For annotating rotated objects (aerial, document, shelf, etc.) — drag to define bounding rectangle, then drag again to rotate |
| **SAM 2 Interactive Segmentation** | Click target area to auto-generate segmentation masks, supports SAM 2 models (Hiera-T/S/B+/L) |
| **Grounding DINO** | Zero-shot detection via text prompts (e.g. "person, car") |
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

### YOLO + SAM2 Batch Auto-Annotation Pipeline

Use case: You already have a YOLO detect model + a large number of unlabeled images (common in industrial scenarios). Manual annotation of tens of thousands of images takes dozens of hours; this pipeline compresses it to a few hours.

**Workflow**:
1. Load a YOLO detect model on the Test page
2. Click the "SAM Segmentation" button in the annotation area to load a SAM2 model
3. Import images to be annotated
4. Click the "Batch Detect" toolbar button
5. In the dialog:
   - Set confidence threshold (default 0.25)
   - **Check "Use SAM2 to generate precise masks (polygon)"**
6. Click OK and wait for batch processing (progress dialog is cancelable)
7. Manually review the few incorrect annotations
8. Export a YOLO segmentation dataset

**Prerequisites**:
- A loaded YOLO model (detect or segm both work)
- A loaded SAM2 model (requires `pip install sam2` and downloading a SAM2 checkpoint first)

**Output**: polygon annotations, directly exportable as a YOLO segmentation dataset.

**Notes**:
- When the SAM2 checkbox is unchecked, behavior is identical to the original "Batch Detect" (pure YOLO, only rects or segm-model masks)
- When checked: YOLO predicts bbox → SAM2 uses bbox as a box prompt to generate a mask → simplified to polygon (max 200 points to avoid file bloat)
- Images with no YOLO detections automatically skip SAM2 encoding (saves VRAM)
- A single bbox failure does not interrupt the whole batch; it is skipped with a warning
- Mid-batch cancellation is supported; partial results for already-processed images are preserved

### 📦 Dataset Export

| Feature | Description |
|---------|-------------|
| **YOLO Format** | Auto-generate images/ + labels/ + data.yaml standard directory structure |
| **Train/Val Split** | Configurable ratio (default 80%), random split |
| **Smart Conversion** | Polygons auto-convert to bounding boxes for detection tasks |
| **Data Validation** | Filter invalid annotations, skip empty images, clean old files before export |
| **Pose Export** | Export YOLO pose format dataset with keypoint coordinates and visibility, auto-generates kpt_shape and flip_idx in data.yaml |

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
- OBB (Oriented Bounding Box): `<class_index> <center_x> <center_y> <width> <height> <angle>` (normalized, angle in radians)

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

#### 📈 Training Curves Visualization

The training interface has a built-in training curve visualization panel — no need to manually open TensorBoard or the runs directory:

- **Real-time updates during training** (polls `results.csv` every 5 seconds):
  - Loss curves: `train/box_loss`, `train/cls_loss`, `train/seg_loss` (seg task), `val/box_loss`, `val/cls_loss`, `val/seg_loss`
  - mAP curves: `mAP50`, `mAP50-95`
- **Auto-loaded after training completes**:
  - PR / F1 / P / R curves (`PR_curve.png`, `F1_curve.png`, `P_curve.png`, `R_curve.png`)
  - Confusion matrix (`confusion_matrix.png`, `confusion_matrix_normalized.png`)
- **Open runs directory**: one click to open the `runs/` directory in the system file manager for full charts and weight files
- Depends on `pyqtgraph` (included in core dependencies)

### 🔍 Inference & Testing

| Feature | Description |
|---------|-------------|
| **Multiple Sources** | Image / directory / video / USB camera / RealSense |
| **Async Inference** | Background thread, auto frame-skip when inference can't keep up |
| **Depth Display** | RealSense RGB + depth side-by-side |
| **Model Validation** | Background async mAP50/mAP50-95 validation (.pt models only, ONNX etc. will show unsupported message) |
| **Model Export** | Background async export, supports 10 formats (ONNX/TorchScript/OpenVINO/TensorRT/CoreML/TFLite/NCNN/Paddle/MNN/RKNN), 8 configurable parameters with format-aware visibility, ONNX auto graph optimization + post-export validation |
| **Multi-format Loading** | .pt / .onnx / .torchscript / .xml |
| **Async Image Inference** | Image inference runs in background thread, no UI freeze when loading ONNX models |
| **ONNX Health Check** | Auto-verify ONNX output validity on load, auto-fallback to CPU if GPU fails |
| **Post-Export Validation** | Auto-verify exported ONNX model can run inference correctly |

**Export Format Comparison:**

| Format | Extension | Use Case |
|--------|-----------|----------|
| ONNX | `.onnx` | Cross-platform deployment, supports FP16/INT8/dynamic |
| TorchScript | `.torchscript` | Native PyTorch deployment |
| OpenVINO | `.xml` | Intel CPU/GPU optimized inference |
| TensorRT | `.engine` | NVIDIA GPU fast inference |
| CoreML | `.mlpackage` | Apple device deployment |
| TFLite | `.tflite` | Mobile/embedded deployment |
| NCNN | `_ncnn_model/` | Mobile lightweight inference |
| PaddlePaddle | `_paddle_model/` | Baidu PaddlePaddle ecosystem |
| MNN | `.mnn` | Alibaba MNN inference engine |
| RKNN | `_rknn_model/` | Rockchip NPU deployment |

**Export Parameters:**

| Parameter | Default | Formats | Description |
|-----------|---------|---------|-------------|
| imgsz | 640 | All | Input image size |
| half | False | onnx, engine, openvino, torchscript, tflite, ncnn, mnn | FP16 half-precision |
| int8 | False | onnx, engine, openvino, coreml, tflite, rknn | INT8 quantization |
| dynamic | False | onnx, engine, openvino, torchscript, coreml | Dynamic input size |
| batch | 1 | All | Batch inference size |
| opset | 17 | onnx | ONNX opset version |
| workspace | 4 GiB | engine | TensorRT workspace size |
| simplify | True | onnx | ONNX graph simplification |

> When switching export format, only parameters supported by that format are shown.

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

**2. Create Virtual Environment (Recommended)**

> 💡 **Recommended**: Use a virtual environment to isolate project dependencies and avoid polluting your system Python.

**Option A: Python venv (Standard Python)**

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate
```

**Option B: Anaconda / Miniconda**

```bash
# Create virtual environment (Python 3.10 recommended)
conda create -n yolo26 python=3.10 -y

# Activate virtual environment
conda activate yolo26
```

**3. Install Dependencies**

**Option A: Basic Install (Recommended for Beginners)**

```bash
pip install -r requirements.txt
```

**Option B: Locked Version Install (Recommended for Production, Versions Verified Compatible)**

```bash
# Install PyTorch first (choose based on your CUDA version, see below)
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
# Then install the rest of the locked dependencies
pip install -r requirements-lock.txt
```

Locked version combination: Python 3.10 + PyTorch 2.3.1 + CUDA 12.1 + Ultralytics 8.3.20 + TensorRT 10.2.0 + PyQt6 6.7.1 + OpenCV 4.10.0.84 + ONNX Runtime 1.18.1.

**4. Install PyTorch (GPU Support)**

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

**5. Optional Dependencies**

```bash
# Intel RealSense depth camera support
pip install -e ".[realsense]"

# SAM 2 segmentation support
pip install -e ".[sam]"

# Grounding DINO support
pip install -e ".[dino]"

# TensorRT ultra-fast GPU inference (requires Ultralytics ≥ 8.3.0)
pip install -e ".[tensorrt]"

# All optional dependencies
pip install -e ".[all]"
```

**6. Launch App**

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

Click "+" in the class panel to add annotation classes. Each class is auto-assigned a different color. Keypoint count can be set when adding a class (0 = no keypoints), displayed as "person (17pt)" in class list.

#### Step 4: Draw Annotations

| Tool | Operation | Shortcut |
|------|-----------|----------|
| Bounding Box | Drag to draw | — |
| Polygon | Click points, double-click to finish | — |
| Keypoint | Click to place keypoints, double-click/Enter to finish | — |
| OBB | Drag to define bounding rectangle, then drag again to rotate | — |
| Select | Click to select | — |
| Delete | Delete selected via Delete or Space | Delete / Space |
| Undo | Undo last action | Ctrl+Z |
| Redo | Redo undone action | Ctrl+Shift+Z |
| Previous | Switch to previous image | ↑ |
| Next | Switch to next image | ↓ or Shift+Space |
| Switch to Rectangle tool | Switch to Rectangle tool mode | R |
| Switch to Polygon tool | Switch to Polygon tool mode | P |
| Switch to OBB tool | Switch to OBB tool mode | O |
| Switch to Keypoint tool | Switch to Keypoint tool mode | K |
| Switch to Select tool | Switch to Select tool mode | S |

#### Polygon Vertex Editing

After selecting a polygon annotation, vertices can be fine-edited:

- **Drag vertex**: hold and drag a vertex handle to update polygon shape in real time
- **Right-click to delete vertex**: right-click on a vertex handle to remove it (minimum 3 vertices retained)
- **Double-click edge to add vertex**: double-click on a polygon edge to insert a new vertex at that position
- All editing operations support Ctrl+Z undo

#### Step 5: Assisted Annotation (Optional)

**SAM 2 Interactive Segmentation:**
1. Switch to SAM tool mode
2. Click target area to generate mask
3. Supports SAM 2 models: sam2.1_hiera_tiny / sam2.1_hiera_small / sam2.1_hiera_base_plus / sam2.1_hiera_large

**Grounding DINO Text Detection:**
1. Click "Text Detection"
2. Enter text prompt (e.g. "person, car, dog")
3. Auto-detect and annotate

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
├── main.py                          # Application entry point (python main.py)
├── code/                            # ✅ Code core (users only need to update this folder)
│   └── yolo26_app/                  # Main package
│       ├── core/                    # Core business logic
│       │   ├── config.py            # Data models (ClassItem, TrainConfig, ProjectConfig)
│       │   ├── paths.py             # Workspace path constants (system_model/my_project)
│       │   ├── project_manager.py   # Project management (create/open/recent/paths)
│       │   ├── label_manager.py     # Annotation class management
│       │   ├── annotation_canvas.py # Annotation canvas (Scene + View + undo/redo)
│       │   ├── yolo_exporter.py     # YOLO dataset export
│       │   ├── trainer.py           # YOLO trainer (QThread + callback progress)
│       │   ├── predictor.py         # YOLO predictor (load/infer/validate/export)
│       │   ├── auto_annotator.py    # Assisted annotation (SAM/DINO)
│       │   ├── gpu_detector.py      # GPU detection (async/timeout/cache/safe mode)
│       │   ├── task_manager.py      # Background task manager
│       │   ├── model_registry.py    # Model/augmentation constants
│       │   ├── persistence.py       # Atomic write utilities
│       │   ├── workspace_manager.py # Workspace management
│       │   ├── realsense_camera.py  # RealSense depth camera
│       │   ├── config_template.yaml # Default config template
│       │   └── utils/               # Common utilities (unicode path image I/O)
│       └── ui/                      # User interface
│           ├── main_window.py       # Main window (async GPU/safe mode/lazy load)
│           ├── annotation.py        # Annotation module
│           ├── training.py          # Training module
│           ├── inference.py         # Inference module
│           ├── export_dialog.py     # Export dialog
│           ├── styles.py            # QSS styles (Catppuccin Mocha/Latte)
│           └── icons/               # SVG icon resources
├── system_model/                    # System models (auto-created, gitignored)
│   ├── yolo/                        # Pretrained YOLO models
│   ├── sam2/                        # SAM2 model weights
│   ├── grounding_dino/              # GroundingDINO model weights
│   └── user_trained/                # User-trained models
├── my_project/                      # User projects (auto-created, gitignored)
│   ├── default/                     # Default workspace (free-space mode)
│   └── project1/                    # User workspace
├── requirements.txt                 # Dependencies
├── pyproject.toml                   # Project config
├── LICENSE                          # MIT License
└── README.md                        # This file
```

> 💡 **Code core is in the `code/` folder**. To update, users only need to replace the `code/` folder without affecting models (`system_model/`), user data (`my_project/`), and configs.

### Core Data Models

```python
@dataclass
class ClassItem:
    name: str = ""           # Class name
    color: str = "#FF0000"   # Hex color
    kpt_count: int = 0       # Keypoint count (0 = no keypoints)

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
    item_type: str = "rect"          # "rect", "polygon", or "keypoint"
    keypoints: List[QPointF] = field(default_factory=list)  # Keypoint list
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

## 🔍 Diagnostics & Troubleshooting

### Log Location

Application runtime logs are automatically written to the `logs/` folder in the project root:
- `logs/app_YYYYMMDD.log` — daily rotation, retained for 7 days
- `logs/crash_YYYYMMDD_HHMMSS.log` — crash log (auto-generated when the program exits abnormally)

### Export Diagnostic Report

When you encounter an issue and need to report it, you can export a diagnostic report in one click:

1. Menu bar → Help → Export Diagnostic Report
2. Choose a save path (default: `my_project/<workspace>/diagnostic_report_YYYYMMDD_HHMMSS.zip`)
3. The report includes:
   - System information (OS, Python, PyQt6, OpenCV, PyTorch, CUDA, Ultralytics, TensorRT versions)
   - GPU status
   - Logs from the last 7 days

The diagnostic report can be attached directly to an Issue, helping developers quickly locate the problem.

---

## ❓ FAQ

### GPU Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| Status shows 🔴 CPU | CPU-only PyTorch installed | Install CUDA PyTorch (see [Installation](#4-install-pytorch-gpu-support)) |
| Slow training | Training on CPU | Verify GPU is available, check device parameter |
| CUDA out of memory | Batch or model too large | Reduce batch_size or choose smaller model |
| Status shows 🔴 CPU (Safe Mode) | App didn't exit normally last time | Close the app normally; safe mode skips GPU detection |
| Status shows 🔴 CPU (Detection Timeout) | CUDA driver hang caused detection timeout | Check CUDA driver health, restart the app |
| Startup shows "⏳ Detecting..." for a long time | GPU detection runs in background, first start is slower | Normal behavior; detection result is cached for 30 minutes |

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
