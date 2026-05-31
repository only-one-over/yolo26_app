<div align="center">

# 🎯 YOLO26 App

**基于 Ultralytics YOLO26 的桌面端标注-训练-推理一体化应用**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/PyQt6-6.0+-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README_EN.md) · [功能特性](#-功能特性) · [快速开始](#-快速开始) · [使用指南](#-使用指南) · [项目架构](#-项目架构) · [开发指南](#-开发指南)

</div>

---

## ✨ 功能特性

### 🏷️ 数据标注

| 功能 | 描述 |
|------|------|
| **矩形框标注** | 拖拽绘制目标检测框，支持任意宽高比 |
| **多边形标注** | 逐点绘制分割掩码，双击完成多边形 |
| **SAM 2 交互式分割** | 点击目标区域自动生成分割掩码，支持 SAM 2 模型（Hiera-T/S/B+/L） |
| **YOLO 预标注** | 使用已训练的 YOLO 模型自动生成标注，支持类别映射对话框 |
| **Grounding DINO** | 输入文本描述（如 "person, car"）进行零样本检测 |
| **视频帧间追踪** | 标注首帧后自动追踪后续帧，支持 CSRT/KCF/MIL 追踪器 |
| **批量检测** | 后台线程逐帧检测，进度对话框 + 取消支持 |
| **撤销/重做** | Ctrl+Z 撤销，Ctrl+Shift+Z 重做，最多 50 步历史 |
| **自动持久化** | 标注数据自动保存到项目目录 annotations.json，切换图片/重新打开自动恢复 |
| **键盘快捷键切换图片** | ↑↓ 键快速切换上一张/下一张图片 |
| **自定义实验名称** | 训练时可自定义实验名称，便于区分不同训练运行 |
| **类别名称显示** | 标注标签显示类别名称（如 "person"）而非索引号 |
| **增量绘制** | 添加/选择标注时 O(1) 更新，不全量重绘，大量标注不卡顿 |

**支持的导入方式：**
- 单张图片（JPG/PNG/BMP 等）
- 视频文件（MP4/AVI 等，自动提取帧）
- 整个目录（批量导入所有图片）

### 📦 数据集导出

| 功能 | 描述 |
|------|------|
| **YOLO 格式导出** | 自动生成 images/ + labels/ + data.yaml 标准目录结构 |
| **训练/验证集划分** | 可配置训练集比例（默认 80%），自动随机划分 |
| **智能格式转换** | detect 任务下多边形自动转为外接矩形框，segment 任务保留原始多边形 |
| **数据校验** | 过滤无效标注（零尺寸、点数不足），跳过无标注图片，导出前清空旧文件 |

**导出目录结构：**
```
output_dir/
├── images/
│   ├── train/           # 训练集图片
│   └── val/             # 验证集图片
├── labels/
│   ├── train/           # 训练集标签 (.txt)
│   └── val/             # 验证集标签 (.txt)
└── data.yaml            # 数据集配置文件
```

**YOLO 标签格式：**
- 矩形框：`<class_index> <center_x> <center_y> <width> <height>`（归一化坐标）
- 多边形：`<class_index> <x1> <y1> <x2> <y2> ... <xn> <yn>`（归一化坐标）

### 🏋️ 模型训练

| 功能 | 描述 |
|------|------|
| **四种任务** | 目标检测 (detect)、实例分割 (segment)、图像分类 (classify)、姿态估计 (pose) |
| **五种模型大小** | Nano / Small / Medium / Large / XLarge，适配不同显存和速度需求 |
| **后台线程训练** | QThread 异步训练，UI 保持响应 |
| **实时进度** | 通过 Ultralytics 回调机制实时更新 Epoch 进度条和日志 |
| **早停机制** | 可配置 patience 参数，验证指标无提升时自动停止 |
| **GPU 自动检测** | 状态栏实时显示 GPU/CPU 状态 |

**模型大小参考：**

| 模型大小 | 参数量 | 显存需求 | 速度 | 适用场景 |
|---------|--------|---------|------|---------|
| n (Nano) | 3.2M | ≥2GB | ★★★★★ | 边缘设备、实时检测 |
| s (Small) | 11.2M | ≥4GB | ★★★★ | 移动端、轻量部署 |
| m (Medium) | 25.9M | ≥8GB | ★★★ | 通用场景、精度与速度平衡 |
| l (Large) | 43.7M | ≥12GB | ★★ | 高精度需求 |
| x (XLarge) | 68.4M | ≥16GB | ★ | 最高精度、服务器部署 |

**训练参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epochs` | 100 | 训练轮数 |
| `batch` | 16 | 批大小（显存不足时自动降低） |
| `imgsz` | 640 | 输入图像尺寸 |
| `device` | auto | 设备：auto/cpu/0/0,1 |
| `optimizer` | auto | 优化器：auto/SGD/Adam/AdamW |
| `lr0` | 0.01 | 初始学习率 |
| `patience` | 100 | 早停耐心值（0 表示不早停） |

### 🔍 推理测试

| 功能 | 描述 |
|------|------|
| **多种输入源** | 单张图片、图片目录、视频文件、USB 摄像头、Intel RealSense 深度相机 |
| **异步推理** | 视频推理在后台线程执行，推理跟不上帧率时自动跳帧，避免延迟堆积 |
| **深度图显示** | RealSense 相机支持彩色图 + 深度图并排显示 |
| **模型验证** | 后台异步执行 mAP50/mAP50-95 指标验证 |
| **模型导出** | 后台异步导出，支持 ONNX / TorchScript / OpenVINO / TensorRT |
| **多格式加载** | 支持 .pt / .onnx / .torchscript / .xml 等格式模型 |

**导出格式对比：**

| 格式 | 扩展名 | 适用场景 |
|------|--------|---------|
| ONNX | `.onnx` | 通用跨平台部署 |
| TorchScript | `.torchscript` | PyTorch 原生部署 |
| OpenVINO | `.xml` | Intel CPU/GPU 优化推理 |
| TensorRT | `.engine` | NVIDIA GPU 极速推理 |

### 🎨 界面主题

- **Catppuccin Mocha** — 深色主题，护眼舒适
- **Catppuccin Latte** — 亮色主题，白天使用

---

## 🚀 快速开始

## 📦 资源与下载

### 必需依赖
| 资源 | 链接 | 说明 |
|------|------|------|
| PyTorch | https://pytorch.org/get-started/locally/ | GPU 加速推理必需，需安装 CUDA 版本 |
| CUDA Toolkit | https://developer.nvidia.com/cuda-toolkit-archive | GPU 支持，版本需与 PyTorch 匹配 |
| Ultralytics | https://docs.ultralytics.com/ | YOLO 模型框架 |

### 可选依赖
| 资源 | 链接 | 说明 |
|------|------|------|
| SAM 2 | https://github.com/facebookresearch/sam2 | 交互式分割标注 |
| SAM 2 模型权重 | https://github.com/facebookresearch/segment-anything-2#download-checkpoints | 推荐 sam2.1_hiera_small.pt (184MB) |
| Grounding DINO | https://github.com/IDEA-Research/GroundingDINO | 文本驱动零样本检测 |
| Grounding DINO 权重 | https://github.com/IDEA-Research/GroundingDINO#model-zoo | groundingdino_swint_ogc.pth |

### 环境要求

- Python 3.9+
- NVIDIA GPU（推荐，CPU 也可运行但速度较慢）

### 安装步骤

**1. 克隆仓库**

```bash
git clone https://github.com/only-one-over/yolo26_app.git
cd yolo26_app
```

**2. 安装依赖**

```bash
pip install -r requirements.txt
```

**3. 安装 PyTorch（GPU 支持）**

> ⚠️ **重要**：不要使用 `pip install torch` 默认安装，那会安装 CPU-only 版本，即使有 GPU 也无法使用！

先检查你的 CUDA 版本：
```bash
nvidia-smi
```
右上角显示 CUDA 版本（如 12.1、11.8），然后安装对应版本：

```bash
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**4. 安装可选依赖**

```bash
# Intel RealSense 深度相机支持
pip install pyrealsense2

# SAM 2 分割支持
pip install sam2

# Grounding DINO 支持
pip install groundingdino
```

**5. 启动应用**

```bash
python main.py
```

### 验证 GPU 是否可用

```bash
python -c "import torch; print('CUDA可用:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')"
```

如果输出 `CUDA可用: True` 和你的 GPU 名称，说明安装成功。应用状态栏会显示 🟢 GPU: [设备名称]。

---

## 📖 使用指南

### 完整工作流

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  新建项目 │───→│  导入图片 │───→│  添加类别 │───→│  绘制标注 │───→│ 导出数据集│───→│  训练模型 │───→│  测试推理 │
└─────────┘    └─────────┘    └────┬────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
                                   │              ↑
                                   │    ┌─────────┘
                                   └───→│ 辅助标注  │
                                        │ (可选加速) │
                                        └─────────┘
```

#### Step 1: 新建项目

1. 菜单栏 → 文件 → 新建项目
2. 输入项目名称和存储路径
3. 项目目录自动创建：
   ```
   project_path/
   ├── project_config.json      # 项目配置
   ├── annotations.json         # 标注数据（自动保存）
   ├── classes.txt              # 类别列表
   ├── datasets/                # 数据集目录
   ├── models/                  # 模型目录
   └── runs/                    # 训练运行记录
   ```

#### Step 2: 导入图片

在标注页面，支持三种导入方式：
- **导入图片** — 选择单张或多张图片
- **导入视频** — 选择视频文件，自动提取帧
- **导入目录** — 选择整个目录，批量导入所有图片

#### Step 3: 添加类别

在左侧类别面板点击 "+" 添加标注类别，每个类别自动分配不同颜色。

#### Step 4: 绘制标注

| 工具 | 操作方式 | 快捷键 |
|------|---------|--------|
| 矩形框 | 按住鼠标拖拽 | — |
| 多边形 | 逐点点击，双击完成 | — |
| 选择 | 点击标注选中 | — |
| 删除 | 选中后按 Delete | Delete |
| 撤销 | 撤销上一步操作 | Ctrl+Z |
| 重做 | 重做被撤销的操作 | Ctrl+Shift+Z |

#### Step 5: 辅助标注（可选）

加速标注的四种方式：

**YOLO 预标注：**
1. 在测试页面加载一个已训练的 YOLO 模型
2. 回到标注页面，点击"YOLO 预标注"
3. 弹出类别映射对话框，将模型类别映射到项目类别
4. 确认后自动生成标注

**SAM 2 交互式分割：**
1. 切换到 SAM 工具模式
2. 点击目标区域，自动生成分割掩码
3. 支持 SAM 2 模型：sam2.1_hiera_tiny / sam2.1_hiera_small / sam2.1_hiera_base_plus / sam2.1_hiera_large

**Grounding DINO 文本检测：**
1. 点击"文本检测"按钮
2. 输入文本描述（如 "person, car, dog"）
3. 自动检测并生成标注

**视频帧间追踪：**
1. 标注视频的第一帧
2. 点击"视频追踪"按钮
3. 自动将标注传播到后续帧

#### Step 6: 导出数据集

1. 点击"导出数据集"按钮
2. 选择输出目录
3. 自动生成 YOLO 格式数据集（images/ + labels/ + data.yaml）

#### Step 7: 训练模型

1. 切换到训练页面
2. 选择 data.yaml 文件
3. 配置训练参数（任务类型、模型大小、Epochs 等）
4. 点击"开始训练"
5. 实时查看进度条和日志输出
6. 训练完成后，最佳模型保存在 `runs/train/weights/best.pt`

#### Step 8: 测试推理

1. 切换到测试页面
2. 加载训练好的模型（best.pt）
3. 选择推理输入源：
   - 单张图片 → 查看单帧推理结果
   - 图片目录 → 批量推理
   - 视频文件 → 实时视频推理
   - USB 摄像头 → 实时摄像头推理
   - RealSense 相机 → RGB + 深度图推理
4. 可选：点击"验证模型"查看 mAP 指标
5. 可选：点击"导出模型"转换为部署格式

---

## 🏗️ 项目架构

### 架构分层

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

### 项目结构

```
yolo26_app/
├── main.py                          # 应用入口
├── yolo26_app/                      # 应用主包
│   ├── core/                        # 核心业务逻辑
│   │   ├── config.py                # 数据模型 (ClassItem, TrainConfig, ProjectConfig)
│   │   ├── project_manager.py       # 项目管理（创建/打开/最近项目/路径）
│   │   ├── label_manager.py         # 标注类别管理（增删改查/颜色分配）
│   │   ├── annotation_canvas.py     # 标注画布 (Scene + View + 撤销/重做 + 增量绘制)
│   │   ├── yolo_exporter.py         # YOLO 数据集导出（格式转换/校验/划分）
│   │   ├── trainer.py               # YOLO 训练器 (QThread + 回调进度)
│   │   ├── predictor.py             # YOLO 推理器（加载/推理/验证/导出）
│   │   ├── auto_annotator.py        # 辅助标注 (YOLO预标注/SAM/DINO/视频追踪)
│   │   └── realsense_camera.py      # RealSense 深度相机（设备枚举/帧获取/深度着色）
│   └── ui/                          # 用户界面
│       ├── main_window.py           # 主窗口 (GPU状态/项目管理/页面导航)
│       ├── annotate_widget.py       # 标注模块 (持久化/批量检测/类别映射/辅助标注)
│       ├── train_widget.py          # 训练模块 (参数配置/进度显示/日志)
│       ├── test_widget.py           # 测试模块 (异步推理/验证/导出/帧跳过)
│       └── styles.py                # QSS 样式表 (Catppuccin Mocha/Latte)
├── requirements.txt                 # 依赖清单
├── pyproject.toml                   # 项目配置
├── CODE_WIKI.md                     # 详细架构文档
├── LICENSE                          # MIT 许可证
└── README.md                        # 本文件
```

### 核心数据模型

```python
# 标注类别
@dataclass
class ClassItem:
    name: str = ""           # 类别名称
    color: str = "#FF0000"   # 十六进制颜色值

# 训练配置
@dataclass
class TrainConfig:
    task: str = "detect"           # 任务类型
    model_size: str = "n"          # 模型大小
    data: str = ""                 # 数据集配置文件路径
    epochs: int = 100              # 训练轮数
    batch: int = 16                # 批大小
    imgsz: int = 640               # 输入图像尺寸
    device: str = ""               # 设备 (auto/cpu/0/0,1)
    optimizer: str = "auto"        # 优化器
    lr0: float = 0.01              # 初始学习率
    patience: int = 100            # 早停耐心值

# 项目配置
@dataclass
class ProjectConfig:
    project_name: str = ""
    project_path: str = ""
    classes: List[ClassItem] = field(...)
    train_config: TrainConfig = field(...)
    created_at: str = ""
    last_opened: str = ""

# 标注项
@dataclass
class AnnotationItem:
    class_index: int
    rect: QRectF = field(...)        # 矩形区域
    polygon: QPolygonF = field(...)  # 多边形点集
    item_type: str = "rect"          # "rect" 或 "polygon"
```

### 信号与数据流

**跨模块通信：**
```
MainWindow.project_config ──→ AnnotateWidget / TrainWidget / TestWidget
TestWidget.model_loaded   ──→ AnnotateWidget.set_yolo_model
```

**训练数据流：**
```
TrainWidget._on_start
  → YOLOTrainer (QThread)
    → model.train() + on_train_epoch_end 回调
    → progress_signal / log_signal / finished_signal / error_signal
  → TrainWidget 更新进度条、日志、结果
```

**推理数据流：**
```
TestWidget._on_timer_timeout (视频流)
  → _InferenceWorker (QThread)
    → predictor.predict_frame()
    → result_signal
  → _on_inference_result → _display_np_image
  (推理忙时自动跳帧)
```

---

## 🛠️ 开发指南

### 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.9+ | 编程语言 |
| PyQt6 | 6.0+ | GUI 框架 |
| Ultralytics | 8.0+ | YOLO 模型框架 |
| PyTorch | — | 深度学习框架 |
| OpenCV | 4.6+ | 图像处理 |
| NumPy | 1.20+ | 数值计算 |

### 添加新的辅助标注器

1. 在 `auto_annotator.py` 中创建新的标注器类
2. 实现 `annotate` 方法，返回 `List[AnnotationItem]`
3. 在 `annotate_widget.py` 中添加对应的 UI 按钮和调用逻辑

### 添加新的导出格式

1. 在 `yolo_exporter.py` 中添加新的导出方法
2. 实现数据转换逻辑
3. 在 UI 中添加导出选项

### 添加新的推理输入源

1. 在 `test_widget.py` 中添加新的输入处理方法
2. 实现帧获取和推理逻辑
3. 更新 UI 添加对应的输入按钮

### 配置文件格式

**project_config.json：**
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

**data.yaml：**
```yaml
path: /path/to/exported/dataset
train: images/train
val: images/val
nc: 2
names: ['person', 'car']
```

---

## ❓ 常见问题

### GPU 相关

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 状态栏显示 🔴 CPU | PyTorch 未安装或安装了 CPU 版本 | 安装 CUDA 版 PyTorch（见[安装步骤](#3-安装-pytorchgpu-支持)） |
| 训练很慢 | 使用了 CPU 训练 | 确认 GPU 可用，检查 device 参数 |
| CUDA out of memory | batch 太大或模型太大 | 减小 batch_size 或选择更小的模型 |

### 训练相关

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `Sizes of tensors must match` | 标签格式不匹配（多边形用于 detect） | 重新导出数据集，多边形会自动转为矩形框 |
| 进度条不动 | 旧版本未实现回调 | 确保使用最新版本，训练器通过 `on_train_epoch_end` 回调更新进度 |
| 数据集配置文件不存在 | data.yaml 路径错误 | 检查路径，确保导出数据集后使用正确的 data.yaml |

### 标注相关

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| SAM 2 模型加载失败 | SAM 2 未安装或权重文件缺失 | 安装 `sam2`，下载 SAM 2 权重（sam2.1_hiera_*.pt） |
| 标注丢失 | 旧版本标注仅存在内存中 | 新版本自动持久化到 annotations.json |
| 画面卡顿 | 大量标注时全量重绘 | 新版本使用增量绘制，仅更新变化的标注 |

---

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

> **注意**：本项目依赖 Ultralytics YOLO26，其采用 [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 许可证。如果你修改并分发 Ultralytics 源码，需遵守 AGPL-3.0 的要求。

---

<div align="center">

**如果这个项目对你有帮助，请给个 ⭐ Star 支持一下！**

</div>
