<div align="center">

# 🎯 YOLO26 App

**基于 Ultralytics YOLO 的桌面端标注-训练-推理一体化应用（支持 YOLO26 / YOLOv8）**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/PyQt6-6.0+-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO26%2Fv8-orange.svg)](https://github.com/ultralytics/ultralytics)
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
| **关键点标注** | 支持自定义关键点数量，点击放置带编号关键点，自动连线，双击/Enter完成 |
| **OBB 旋转框标注** | 用于标注旋转目标（航拍、文档、货架等），先拖拽确定外接矩形，再拖拽旋转确定角度 |
| **SAM 2 交互式分割** | 点击目标区域自动生成分割掩码，支持 SAM 2 模型（Hiera-T/S/B+/L） |
| **Grounding DINO** | 输入文本描述（如 "person, car"）进行零样本检测 |
| **批量检测** | 后台线程逐帧检测，进度对话框 + 取消支持 |
| **撤销/重做** | Ctrl+Z 撤销，Ctrl+Shift+Z 重做，最多 50 步历史 |
| **自动持久化** | 标注数据自动保存到项目目录 annotations.json，切换图片/重新打开自动恢复 |
| **关键点持久化** | 关键点标注保存/加载完整支持，项目文件包含 keypoints 字段 |
| **类别持久化** | 添加/删除类别后自动保存到项目配置 |
| **键盘快捷键切换图片** | ↑↓ 键快速切换上一张/下一张图片 |
| **自定义实验名称** | 训练时可自定义实验名称，便于区分不同训练运行 |
| **类别名称显示** | 标注标签显示类别名称（如 "person"）而非索引号 |
| **增量绘制** | 添加/选择标注时 O(1) 更新，不全量重绘，大量标注不卡顿 |

**支持的导入方式：**
- 单张图片（JPG/PNG/BMP 等）
- 视频文件（MP4/AVI 等，自动提取帧）
- 整个目录（批量导入所有图片）

### YOLO + SAM2 批量自动标注流水线

适用场景:已有 YOLO detect 模型 + 大量未标注图片(工业场景常见),人工标注上万张图耗时数十小时,本流水线可压到数小时。

**工作流程**:
1. 在测试页面加载 YOLO detect 模型
2. 在标注区点击「SAM 分割」按钮加载 SAM2 模型
3. 导入待标注图片
4. 点击工具栏「逐帧检测」按钮
5. 在弹出对话框中:
   - 设置置信度阈值(默认 0.25)
   - **勾选「使用 SAM2 生成精确掩码(polygon)」**
6. 点击 OK,等待批量处理完成(进度对话框可取消)
7. 人工复核少量错误标注
8. 导出 YOLO segmentation 数据集

**前置条件**:
- 已加载 YOLO 模型(detect 或 segm 均可)
- 已加载 SAM2 模型(需先 `pip install sam2` 并下载 SAM2 checkpoint)

**输出**:polygon 类型标注,可直接导出 YOLO segmentation 格式数据集。

**说明**:
- 不勾选 SAM2 复选框时,行为与原「逐帧检测」完全一致(纯 YOLO,仅出 rect 或 segm 模型的 mask)
- 勾选后:YOLO 预测 bbox → SAM2 用 bbox 作为 box prompt 生成 mask → 简化为 polygon(点数上限 200,避免文件膨胀)
- 单张图 YOLO 无检测时自动跳过 SAM2 编码(节省显存)
- 单个 bbox 处理失败不中断整批流程,仅 warning 跳过
- 支持中途取消,已处理图片的部分结果会保留

### 📦 数据集导出

| 功能 | 描述 |
|------|------|
| **YOLO 格式导出** | 自动生成 images/ + labels/ + data.yaml 标准目录结构 |
| **训练/验证集划分** | 可配置训练集比例（默认 80%），自动随机划分 |
| **智能格式转换** | detect 任务下多边形自动转为外接矩形框，segment 任务保留原始多边形 |
| **Pose 导出** | 导出 YOLO pose 格式数据集，标签包含关键点坐标和可见性，data.yaml 自动生成 kpt_shape 和 flip_idx |
| **数据校验** | 过滤无效标注（零尺寸、点数不足），跳过无标注图片，导出前清空旧文件 |
| **导出安全保护** | 输出目录非空时自动创建带时间戳子目录，不删除已有文件 |
| **按任务严格导出** | segment 只导出 polygon，detect 只导出 bbox，pose 只导出 bbox+关键点 |
| **INT8 校验** | INT8 导出强制要求校准数据 data.yaml，未选则阻止导出 |

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
- OBB 旋转框：`<class_index> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>`（归一化坐标，四个角点）

### 🏋️ 模型训练

| 功能 | 描述 |
|------|------|
| **四种任务** | 目标检测 (detect)、实例分割 (segment)、图像分类 (classify)、姿态估计 (pose) |
| **五种模型大小** | Nano / Small / Medium / Large / XLarge，适配不同显存和速度需求 |
| **后台线程训练** | QThread 异步训练，UI 保持响应 |
| **实时进度** | 通过 Ultralytics 回调机制实时更新 Epoch 进度条和日志 |
| **早停机制** | 可配置 patience 参数，验证指标无提升时自动停止 |
| **GPU 自动检测** | 状态栏实时显示 GPU/CPU 状态 |
| **YOLOv8 支持** | 训练页支持 YOLO26 / YOLOv8 模型族切换，自定义模型路径 |
| **数据增强配置** | 4 档预设（关闭/轻度/默认/强增强）+ 15 个可调增强参数 |
| **训练配置持久化** | 训练参数自动保存到项目配置，重新打开恢复 |

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
| `model_family` | yolo26 | 模型族：yolo26 / yolov8 |
| `augmentation_preset` | 默认 | 增强预设：关闭/轻度/默认/强增强/自定义 |

#### 📈 训练曲线可视化

训练界面内置训练曲线可视化面板,无需手动打开 TensorBoard 或 runs 目录:

- **训练中实时更新**(每 5 秒轮询 `results.csv`):
  - Loss 曲线:`train/box_loss`、`train/cls_loss`、`train/seg_loss`(seg 任务)、`val/box_loss`、`val/cls_loss`、`val/seg_loss`
  - mAP 曲线:`mAP50`、`mAP50-95`
- **训练完成后自动加载**:
  - PR / F1 / P / R 曲线(`PR_curve.png`、`F1_curve.png`、`P_curve.png`、`R_curve.png`)
  - 混淆矩阵(`confusion_matrix.png`、`confusion_matrix_normalized.png`)
- **打开 runs 目录**:一键在系统文件管理器中打开 `runs/` 目录,访问完整图表与权重文件
- 依赖 `pyqtgraph`(已纳入核心依赖)

### 🔍 推理测试

| 功能 | 描述 |
|------|------|
| **多种输入源** | 单张图片、图片目录、视频文件、USB 摄像头、Intel RealSense 深度相机 |
| **异步推理** | 视频推理在后台线程执行，推理跟不上帧率时自动跳帧，避免延迟堆积 |
| **深度图显示** | RealSense 相机支持彩色图 + 深度图并排显示 |
| **模型验证** | 后台异步执行 mAP50/mAP50-95 指标验证（仅支持 .pt 模型，ONNX 等格式会提示不支持） |
| **模型导出** | 后台异步导出，支持 10 种格式（ONNX/TorchScript/OpenVINO/TensorRT/CoreML/TFLite/NCNN/Paddle/MNN/RKNN），8 个可配置参数，参数联动，ONNX 自动图优化 + 导出后验证 |
| **多格式加载** | 支持 .pt / .onnx / .torchscript / .xml 等格式模型 |
| **异步图片推理** | 图片推理在后台线程执行，加载 ONNX 模型时不再卡死 UI |
| **ONNX 健康检查** | 加载 ONNX 模型时自动验证输出有效性，GPU 异常自动回退 CPU |
| **导出后验证** | 导出 ONNX 后自动验证模型可正常推理 |

**导出格式对比：**

| 格式 | 扩展名 | 适用场景 |
|------|--------|---------|
| ONNX | `.onnx` | 通用跨平台部署，支持 FP16/INT8/动态尺寸 |
| TorchScript | `.torchscript` | PyTorch 原生部署 |
| OpenVINO | `.xml` | Intel CPU/GPU 优化推理 |
| TensorRT | `.engine` | NVIDIA GPU 极速推理 |
| CoreML | `.mlpackage` | Apple 设备部署 |
| TFLite | `.tflite` | 移动端/嵌入式部署 |
| NCNN | `_ncnn_model/` | 移动端轻量推理 |
| PaddlePaddle | `_paddle_model/` | 百度飞桨生态 |
| MNN | `.mnn` | 阿里 MNN 推理引擎 |
| RKNN | `_rknn_model/` | 瑞芯微 NPU 部署 |

**导出参数配置：**

| 参数 | 默认值 | 适用格式 | 说明 |
|------|--------|---------|------|
| imgsz | 640 | 全部 | 输入图像尺寸 |
| half | False | onnx, engine, openvino, torchscript, tflite, ncnn, mnn | FP16 半精度量化 |
| int8 | False | onnx, engine, openvino, coreml, tflite, rknn | INT8 量化 |
| dynamic | False | onnx, engine, openvino, torchscript, coreml | 动态输入尺寸 |
| batch | 1 | 全部 | 批量推理大小 |
| opset | 17 | onnx | ONNX opset 版本 |
| workspace | 4 GiB | engine | TensorRT 工作空间大小 |
| simplify | True | onnx | ONNX 图简化 |

> 切换导出格式时，仅显示该格式支持的参数控件。

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

- Python 3.9+（推荐 3.10/3.11，3.12 部分依赖可能不兼容）
- NVIDIA GPU（推荐，CPU 也可运行但速度较慢）
- CUDA Toolkit 11.8 或 12.1+（需与 PyTorch 匹配）

> ⚠️ **版本兼容性至关重要**：PyTorch / CUDA / TensorRT / Ultralytics 之间有严格的版本对应关系，版本不匹配是绝大多数问题的根因。请务必阅读下方[版本兼容与环境问题](#-版本兼容与环境问题)章节。

### 安装步骤

**1. 克隆仓库**

```bash
git clone https://github.com/only-one-over/yolo26_app.git
cd yolo26_app
```

**2. 创建虚拟环境（推荐）**

> 💡 **推荐**：使用虚拟环境隔离项目依赖，避免污染系统 Python 环境。

**方式一：Python venv（标准 Python）**

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate
```

**方式二：Anaconda / Miniconda**

```bash
# 创建虚拟环境（推荐 Python 3.10）
conda create -n yolo26 python=3.10 -y

# 激活虚拟环境
conda activate yolo26
```

**3. 安装依赖**

**方式一：基础安装（推荐新手）**

```bash
pip install -r requirements.txt
```

**方式二：锁定版安装（推荐生产环境，版本已验证兼容）**

```bash
# 先安装 PyTorch（需根据 CUDA 版本选择，见下方）
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
# 再安装其他锁定依赖
pip install -r requirements-lock.txt
```

锁定版本组合：Python 3.10 + PyTorch 2.3.1 + CUDA 12.1 + Ultralytics 8.3.20 + TensorRT 10.2.0 + PyQt6 6.7.1 + OpenCV 4.10.0.84 + ONNX Runtime 1.18.1。

> ⚠️ **Ultralytics 版本注意**：`requirements.txt` 要求 `ultralytics>=8.0`，但 TensorRT 10.x 用户需确保 Ultralytics ≥ 8.3.0（旧版本的 `BuilderFlag.FP16` 在 TensorRT 10 中已改为 `kFP16`，会导致导出报错）。如遇 `BuilderFlag has no attribute 'FP16'` 错误，执行 `pip install -U ultralytics`。

**4. 安装 PyTorch（GPU 支持）**

> ⚠️ **重要**：不要使用 `pip install torch` 默认安装，那会安装 CPU-only 版本，即使有 GPU 也无法使用！

先检查你的 CUDA 版本：
```bash
nvidia-smi
```
右上角显示 CUDA 版本（如 12.1、11.8），然后安装对应版本：

```bash
# CUDA 12.1（推荐，兼容性最好）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

> ⚠️ **版本对应关系**：PyTorch 2.1+ 对应 CUDA 11.8/12.1；PyTorch 2.3+ 对应 CUDA 12.4。安装后务必运行下方"验证 GPU"步骤确认 `torch.cuda.is_available()` 返回 `True`。

**5. 安装可选依赖**

```bash
# Intel RealSense 深度相机支持
pip install -e ".[realsense]"

# SAM 2 分割支持
pip install -e ".[sam]"

# Grounding DINO 支持
pip install -e ".[dino]"

# TensorRT 极速 GPU 推理（需 Ultralytics ≥ 8.3.0）
pip install -e ".[tensorrt]"

# 全部可选依赖
pip install -e ".[all]"
```

**6. 启动应用**

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
2. 输入项目名称（默认自动编号 project1、project2...），路径自动设为 `my_project/`
3. 项目目录自动创建：
   ```
   my_project/你的项目名/
   ├── images/                  # 素材目录（导入图片/视频帧自动复制到此）
   ├── datasets/                # 数据集目录（导出数据集自动存到此）
   ├── models/                  # 导出模型目录
   ├── runs/                    # 训练运行记录
   ├── project_config.json      # 项目配置
   ├── annotations.json         # 标注数据（自动保存，使用相对路径）
   └── classes.txt              # 类别列表
   ```

> 💡 项目统一存放在 `my_project/` 下，系统模型统一存放在 `system_model/` 下。整个项目文件夹可整体移动，标注引用不会失效。

#### Step 2: 导入图片

在标注页面，支持三种导入方式：
- **导入图片** — 选择单张或多张图片
- **导入视频** — 选择视频文件，自动提取帧
- **导入目录** — 选择整个目录，批量导入所有图片

#### Step 3: 添加类别

在左侧类别面板点击 "+" 添加标注类别，每个类别自动分配不同颜色。添加类别时可设置关键点数量（0表示无关键点），类别列表中显示如 "person (17pt)"。

#### Step 4: 绘制标注

| 工具 | 操作方式 | 快捷键 |
|------|---------|--------|
| 矩形框 | 按住鼠标拖拽 | — |
| 多边形 | 逐点点击，双击完成 | — |
| 关键点 | 点击放置关键点，双击/Enter 完成 | — |
| OBB 旋转框 | 先拖拽确定外接矩形，再拖拽旋转确定角度 | — |
| 选择 | 点击标注选中 | — |
| 删除 | 选中后按 Delete 或 空格 | Delete / 空格 |
| 撤销 | 撤销上一步操作 | Ctrl+Z |
| 重做 | 重做被撤销的操作 | Ctrl+Shift+Z |
| 上一张 | 切换到上一张图片 | ↑ |
| 下一张 | 切换到下一张图片 | ↓ 或 Shift+空格 |
| 切换到矩形工具 | 切换至矩形工具模式 | R |
| 切换到多边形工具 | 切换至多边形工具模式 | P |
| 切换到 OBB 旋转框工具 | 切换至 OBB 工具模式 | O |
| 切换到关键点工具 | 切换至关键点工具模式 | K |
| 切换到选择工具 | 切换至选择工具模式 | S |

#### 多边形顶点编辑

选中多边形标注后，可对顶点进行精细编辑：

- **拖拽顶点**：按住顶点圆点拖拽，实时更新多边形形状
- **右键删除顶点**：在顶点圆点上右键，删除该顶点（至少保留 3 个顶点）
- **双击边添加顶点**：在多边形的边上双击，在该位置插入新顶点
- 所有编辑操作支持 Ctrl+Z 撤销

#### Step 5: 辅助标注（可选）

加速标注的两种方式：

**SAM 2 交互式分割：**
1. 切换到 SAM 工具模式
2. 点击目标区域，自动生成分割掩码
3. 支持 SAM 2 模型：sam2.1_hiera_tiny / sam2.1_hiera_small / sam2.1_hiera_base_plus / sam2.1_hiera_large

**Grounding DINO 文本检测：**
1. 点击"文本检测"按钮
2. 输入文本描述（如 "person, car, dog"）
3. 自动检测并生成标注

#### Step 6: 导出数据集

1. 点击"导出数据集"按钮
2. 数据集自动导出到项目内 `datasets/` 目录（已打开项目时无需选择目录）
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
├── main.py                          # 应用入口（python main.py 启动）
├── code/                            # ✅ 代码核心（用户更新只需替换此文件夹）
│   └── yolo26_app/                  # 应用主包
│       ├── core/                    # 核心业务逻辑
│       │   ├── config.py            # 数据模型 (ClassItem, TrainConfig, ProjectConfig)
│       │   ├── paths.py             # 工作区路径常量 (system_model/my_project)
│       │   ├── project_manager.py   # 项目管理（创建/打开/最近项目/路径）
│       │   ├── label_manager.py     # 标注类别管理（增删改查/颜色分配）
│       │   ├── annotation_canvas.py # 标注画布 (Scene + View + 撤销/重做 + 增量绘制)
│       │   ├── yolo_exporter.py     # YOLO 数据集导出（格式转换/校验/划分）
│       │   ├── trainer.py           # YOLO 训练器 (QThread + 回调进度)
│       │   ├── predictor.py         # YOLO 推理器（加载/推理/验证/导出/TensorRT兼容）
│       │   ├── auto_annotator.py    # 辅助标注 (SAM/DINO)
│       │   ├── gpu_detector.py      # GPU 检测 (异步/超时保护/缓存/安全模式)
│       │   ├── task_manager.py      # 后台任务管理器 (异步/超时/回调)
│       │   ├── model_registry.py    # 模型/增强预设常量统一管理
│       │   ├── persistence.py       # 原子写入工具
│       │   ├── workspace_manager.py # 工作区间管理
│       │   ├── realsense_camera.py  # RealSense 深度相机（设备枚举/帧获取/深度着色）
│       │   ├── config_template.yaml # 默认配置模板
│       │   └── utils/               # 通用工具（中文路径图像读写等）
│       └── ui/                      # 用户界面
│           ├── main_window.py       # 主窗口 (异步GPU检测/安全模式/延迟加载/页面导航)
│           ├── annotation.py        # 标注模块 (持久化/批量检测/类别映射/辅助标注)
│           ├── training.py          # 训练模块 (参数配置/进度显示/日志)
│           ├── inference.py         # 测试模块 (异步推理/验证/导出/帧跳过)
│           ├── export_dialog.py     # 导出对话框 (格式/参数/预设)
│           ├── styles.py            # QSS 样式表 (Catppuccin Mocha/Latte)
│           └── icons/               # SVG 图标资源
├── system_model/                    # 系统模型目录（自动创建，gitignore）
│   ├── yolo/                        # 训练预训练模型 (yolo26n.pt 等)
│   ├── sam2/                        # SAM2 分割模型权重
│   ├── grounding_dino/              # GroundingDINO 模型权重
│   └── user_trained/                # 用户训练模型
├── my_project/                      # 用户项目目录（自动创建，gitignore）
│   ├── default/                     # 默认工作区间（自由空间模式使用）
│   └── project1/                    # 用户工作区间
├── requirements.txt                 # 依赖清单
├── pyproject.toml                   # 项目配置
├── LICENSE                          # MIT 许可证
└── README.md                        # 本文件
```

> 💡 **代码核心在 `code/` 文件夹下**，用户更新项目时只需替换 `code/` 文件夹即可获得最新功能，不影响模型（`system_model/`）、用户数据（`my_project/`）和配置。

### 核心数据模型

```python
# 标注类别
@dataclass
class ClassItem:
    name: str = ""           # 类别名称
    color: str = "#FF0000"   # 十六进制颜色值
    kpt_count: int = 0       # 关键点数量（0 表示无关键点）

# 训练配置
@dataclass
class TrainConfig:
    task: str = "detect"           # 任务类型
    model_size: str = "n"          # 模型大小
    model_family: str = "yolo26"   # 模型族 (yolo26 / yolov8)
    pretrained_model: str = ""     # 自定义模型路径
    data: str = ""                 # 数据集配置文件路径
    epochs: int = 100              # 训练轮数
    batch: int = 16                # 批大小
    imgsz: int = 640               # 输入图像尺寸
    device: str = ""               # 设备 (auto/cpu/0/0,1)
    optimizer: str = "auto"        # 优化器
    lr0: float = 0.01              # 初始学习率
    patience: int = 100            # 早停耐心值
    augmentation_enabled: bool = True   # 启用数据增强
    augmentation_preset: str = "default" # 增强预设
    mosaic: float = 1.0            # Mosaic 增强
    mixup: float = 0.0             # MixUp 增强
    # ... 更多增强参数

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
    item_type: str = "rect"          # "rect"、"polygon" 或 "keypoint"
    keypoints: List[QPointF] = field(default_factory=list)  # 关键点列表
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

## 🔧 版本兼容与环境问题

本章节详细说明各依赖之间的版本兼容关系，以及常见环境问题的排查方法。

### 版本对应关系总表

| 组件 | 推荐版本 | 兼容版本 | 备注 |
|------|---------|---------|------|
| Python | 3.10 / 3.11 | 3.9 - 3.11 | 3.12 部分依赖（如 SAM2）可能不兼容 |
| PyTorch | 2.1+ | 2.0 - 2.4 | 需与 CUDA 版本严格匹配 |
| CUDA Toolkit | 12.1 | 11.8 / 12.1 / 12.4 | `nvidia-smi` 显示的版本是驱动支持的最高版本 |
| Ultralytics | ≥ 8.3.0 | ≥ 8.0 | TensorRT 10 用户必须 ≥ 8.3.0 |
| TensorRT | 8.6 / 10.x | 8.5 - 10.x | 10.x 枚举名变更，需 Ultralytics ≥ 8.3.0 |
| PyQt6 | ≥ 6.0 | 6.0 - 6.7 | — |
| OpenCV | ≥ 4.6 | 4.6 - 4.10 | — |
| ONNX Runtime | 1.16+ | 1.15 - 1.19 | GPU 版需与 CUDA 匹配 |

### TensorRT 版本问题（重点）

#### 问题：`BuilderFlag has no attribute 'FP16'`

**根因**：TensorRT 10.x 将绑定拆分为 `tensorrt` + `tensorrt_bindings`，且 `BuilderFlag` 枚举从 `FP16` 改名为 `kFP16`。旧版 Ultralytics（< 8.3.0）仍使用 `BuilderFlag.FP16`，触发 `AttributeError`。

**解决方案**（任选其一）：
1. **升级 Ultralytics**（推荐）：`pip install -U ultralytics`
2. **降级 TensorRT 到 8.x**：`pip install tensorrt==8.6.1`
3. **本应用已内置兼容处理**：[predictor.py](yolo26_app/core/predictor.py) 在导出 engine 前自动补齐 `BuilderFlag.FP16` → `kFP16` 别名，但若 Ultralytics 内部其他位置也调用了旧枚举，仍需升级。

#### TensorRT 安装注意事项

```bash
# TensorRT 10.x（需 Ultralytics ≥ 8.3.0）
pip install tensorrt

# TensorRT 8.x（兼容旧版 Ultralytics）
pip install tensorrt==8.6.1
```

> TensorRT 版本必须与 CUDA 版本匹配。TensorRT 10.x 需要 CUDA 12.x；TensorRT 8.x 支持 CUDA 11.8/12.x。

### PyTorch 与 CUDA 版本匹配

#### 常见错误

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `CUDA out of memory` | 显存不足 | 减小 batch_size 或选择更小模型 |
| `torch.cuda.is_available()` 返回 `False` | PyTorch 是 CPU 版本，或 CUDA 版本不匹配 | 重装 GPU 版 PyTorch（见[安装步骤](#4-安装-pytorchgpu-支持)） |
| `RuntimeError: CUDA error: no kernel image` | PyTorch 编译的 CUDA 版本与驱动不匹配 | 安装与 `nvidia-smi` 显示版本匹配的 PyTorch |
| `undefined symbol: ...` | ONNX Runtime GPU 与 CUDA 版本冲突 | 用 CPU 版：`pip uninstall onnxruntime-gpu && pip install onnxruntime` |

#### 验证安装

```bash
# 验证 PyTorch + CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# 验证 TensorRT
python -c "import tensorrt as trt; print('TensorRT:', trt.__version__)"

# 验证 Ultralytics
python -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"
```

### ONNX Runtime 版本冲突

| 场景 | 推荐安装 |
|------|---------|
| 仅 CPU 推理 | `pip install onnxruntime` |
| GPU 推理（CUDA 12.x） | `pip install onnxruntime-gpu` |
| GPU 推理（CUDA 11.8） | `pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/` |

> ⚠️ **不要同时安装 `onnxruntime` 和 `onnxruntime-gpu`**，会产生冲突。如需切换，先 `pip uninstall` 其中一个。

### SAM 2 安装问题

```bash
# SAM 2 需要单独安装
pip install sam2

# 如果安装失败，尝试从源码安装
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

SAM 2 模型权重下载后放到 `system_model/sam2/` 目录，应用会自动扫描。支持的模型：
- `sam2.1_hiera_t.pt`（Tiny，最快）
- `sam2.1_hiera_s.pt`（Small，推荐）
- `sam2.1_hiera_b+.pt`（Base Plus）
- `sam2.1_hiera_l.pt`（Large，最精准）

### 系统模型目录说明

应用首次启动时会自动创建 `system_model/` 目录：

```
system_model/
├── yolo/                    # 训练预训练模型（首次训练时自动下载）
│   ├── yolo26n.pt           # 检测
│   ├── yolo26n-seg.pt       # 分割
│   ├── yolo26n-cls.pt       # 分类
│   ├── yolo26n-pose.pt      # 姿态
│   └── ...
├── sam2/                    # SAM2 模型（需手动下载放入）
│   └── sam2.1_hiera_s.pt
└── grounding_dino/          # GroundingDINO 模型（需手动下载放入）
    └── groundingdino_swint_ogc.pth
```

> 训练预训练模型会在首次训练时由 Ultralytics 自动下载到 `system_model/yolo/`。SAM2 和 GroundingDINO 模型需手动下载后放入对应目录。

---

## 🔍 诊断与排障

### 日志位置

应用运行日志自动写入项目根目录的 `logs/` 文件夹：
- `logs/app_YYYYMMDD.log` — 每日轮转，保留 7 天
- `logs/crash_YYYYMMDD_HHMMSS.log` — 崩溃日志（程序异常退出时自动生成）

### 导出诊断报告

遇到问题需要反馈时，可一键导出诊断报告：

1. 菜单栏 → 帮助 → 导出诊断报告
2. 选择保存路径（默认 `my_project/<workspace>/diagnostic_report_YYYYMMDD_HHMMSS.zip`）
3. 报告包含：
   - 系统信息（OS、Python、PyQt6、OpenCV、PyTorch、CUDA、Ultralytics、TensorRT 版本）
   - GPU 状态
   - 最近 7 天日志

诊断报告可直接作为 Issue 附件上传，便于开发者快速定位问题。

---

## ❓ 常见问题

### GPU 相关

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 状态栏显示 🔴 CPU | PyTorch 未安装或安装了 CPU 版本 | 安装 CUDA 版 PyTorch（见[安装步骤](#4-安装-pytorchgpu-支持)） |
| 训练很慢 | 使用了 CPU 训练 | 确认 GPU 可用，检查 device 参数 |
| CUDA out of memory | batch 太大或模型太大 | 减小 batch_size 或选择更小的模型 |
| 状态栏显示 🔴 CPU (安全模式) | 上次应用未正常退出 | 正常关闭应用即可，安全模式使用缩短超时 (8s) 检测 GPU |
| 状态栏显示 🔴 CPU (检测超时) | CUDA 驱动初始化耗时过长 | 检查 CUDA 驱动是否正常，重启应用；已增加超时至 10s |
| 启动时状态栏显示"⏳ 检测设备..."时间长 | GPU 检测在后台进行，首次启动较慢 | 正常现象，检测结果会缓存 30 分钟 |

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

### ONNX 相关

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| ONNX 模型推理无检测框 | onnxruntime-gpu 与 CUDA 版本不匹配 | 安装 CPU 版: `pip uninstall onnxruntime-gpu && pip install onnxruntime` |
| ONNX 模型验证失败 | ONNX 格式不支持验证（val），仅 .pt 支持 | 使用 .pt 模型进行 mAP 验证 |
| 加载 ONNX 模型后程序卡死 | ONNX Runtime 首次推理初始化耗时阻塞主线程 | 新版本已修复：图片推理改为异步执行 |
| 导出 ONNX 后推理效果差 | 导出时缺少图优化 | 新版本已修复：自动添加 simplify=True |

### TensorRT 导出相关

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `BuilderFlag has no attribute 'FP16'` | TensorRT 10.x 枚举名改为 `kFP16`，Ultralytics 版本过低 | `pip install -U ultralytics`（≥ 8.3.0），或降级 TensorRT 到 8.x |
| TensorRT 导出报版本不兼容 | TensorRT 与 CUDA/Ultralytics 版本不匹配 | 参见[版本兼容与环境问题](#-版本兼容与环境问题)章节 |
| 导出 engine 后推理结果异常 | INT8 校准数据集与模型任务不匹配 | 确保校准 data.yaml 的标注类型与模型任务一致（检测/分割/关键点） |

---

## 📄 许可证

本项目基于 [MIT License](LICENSE) 开源。

> **注意**：本项目依赖 Ultralytics YOLO26，其采用 [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 许可证。如果你修改并分发 Ultralytics 源码，需遵守 AGPL-3.0 的要求。

---

<div align="center">

**如果这个项目对你有帮助，请给个 ⭐ Star 支持一下！**

</div>
