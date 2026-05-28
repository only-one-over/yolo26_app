# YOLO26 App

基于 [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) 的桌面端标注-训练-推理一体化应用。使用 PyQt6 构建 GUI，提供从数据标注、模型训练到推理测试的完整工作流。

## 功能特性

- **数据标注** — 支持矩形框和多边形两种标注工具，支持图片/视频/目录批量导入
- **数据集导出** — 将标注数据导出为 YOLO 格式（images/ + labels/ + data.yaml），自动划分训练/验证集
- **模型训练** — 基于 Ultralytics YOLO26 的后台线程训练，支持 detect/segment/classify/pose 四种任务
- **推理测试** — 支持图片/视频/摄像头/RealSense 多种输入源，实时显示推理结果
- **模型验证与导出** — 支持 mAP50/mAP50-95 指标验证，支持 ONNX/TorchScript/OpenVINO/TensorRT 格式导出
- **深色/亮色主题** — Catppuccin Mocha / Latte 双主题

## 界面预览

应用采用侧边栏导航布局，包含三个核心模块：

| 模块 | 功能 |
|------|------|
| 🏷️ 标注 | 导入图片、绘制标注、管理类别、导出数据集 |
| 🏋️ 训练 | 配置超参数、后台训练、实时日志、结果展示 |
| 🔍 测试 | 加载模型、多种推理源、验证指标、模型导出 |

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/<your-username>/yolo26-app.git
cd yolo26-app
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

如需 Intel RealSense 深度相机支持：

```bash
pip install pyrealsense2
```

### 4. 安装 PyTorch（GPU 支持）

如需 GPU 训练，请参考 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装对应 CUDA 版本的 PyTorch。

## 使用方法

### 启动应用

```bash
python -m yolo26_app.main
```

### 完整工作流

1. **新建项目** — 文件 → 新建项目，输入名称和路径
2. **导入图片** — 标注页面 → 导入图片/视频/目录
3. **添加类别** — 左侧类别面板点击 "+" 添加标注类别
4. **绘制标注** — 选择矩形或多边形工具，在画布上标注
5. **导出数据集** — 点击"导出数据集"，选择输出目录
6. **训练模型** — 切换到训练页面，选择 data.yaml，配置参数，开始训练
7. **测试推理** — 切换到测试页面，加载 best.pt 模型，选择推理源

### 支持的任务类型

| 任务 | 模型 | 说明 |
|------|------|------|
| detect | yolo26{n/s/m/l/x}.pt | 目标检测 |
| segment | yolo26{n/s/m/l/x}-seg.pt | 实例分割 |
| classify | yolo26{n/s/m/l/x}-cls.pt | 图像分类 |
| pose | yolo26{n/s/m/l/x}-pose.pt | 姿态估计 |

## 项目结构

```
yolo26-app/
├── yolo26_app/                  # 应用主包
│   ├── main.py                  # 应用入口
│   ├── core/                    # 核心业务逻辑
│   │   ├── config.py            # 数据模型 (ClassItem, TrainConfig, ProjectConfig)
│   │   ├── project_manager.py   # 项目管理
│   │   ├── label_manager.py     # 标注类别管理
│   │   ├── annotation_canvas.py # 标注画布 (Scene + View)
│   │   ├── yolo_exporter.py     # YOLO 数据集导出
│   │   ├── trainer.py           # YOLO 训练器 (QThread)
│   │   ├── predictor.py         # YOLO 推理器
│   │   └── realsense_camera.py  # RealSense 深度相机
│   └── ui/                      # 用户界面
│       ├── main_window.py       # 主窗口
│       ├── annotate_widget.py   # 标注模块
│       ├── train_widget.py      # 训练模块
│       ├── test_widget.py       # 测试模块
│       └── styles.py            # QSS 样式表
├── requirements.txt             # 依赖清单
├── pyproject.toml               # 项目配置
├── LICENSE                      # MIT 许可证
└── README.md                    # 本文件
```

## 技术栈

| 技术 | 用途 |
|------|------|
| Python 3.9+ | 编程语言 |
| PyQt6 | GUI 框架 |
| Ultralytics YOLO26 | 目标检测模型 |
| PyTorch | 深度学习框架 |
| OpenCV | 图像处理 |
| NumPy | 数值计算 |

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

> **注意**：本项目依赖 Ultralytics YOLO26，其采用 [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 许可证。如果你修改并分发 Ultralytics 源码，需遵守 AGPL-3.0 的要求。
