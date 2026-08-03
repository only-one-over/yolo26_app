<div align="center">

# YOLO26 App

基于 Ultralytics YOLO 的桌面端标注、训练和推理应用，支持 YOLO26 与 YOLOv8。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/PyQt6-6.0+-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](README_EN.md) · [完整文档](docs/README.md) · [环境与模型](docs/environment-and-models.md) · [常见问题](docs/troubleshooting.md)

</div>

## 安装

### Windows 一键安装

准备 Python 3.10 及以上版本后，克隆项目并运行安装脚本：

~~~bat
git clone https://github.com/only-one-over/yolo26_app.git
cd yolo26_app
install.bat
~~~

脚本会创建虚拟环境、安装 PyTorch 与应用依赖，并检查 PyQt6、OpenCV、Ultralytics 和可选 TensorRT 环境。

### 手动安装

~~~bash
git clone https://github.com/only-one-over/yolo26_app.git
cd yolo26_app
python -m venv venv
~~~

Windows：

~~~bat
venv\Scripts\activate
pip install -e .
~~~

Linux 或 macOS：

~~~bash
source venv/bin/activate
pip install -e .
~~~

GPU、SAM2、Grounding DINO、RealSense 与 TensorRT 的安装方式见[环境与模型](docs/environment-and-models.md)。

### 构建与安装 wheel

发布或验证安装产物时，使用标准 wheel：

~~~bash
python -m pip install build
python -m build
python -m pip install --force-reinstall dist/yolo26_app-*.whl
~~~

构建命令会同时生成 wheel 与源码分发包。wheel 已包含界面 SVG 图标和 YAML 模板。

## 启动

~~~bash
python main.py
~~~

Windows 使用一键安装脚本后，也可以直接执行：

~~~bat
venv\Scripts\python.exe main.py
~~~

安装 wheel 后，Windows 可以直接运行图形启动命令：

~~~bat
yolo26-app
~~~

## PyPI 安装

已发布的 Python 包可在新虚拟环境中安装：

~~~bash
python -m pip install yolo26-app
yolo26-app
~~~

PyPI 包不包含模型权重、CUDA/TensorRT、SAM2、Grounding DINO 或 Windows 便携运行时；这些组件按需安装。普通 Windows 用户可继续从 GitHub Release 下载 CPU 或 CUDA 便携包。

## 基本使用

1. 通过“文件 -> 新建项目”创建工作区。
2. 在“标注”页导入图片、视频或素材目录，并添加类别。
3. 使用矩形框、多边形、关键点或 OBB 工具完成标注。
4. 点击“导出数据集”，生成 YOLO 格式的数据集。
5. 在“训练”页选择 data.yaml，设置模型与训练参数后开始训练。
6. 在“测试”页加载 best.pt，对图片、视频或相机执行推理，必要时导出 ONNX、TensorRT 等模型。

标注会自动保存；重新打开项目或异常退出后可恢复。更详细的标注、数据集和训练说明见[标注、数据集与训练](docs/annotation-and-training.md)。

## 文档

| 主题 | 说明 |
| --- | --- |
| [标注、数据集与训练](docs/annotation-and-training.md) | 素材导入、标注工具、辅助标注、数据集导出和训练 |
| [推理与模型导出](docs/inference-and-export.md) | 图片、视频、相机推理与部署模型导出 |
| [环境与模型](docs/environment-and-models.md) | GPU、可选依赖、模型权重和 TensorRT |
| [开发指南](docs/development.md) | 架构、测试和扩展方式 |
| [诊断与排障](docs/troubleshooting.md) | 日志、诊断报告和常见异常 |

## 许可证

本项目采用 [MIT License](LICENSE)。项目依赖 Ultralytics YOLO，请同时遵守其许可证要求。
