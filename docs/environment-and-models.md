# 环境与模型

## 基础环境

项目要求 Python 3.10 及以上。首次安装建议使用项目根目录的 `install.bat`，它会创建虚拟环境、安装 PyTorch 和应用依赖，并提供基本环境检查。

也可以手动安装：

```bash
python -m venv venv
venv\Scripts\activate
pip install -e .
```

Linux 或 macOS 请使用对应的虚拟环境激活命令。CPU 环境可以运行应用，但训练和实时推理速度会较慢。

## GPU 与 PyTorch

GPU 用户应先确认驱动与 PyTorch 组合可用：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

`nvidia-smi` 显示的 CUDA 版本代表驱动支持能力，不等同于本机一定安装了相同版本的 CUDA Toolkit。以项目安装脚本和 PyTorch 官方安装页给出的兼容组合为准。

## 可选组件

| 组件 | 安装方式 | 用途 |
| --- | --- | --- |
| SAM2 | `pip install -e ".[sam]"` | 交互式分割和 YOLO+SAM2 批量标注 |
| Grounding DINO | `pip install -e ".[dino]"` | 文本提示词零样本检测 |
| RealSense | `pip install -e ".[realsense]"` | Intel 深度相机 |
| TensorRT | `pip install -e ".[tensorrt]"` | NVIDIA 高性能推理与 engine 导出 |

可选组件按需安装。不要把 `onnxruntime` 与 `onnxruntime-gpu` 同时装入同一个环境。

## 模型位置

应用首次运行会创建模型目录：

```text
system_model/
├── yolo/              # YOLO 预训练模型
├── sam2/              # SAM2 权重
├── grounding_dino/    # Grounding DINO 权重
└── user_trained/      # 用户训练模型
```

YOLO 预训练权重通常会在第一次使用时自动下载。SAM2 与 Grounding DINO 权重可通过界面下载或手动放入对应目录。

模型下载会校验受信任来源和 SHA256；校验失败的临时文件会被移除，不会替换已有模型。

## TensorRT 安装与检查

安装完成后可验证 Python 绑定：

```bash
python -c "import tensorrt as trt; print(trt.__version__)"
```

若出现 `No module named 'tensorrt'`，请确认安装命令使用的是运行应用的同一个虚拟环境。若导出阶段报版本不兼容，请记录 TensorRT、CUDA、PyTorch、Ultralytics 与 GPU 型号后再查询[诊断与排障](troubleshooting.md)。
