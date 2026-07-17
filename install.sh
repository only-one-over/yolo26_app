#!/bin/bash
set -e

echo "========================================"
echo "  YOLO26 App 一键安装脚本 (Linux/Mac)"
echo "========================================"
echo

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未找到 Python3，请先安装 Python 3.9+"
    exit 1
fi

# 检查是否在虚拟环境中
if [ -z "$VIRTUAL_ENV" ]; then
    echo "[提示] 当前不在虚拟环境中，正在创建..."
    python3 -m venv venv
    source venv/bin/activate
    echo "[完成] 虚拟环境已激活"
    # 验证虚拟环境激活成功(当前 python 不应是系统 Python)
    if ! python3 -c "import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)" 2>/dev/null; then
        echo "[警告] 虚拟环境激活失败,当前仍为系统 Python"
        echo "[排障] 请手动执行: source venv/bin/activate"
        echo "[排障] 然后重新运行本安装脚本"
        exit 1
    fi
    echo "[验证] 虚拟环境激活成功"
fi

# 检测 CUDA 版本
echo
echo "[检测] 正在检测 CUDA..."
HAS_CUDA=0
CUDA_VERSION=""
TORCH_INDEX="https://download.pytorch.org/whl/cpu"

if command -v nvidia-smi &> /dev/null; then
    echo "[检测] 发现 NVIDIA GPU"
    HAS_CUDA=1
    # 解析 nvidia-smi 输出获取 CUDA 版本(CUDA Version: XX.X)
    CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | awk '{print $9}')
    if [ -z "$CUDA_VERSION" ]; then
        echo "[检测] 无法解析 CUDA 版本号,默认使用 cu124 索引"
        CUDA_VERSION="12.4"
    fi
    echo "[检测] CUDA 版本: $CUDA_VERSION"
    # 根据 CUDA 主/次版本号选择 PyTorch --index-url
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
    # 默认 cu124(CUDA 12.2+ 或未知版本)
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    if [ "$CUDA_MAJOR" = "11" ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    fi
    if [ "$CUDA_MAJOR" = "12" ] && { [ "$CUDA_MINOR" = "0" ] || [ "$CUDA_MINOR" = "1" ]; }; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
    fi
    echo "[检测] PyTorch 索引: $TORCH_INDEX"
else
    echo "[检测] 未检测到 NVIDIA GPU，将安装 CPU 版本"
fi

# 安装 PyTorch
echo
echo "[安装] 正在安装 PyTorch..."
if [ "$HAS_CUDA" = "1" ]; then
    pip install torch torchvision --index-url "$TORCH_INDEX"
else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

# 安装项目依赖
echo
echo "[安装] 正在安装项目核心依赖..."
pip install -e .

# 询问是否安装可选依赖
echo
read -p "[可选] 是否安装 SAM 2 支持？(y/n) " INSTALL_SAM
if [ "$INSTALL_SAM" = "y" ]; then
    pip install -e ".[sam]"
fi

read -p "[可选] 是否安装 Grounding DINO 支持？(y/n) " INSTALL_DINO
if [ "$INSTALL_DINO" = "y" ]; then
    pip install -e ".[dino]"
fi

# ===== 环境自检 =====
echo
echo "======== 环境自检 ========"

echo
echo "[自检] PyTorch..."
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" || echo "[自检] PyTorch 导入失败"

echo
echo "[自检] OpenCV..."
python -c "import cv2; print('OpenCV:', cv2.__version__)" || echo "[自检] OpenCV 导入失败"

echo
echo "[自检] PyQt6..."
python -c "import PyQt6; print('PyQt6 已安装')" || echo "[自检] PyQt6 导入失败"

echo
echo "[自检] Ultralytics..."
python -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)" || echo "[自检] Ultralytics 导入失败"

echo
echo "[自检] TensorRT (可选)..."
python -c "import tensorrt; print('TensorRT:', tensorrt.__version__)" 2>/dev/null || echo "[自检] TensorRT 未安装(可选,需要 GPU 推理加速)"

echo
echo "[自检] ONNX Runtime (可选)..."
python -c "from importlib.metadata import version; print('ONNX Runtime:', version('onnxruntime'))" 2>/dev/null || echo "[自检] ONNX Runtime 未安装"

# ===== 排障提示 =====
echo
echo "======== 排障提示 ========"

# 检查 PyTorch CUDA 可用性 vs nvidia-smi 可用性
if [ "$HAS_CUDA" = "1" ]; then
    if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo "[排障] PyTorch 不可用 GPU,但检测到 NVIDIA 显卡"
        echo "[排障] 请重新安装 CUDA 版本:"
        echo "       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
    fi
fi

# 检查 onnxruntime 与 onnxruntime-gpu 冲突
if pip show onnxruntime > /dev/null 2>&1 && pip show onnxruntime-gpu > /dev/null 2>&1; then
    echo "[排障] 检测到 onnxruntime 与 onnxruntime-gpu 冲突,请卸载其一:"
    echo "       pip uninstall onnxruntime onnxruntime-gpu && pip install onnxruntime-gpu"
fi

echo
echo "========================================"
echo "  安装完成！"
echo "  运行方式: python main.py"
echo "========================================"
