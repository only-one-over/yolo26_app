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
fi

# 检测 CUDA
echo
echo "[检测] 正在检测 CUDA..."
HAS_CUDA=0
if command -v nvidia-smi &> /dev/null; then
    echo "[检测] 发现 NVIDIA GPU 和 CUDA"
    HAS_CUDA=1
else
    echo "[检测] 未检测到 NVIDIA GPU，将安装 CPU 版本"
fi

# 安装 PyTorch
echo
echo "[安装] 正在安装 PyTorch..."
if [ "$HAS_CUDA" = "1" ]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
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

echo
echo "========================================"
echo "  安装完成！"
echo "  运行方式: python main.py"
echo "========================================"
