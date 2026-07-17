@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion
echo ========================================
echo   YOLO26 App 一键安装脚本 (Windows)
echo ========================================
echo.

REM 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.9+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 检查是否在虚拟环境中
python -c "import sys; sys.exit(0 if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [提示] 当前不在虚拟环境中，正在创建...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo [完成] 虚拟环境已激活
    REM 验证虚拟环境激活成功(当前 python 不应是系统 Python)
    python -c "import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)" >nul 2>&1
    if errorlevel 1 (
        echo [警告] 虚拟环境激活失败,当前仍为系统 Python
        echo [排障] 请手动执行: venv\Scripts\activate.bat
        echo [排障] 然后重新运行本安装脚本
        pause
        exit /b 1
    )
    echo [验证] 虚拟环境激活成功
)

REM 检测 CUDA 版本
echo.
echo [检测] 正在检测 CUDA...
python -c "import subprocess; result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True); print(result.stdout if result.returncode == 0 else '')" >nul 2>&1
set HAS_CUDA=0
set CUDA_VERSION=
set TORCH_INDEX=https://download.pytorch.org/whl/cpu

nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo [检测] 发现 NVIDIA GPU
    set HAS_CUDA=1
    REM 解析 nvidia-smi 输出获取 CUDA 版本(CUDA Version: XX.X)
    set NVSMI_LINE=
    for /f "delims=" %%l in ('nvidia-smi 2^>nul ^| findstr /C:"CUDA Version"') do set NVSMI_LINE=%%l
    if defined NVSMI_LINE (
        REM 取 "CUDA Version:" 之后的部分,再提取第一个空格分隔的 token
        set CUDA_TAIL=!NVSMI_LINE:*CUDA Version:=!
        for /f "tokens=1" %%v in ("!CUDA_TAIL!") do set CUDA_VERSION=%%v
    )
    if "!CUDA_VERSION!"=="" (
        echo [检测] 无法解析 CUDA 版本号,默认使用 cu124 索引
        set CUDA_VERSION=12.4
    )
    echo [检测] CUDA 版本: !CUDA_VERSION!
    REM 根据 CUDA 主/次版本号选择 PyTorch --index-url
    for /f "tokens=1 delims=." %%m in ("!CUDA_VERSION!") do set CUDA_MAJOR=%%m
    for /f "tokens=2 delims=." %%n in ("!CUDA_VERSION!") do set CUDA_MINOR=%%n
    REM 默认 cu124(CUDA 12.2+ 或未知版本)
    set TORCH_INDEX=https://download.pytorch.org/whl/cu124
    if "!CUDA_MAJOR!"=="11" set TORCH_INDEX=https://download.pytorch.org/whl/cu118
    if "!CUDA_MAJOR!"=="12" if "!CUDA_MINOR!"=="0" set TORCH_INDEX=https://download.pytorch.org/whl/cu121
    if "!CUDA_MAJOR!"=="12" if "!CUDA_MINOR!"=="1" set TORCH_INDEX=https://download.pytorch.org/whl/cu121
    echo [检测] PyTorch 索引: !TORCH_INDEX!
) else (
    echo [检测] 未检测到 NVIDIA GPU，将安装 CPU 版本
)

REM 安装 PyTorch
echo.
echo [安装] 正在安装 PyTorch...
if "!HAS_CUDA!"=="1" (
    pip install torch torchvision --index-url !TORCH_INDEX!
) else (
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

REM 安装项目依赖
echo.
echo [安装] 正在安装项目核心依赖...
pip install -e .

REM 询问是否安装可选依赖
echo.
echo [可选] 是否安装 SAM 2 支持？(y/n)
set /p INSTALL_SAM=
if /i "%INSTALL_SAM%"=="y" (
    pip install -e ".[sam]"
)

echo [可选] 是否安装 Grounding DINO 支持？(y/n)
set /p INSTALL_DINO=
if /i "%INSTALL_DINO%"=="y" (
    pip install -e ".[dino]"
)

REM ===== 环境自检 =====
echo.
echo ======== 环境自检 ========

echo.
echo [自检] PyTorch...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA 可用:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
if errorlevel 1 (
    echo [自检] PyTorch 导入失败
)

echo.
echo [自检] OpenCV...
python -c "import cv2; print('OpenCV:', cv2.__version__)"
if errorlevel 1 (
    echo [自检] OpenCV 导入失败
)

echo.
echo [自检] PyQt6...
python -c "import PyQt6; print('PyQt6 已安装')"
if errorlevel 1 (
    echo [自检] PyQt6 导入失败
)

echo.
echo [自检] Ultralytics...
python -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"
if errorlevel 1 (
    echo [自检] Ultralytics 导入失败
)

echo.
echo [自检] TensorRT (可选)...
python -c "import tensorrt; print('TensorRT:', tensorrt.__version__)" 2>nul
if errorlevel 1 (
    echo [自检] TensorRT 未安装(可选,需要 GPU 推理加速)
)

echo.
echo [自检] ONNX Runtime (可选)...
python -c "from importlib.metadata import version; print('ONNX Runtime:', version('onnxruntime'))" 2>nul
if errorlevel 1 (
    echo [自检] ONNX Runtime 未安装
)

REM ===== 排障提示 =====
echo.
echo ======== 排障提示 ========

REM 检查 PyTorch CUDA 可用性 vs nvidia-smi 可用性
if "!HAS_CUDA!"=="1" (
    python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
    if errorlevel 1 (
        echo [排障] PyTorch 不可用 GPU,但检测到 NVIDIA 显卡
        echo [排障] 请重新安装 CUDA 版本:
        echo        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    )
)

REM 检查 onnxruntime 与 onnxruntime-gpu 冲突
pip show onnxruntime >nul 2>&1
if not errorlevel 1 (
    pip show onnxruntime-gpu >nul 2>&1
    if not errorlevel 1 (
        echo [排障] 检测到 onnxruntime 与 onnxruntime-gpu 冲突,请卸载其一:
        echo        pip uninstall onnxruntime onnxruntime-gpu ^&^& pip install onnxruntime-gpu
    )
)

echo.
echo ========================================
echo   安装完成！
echo   运行方式: python main.py
echo ========================================
pause
