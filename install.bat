@echo off
chcp 65001 >nul 2>&1
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
)

REM 检测 CUDA
echo.
echo [检测] 正在检测 CUDA...
python -c "import subprocess; result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True); print(result.stdout if result.returncode == 0 else '')" >nul 2>&1
set HAS_CUDA=0
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo [检测] 发现 NVIDIA GPU 和 CUDA
    set HAS_CUDA=1
) else (
    echo [检测] 未检测到 NVIDIA GPU，将安装 CPU 版本
)

REM 安装 PyTorch
echo.
echo [安装] 正在安装 PyTorch...
if "%HAS_CUDA%"=="1" (
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
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

echo.
echo ========================================
echo   安装完成！
echo   运行方式: python main.py
echo ========================================
pause
