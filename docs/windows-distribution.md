# Windows Portable Distribution

The release workflow produces three Windows x64 portable onedir archives:

- YOLO26-App-CPU.zip: use this on a computer without an NVIDIA GPU.
- YOLO26-App-CUDA.zip: use this on a computer with a compatible NVIDIA GPU and driver. It bundles CUDA-enabled PyTorch built against CUDA 12.1. It does not install a GPU driver.
- YOLO26-App-CUDA-FULL.zip: use this on a compatible NVIDIA GPU when offline annotation, training, and inference are required. It bundles CUDA-enabled PyTorch, SAM2 compatibility runtime, SAM2.1 Hiera Tiny, and YOLO26 n/s detection weights.

Download one archive from the GitHub Release page, extract it completely, then start the matching YOLO26-App-<VARIANT>.exe. Do not move only the .exe; it depends on the adjacent files in its onedir folder.

GitHub Release assets are limited to 2 GiB. If the CUDA archive exceeds that size, the release provides YOLO26-App-CUDA.zip.part001, additional sequential parts, YOLO26-App-CUDA.zip.sha256, and YOLO26-App-CUDA.reassemble.ps1. Download every part and the two helper files into one folder, then run:

~~~powershell
powershell -ExecutionPolicy Bypass -File .\YOLO26-App-CUDA.reassemble.ps1
Expand-Archive .\YOLO26-App-CUDA.zip
~~~

The script joins the parts and verifies the resulting ZIP checksum before extraction.

The CPU and CUDA distributions contain the application, Python runtime, core dependencies, SVG/YAML resources, LICENSE, README.txt, and build-info.json. CUDA-FULL additionally contains its documented pretrained model files. All variants keep user data outside the archive; on first start, user projects, annotations, models, and logs are created under %USERPROFILE%\.yolo26_app\workspace.

SAM2 remains optional in CPU and CUDA archives. CUDA-FULL includes SAM2 Tiny without its optional custom CUDA extension, so its core image segmentation workflow works without requiring a local CUDA Toolkit or NVCC. Grounding DINO, RealSense, and TensorRT remain optional integrations. TensorRT must match the target machine's CUDA environment and is therefore not included in any archive.

## Creating a Release

The GitHub Actions workflow windows-release.yml builds all three variants and publishes them as Release assets.

1. Create and push a version tag such as v1.0.1, or run **Windows onedir release** manually from the Actions page and enter a new tag.
2. Wait for Build Windows cpu onedir, Build Windows cuda onedir, and Build Windows cuda-full onedir jobs to complete.
3. The final job creates or updates the GitHub Release and uploads the ZIP files, split archive parts when needed, and their .sha256 checksums.

For a local Windows build, install either CPU or CUDA PyTorch first, install the application and PyInstaller, then run:

~~~powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
python -m pip install "pyinstaller>=6.14,<7"
python tools/build_windows_onedir.py --variant cpu
~~~

Use the cu121 PyTorch index and --variant cuda for the CUDA archive.
