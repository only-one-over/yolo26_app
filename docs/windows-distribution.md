# Windows Portable Distribution

The release workflow produces two Windows x64 portable onedir archives:

- YOLO26-App-CPU.zip: use this on a computer without an NVIDIA GPU.
- YOLO26-App-CUDA.zip: use this on a computer with a compatible NVIDIA GPU and driver. It bundles CUDA-enabled PyTorch built against CUDA 12.1. It does not install a GPU driver.

Download one archive from the GitHub Release page, extract it completely, then start the matching YOLO26-App-<VARIANT>.exe. Do not move only the .exe; it depends on the adjacent files in its onedir folder.

The portable distribution contains the application, Python runtime, core dependencies, SVG/YAML resources, LICENSE, README.txt, and build-info.json. It deliberately excludes model weights and user data. On first start, user projects, annotations, models, and logs are created under %USERPROFILE%\.yolo26_app\workspace.

SAM2, Grounding DINO, RealSense, and TensorRT remain optional integrations. TensorRT must match the target machine's CUDA environment and is therefore not included in either archive.

## Creating a Release

The GitHub Actions workflow windows-release.yml builds both variants and publishes them as Release assets.

1. Create and push a version tag such as v1.0.1, or run **Windows onedir release** manually from the Actions page and enter a new tag.
2. Wait for both Build Windows cpu onedir and Build Windows cuda onedir jobs to complete.
3. The final job creates or updates the GitHub Release and uploads the two ZIP files and their .sha256 checksums.

For a local Windows build, install either CPU or CUDA PyTorch first, install the application and PyInstaller, then run:

~~~powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install .
python -m pip install "pyinstaller>=6.14,<7"
python tools/build_windows_onedir.py --variant cpu
~~~

Use the cu121 PyTorch index and --variant cuda for the CUDA archive.
