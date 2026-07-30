# YOLO26 App — Code Wiki

> 基于 Ultralytics YOLO 的桌面端标注-训练-推理一体化应用（支持 YOLO26 / YOLOv8）

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 项目架构](#2-项目架构)
- [3. 目录结构](#3-目录结构)
- [4. 核心模块详解](#4-核心模块详解)
  - [4.1 入口层](#41-入口层)
  - [4.2 核心业务层 (core/)](#42-核心业务层-core)
  - [4.3 UI 表现层 (ui/)](#43-ui-表现层-ui)
- [5. 关键类与函数说明](#5-关键类与函数说明)
- [6. 依赖关系](#6-依赖关系)
- [7. 项目运行方式](#7-项目运行方式)
- [8. 配置系统](#8-配置系统)
- [9. 数据流与持久化](#9-数据流与持久化)
- [10. 线程模型](#10-线程模型)
- [11. 样式系统](#11-样式系统)
- [12. 工程约定与最佳实践](#12-工程约定与最佳实践)

---

## 1. 项目概述

YOLO26 App 是一款基于 PyQt6 构建的桌面应用程序，将 YOLO 目标检测工作流的三大核心环节——**数据标注、模型训练、推理测试**——整合到统一的图形界面中。项目支持 YOLO26 和 YOLOv8 两大模型系列，覆盖检测（detect）、分割（segment）、分类（classify）、关键点（pose）、旋转框（obb）五类任务。

### 核心能力

| 能力域 | 功能 |
|--------|------|
| 数据标注 | 矩形框、多边形、关键点、OBB 旋转框标注；SAM2 交互式分割；Grounding DINO 零样本检测；YOLO+SAM2 批量自动标注流水线；撤销/重做；自动持久化 |
| 模型训练 | 多任务/多模型系列训练；数据增强预设系统；实时进度与日志；训练曲线可视化；可中断训练 |
| 推理测试 | 图片/视频/摄像头/RealSense 多源推理；模型验证（mAP 等）；多格式模型导出（ONNX/TensorRT/OpenVINO 等）；ONNX GPU/CPU 自动回退 |
| 工程化 | 原子写入防损坏；全局异常处理与崩溃恢复；GPU 子进程检测与缓存；统一日志体系；暗/亮双主题 |

---

## 2. 项目架构

### 2.1 三层分层架构

```
┌─────────────────────────────────────────────────────┐
│                    UI 表现层 (ui/)                    │
│  MainWindow · AnnotateWidget · TrainWidget · TestWidget  │
│  ExportDialog · styles.py                            │
├─────────────────────────────────────────────────────┤
│                  核心业务层 (core/)                    │
│  配置管理 │ 标注画布 │ 训练器 │ 推理器 │ 自动标注器     │
│  数据集导出 │ 项目管理 │ 工作区管理 │ GPU检测 │ 日志     │
│  异常处理 │ RealSense │ 任务管理 │ 持久化 │ 标签管理    │
├─────────────────────────────────────────────────────┤
│                    基础设施层                         │
│  Ultralytics YOLO │ PyQt6 │ OpenCV │ PyTorch        │
│  SAM2 (可选) │ Grounding DINO (可选) │ pyrealsense2  │
└─────────────────────────────────────────────────────┘
```

### 2.2 架构设计原则

- **关注点分离**：UI 层只负责交互与展示，核心业务逻辑全部封装在 `core/` 中，两者通过信号/槽和直接方法调用通信。
- **代码核心隔离**：所有功能代码集中在 `code/` 文件夹下，用户更新项目时只需替换此目录。
- **原子写入**：关键配置文件（`project_config.json`、`annotations.json`）使用 `QSaveFile` 或临时文件 + `os.replace` 模式写入，杜绝中途崩溃导致的文件损坏。
- **异步非阻塞**：所有耗时操作（训练、推理、模型加载、GPU 检测）在 QThread 中执行，UI 保持响应。
- **优雅降级**：可选依赖（SAM2、Grounding DINO、RealSense、TensorRT）采用 try-import 模式，缺失时功能不可用但不影响应用启动。

### 2.3 模块依赖关系图

```
main.py
  └── ui/main_window.py (MainWindow)
        ├── ui/annotation.py (AnnotateWidget)
        │     ├── core/annotation_canvas.py (AnnotationScene, AnnotationView, AnnotationItem)
        │     ├── core/label_manager.py (LabelManager)
        │     ├── core/auto_annotator.py (YOLOPreAnnotator, SAMAnnotator, GroundingDINOAnnotator)
        │     ├── core/yolo_exporter.py (YOLOExporter)
        │     ├── core/persistence.py (write_json_atomic)
        │     └── core/utils/common.py (imread_unicode, imwrite_unicode)
        ├── ui/training.py (TrainWidget)
        │     ├── core/trainer.py (YOLOTrainer)
        │     ├── core/model_registry.py (MODEL_FAMILY_TASK_MODEL_MAP, AUGMENTATION_PRESETS)
        │     └── core/config.py (TrainConfig)
        ├── ui/inference.py (TestWidget)
        │     ├── core/predictor.py (YOLOPredictor)
        │     ├── core/realsense_camera.py (RealSenseCamera)
        │     └── ui/export_dialog.py (ExportDialog)
        ├── core/project_manager.py (ProjectManager)
        ├── core/workspace_manager.py (WorkspaceManager)
        ├── core/gpu_detector.py (GPUDetectWorker)
        ├── core/paths.py (路径常量)
        └── core/config.py (ProjectConfig)
  └── core/logger.py (init_logging, get_logger)
  └── core/exception_handler.py (install_exception_hooks)
```

---

## 3. 目录结构

```
ultralytics-main/
├── main.py                      # 应用入口
├── pyproject.toml               # 项目元数据与依赖声明
├── requirements.txt             # 核心依赖（pip install -r）
├── requirements-lock.txt        # 锁定版依赖（生产环境推荐）
├── install.bat                  # Windows 一键安装脚本
├── install.sh                   # Linux/macOS 一键安装脚本
├── README.md                    # 中文文档
├── README_EN.md                 # 英文文档
├── LICENSE                      # MIT 许可证
├── .gitignore                   # Git 忽略规则
│
├── code/                        # ★ 代码核心目录（用户更新只需替换此目录）
│   └── yolo26_app/
│       ├── __init__.py
│       ├── core/                # 核心业务逻辑
│       │   ├── __init__.py
│       │   ├── config.py            # 数据类: ClassItem, TrainConfig, ProjectConfig
│       │   ├── paths.py             # 工作区路径常量与目录初始化
│       │   ├── project_manager.py   # 项目创建/打开/最近项目
│       │   ├── workspace_manager.py # 工作区间扫描/创建/校验
│       │   ├── annotation_canvas.py # 标注场景与视图 (QGraphicsScene/View)
│       │   ├── label_manager.py     # 类别标签管理
│       │   ├── trainer.py           # YOLO 训练 QThread
│       │   ├── predictor.py         # YOLO 推理/验证/导出
│       │   ├── auto_annotator.py    # AI 辅助标注 (YOLO/SAM2/GroundingDINO)
│       │   ├── yolo_exporter.py     # 数据集导出 (YOLO 格式)
│       │   ├── model_registry.py    # 模型家族模板与增强预设常量
│       │   ├── persistence.py       # 原子 JSON 写入 (QSaveFile)
│       │   ├── gpu_detector.py      # GPU 子进程检测与缓存
│       │   ├── logger.py            # 统一日志体系
│       │   ├── exception_handler.py # 全局异常处理与崩溃恢复
│       │   ├── task_manager.py      # 通用异步任务管理器
│       │   ├── realsense_camera.py  # Intel RealSense 深度相机封装
│       │   ├── config_template.yaml # 默认配置模板（参考文档）
│       │   └── utils/
│       │       ├── __init__.py
│       │       └── common.py        # 通用工具: 中文路径图像读写
│       │
│       └── ui/                  # UI 表现层
│           ├── __init__.py
│           ├── main_window.py       # 主窗口 (导航/工作区/页面切换)
│           ├── annotation.py        # 标注页面
│           ├── training.py          # 训练页面
│           ├── inference.py         # 推理/测试页面
│           ├── export_dialog.py     # 模型导出对话框
│           ├── styles.py            # 设计令牌与 QSS 样式表
│           └── icons/               # SVG 图标资源
│
├── system_model/               # 系统模型目录（预训练/SAM2/GroundingDINO）
│   ├── yolo/                    #   YOLO 预训练权重
│   ├── sam2/                    #   SAM2 模型
│   ├── grounding_dino/          #   GroundingDINO 模型
│   └── user_trained/            #   用户训练产出的模型
│
├── my_project/                 # 用户工作区根目录
│   └── default/                 #   默认工作区间（自由空间模式回退）
│       └── (project_config.json, images/, datasets/, models/, runs/)
│
├── datasets/                   # 数据集输出目录
├── runs/                       # 训练输出目录
└── logs/                       # 运行日志（按日期切割，保留 7 天）
```

---

## 4. 核心模块详解

### 4.1 入口层

#### `main.py`

应用入口点，职责简洁明确：

1. 创建 `QApplication`，设置 Fusion 风格
2. 将 `code/` 目录插入 `sys.path`，使 `yolo26_app` 包可被导入
3. 初始化统一日志体系（`init_logging`）
4. 创建 `MainWindow` 实例
5. 安装全局异常钩子（`install_exception_hooks`）
6. 进入 Qt 事件循环

```python
def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO26 App")
    app.setStyle("Fusion")
    init_logging(Path(__file__).parent)
    window = MainWindow()
    install_exception_hooks(window)
    window.show()
    return app.exec()
```

### 4.2 核心业务层 (core/)

#### 4.2.1 `config.py` — 配置数据类

定义三个核心数据类，使用 `@dataclass` 实现序列化/反序列化：

| 类 | 职责 | 关键字段 |
|----|------|----------|
| `ClassItem` | 标注类别 | `name`, `color`, `kpt_count` |
| `TrainConfig` | 训练配置 | `task`, `model_family`, `model_size`, `epochs`, `batch`, `imgsz`, `device`, `optimizer`, `lr0`, `patience`, 16 个数据增强参数 |
| `ProjectConfig` | 项目配置 | `project_name`, `project_path`, `classes: List[ClassItem]`, `train_config: TrainConfig`, `created_at`, `last_opened` |

**原子写入模式**：`ProjectConfig.save()` 使用 `tempfile.mkstemp` + `os.replace` 实现原子写入，异常时自动清理临时文件。

**增强预设归一化**：`normalize_augmentation_preset()` 支持中英文别名映射（如 `"关闭"→"off"`, `"默认"→"default"`）。

#### 4.2.2 `paths.py` — 路径常量

集中管理所有工作区路径，避免散落各模块：

| 常量 | 路径 | 用途 |
|------|------|------|
| `WORKSPACE_ROOT` | 项目根目录 | 所有路径的基准 |
| `SYSTEM_MODEL_DIR` | `system_model/` | 系统模型根目录 |
| `SYSTEM_MODEL_SUBDIRS` | 子目录字典 | `yolo/`, `sam2/`, `grounding_dino/` |
| `USER_TRAINED_MODELS_DIR` | `system_model/user_trained/` | 用户训练产出 |
| `PROJECTS_ROOT` | `my_project/` | 用户项目根目录 |
| `DEFAULT_PROJECT_DIR` | `my_project/default/` | 默认工作区间 |
| `APP_DATA_DIR` | `~/.yolo26_app/` | 应用状态目录 |

`ensure_workspace_dirs()` 在首次运行时创建完整目录结构。

#### 4.2.3 `project_manager.py` — 项目管理

`ProjectManager` 提供项目全生命周期管理的静态方法：

| 方法 | 功能 |
|------|------|
| `create_project(name, path)` | 创建项目目录结构（datasets/models/runs/images/ + classes.txt + project_config.json） |
| `open_project(path)` | 加载项目配置，更新 last_opened，记录到最近项目 |
| `get_recent_projects()` | 读取最近项目列表（最多 20 个） |
| `add_recent_project(path)` | 添加/更新最近项目 |
| `get_dataset_dir(config)` | 获取数据集目录 |
| `get_images_dir(config)` | 获取图片目录 |
| `get_models_dir(config)` | 获取模型目录 |
| `get_annotations_path(config)` | 获取标注文件路径 |

#### 4.2.4 `workspace_manager.py` — 工作区间管理

`WorkspaceManager` 管理工作区间的扫描、创建和校验：

- `list_workspaces()` — 扫描 `PROJECTS_ROOT` 下所有含 `project_config.json` 的子文件夹
- `is_valid_workspace(path)` — 校验路径是否为合法工作区间
- `create_workspace(name)` — 创建新工作区间（含名称合法性校验：非空、无非法字符 `\ / : * ? " < > |`、不重名）
- `get_workspace_path(name)` — 返回工作区间完整路径

#### 4.2.5 `annotation_canvas.py` — 标注画布

这是标注功能的核心模块，基于 PyQt6 的 Graphics View 框架实现：

**数据模型**：
- `AnnotationItem` (dataclass) — 标注项，包含 `class_index`、`rect`、`polygon`、`item_type`（rect/polygon/keypoint/obb）、`keypoints`、`angle`

**工具常量**：
- `TOOL_SELECT`, `TOOL_RECT`, `TOOL_POLYGON`, `TOOL_KEYPOINT`, `TOOL_OBB`, `TOOL_SAM`

**核心类**：

| 类 | 基类 | 职责 |
|----|------|------|
| `AnnotationScene` | `QGraphicsScene` | 标注场景：管理标注列表、绘制工具状态、鼠标交互、撤销/重做栈、SAM 交互、顶点编辑、OBB 旋转 |
| `AnnotationView` | `QGraphicsView` | 标注视图：缩放、平移、鼠标坐标映射 |
| `_SignalHolder` | `QObject` | 信号持有者，提供 `annotations_changed` 信号 |

`AnnotationScene` 支持的标注类型与交互流程：
- **矩形框**：拖拽绘制 → 释放完成
- **多边形**：逐点点击 → 双击/Enter 闭合 → 支持顶点拖拽编辑
- **关键点**：在矩形框内点击放置关键点 → 自动连线 → 双击/Enter 完成
- **OBB 旋转框**：第一次拖拽确定外接矩形 → 第二次拖拽围绕中心旋转确定角度
- **SAM 交互**：点击目标区域 → 正/负点提示 → 自动生成分割多边形

#### 4.2.6 `trainer.py` — 训练器

`YOLOTrainer(QThread)` 在后台线程执行 YOLO 训练：

| 信号 | 用途 |
|------|------|
| `progress_signal(int, int)` | 当前 epoch / 总 epoch |
| `log_signal(str)` | 训练日志文本 |
| `finished_signal(str)` | 训练完成消息（含最佳模型路径和指标） |
| `error_signal(str)` | 训练错误消息 |

**关键方法**：
- `run()` — 加载模型 → 注册回调 → 构建增强参数 → 调用 `model.train()` → 清理 GPU 缓存
- `stop()` — 设置 `_stop_flag`，通过 `on_train_epoch_end` / `on_train_batch_end` 回调中断训练
- `_build_augmentation_kwargs()` — 根据 `augmentation_enabled` 构建增强参数字典
- `_QtLogHandler` — 自定义 logging.Handler，将 Ultralytics 日志转发到 Qt 信号

**模型加载逻辑**：优先使用 `pretrained_model` 路径；否则根据 `model_family` + `task` + `model_size` 从 `MODEL_FAMILY_TASK_MODEL_MAP` 拼接权重文件名，在 `system_model/yolo/` 目录下查找。

#### 4.2.7 `predictor.py` — 推理器

`YOLOPredictor` 封装模型的加载、推理、验证和导出：

| 方法 | 功能 |
|------|------|
| `load_model(path, task)` | 加载模型，ONNX 模型自动验证并支持 CPU 回退 |
| `predict_image(image_path, conf, iou, imgsz, device, max_det)` | 图片推理，返回（标注图像, results） |
| `predict_frame(frame_np, conf, iou, imgsz, device, max_det)` | 视频帧推理 |
| `validate_model(data)` | 模型验证，根据任务类型返回 mAP/top1 等指标 |
| `export_model(format, output_dir, **kwargs)` | 多格式导出，含备份/恢复/验证机制 |
| `get_model_info()` | 获取模型任务类型和类别名 |
| `get_onnx_diag()` | ONNX 诊断信息 |

**TensorRT 兼容性**：`_apply_tensorrt_enum_compat()` 为 TensorRT 10.x 补齐旧式 `BuilderFlag` 枚举别名（`FP16→kFP16` 等），解决与旧版 Ultralytics 的兼容问题。

**ONNX GPU/CPU 回退**：`_verify_onnx_model()` 用 dummy 推理测试 ONNX 模型；失败时 `_reload_onnx_cpu()` 隐藏 GPU 设备并重新加载为 CPU 推理。

**导出安全机制**：导出前备份现有文件 → 导出 → 验证（仅 onnx/engine）→ 失败时恢复备份 → 成功后清理备份。

#### 4.2.8 `auto_annotator.py` — AI 辅助标注

提供三种 AI 辅助标注器：

| 类 | 功能 | 依赖 |
|----|------|------|
| `YOLOPreAnnotator` | 使用已加载 YOLO 模型自动预标注（rect/polygon） | ultralytics |
| `SAMAnnotator` | SAM2 交互式分割，支持 8 种模型配置自动匹配 | sam2 (可选) |
| `GroundingDINOAnnotator` | 文本驱动零样本检测 | groundingdino-pip (可选) |

`SAMAnnotator` 内置模型配置映射 `_SAM2_CONFIG_MAP` 和下载 URL 列表 `SAM2_MODEL_URLS`，`scan_model_file()` 自动扫描目录下的模型文件并匹配配置。

#### 4.2.9 `yolo_exporter.py` — 数据集导出

`YOLOExporter` 将标注数据导出为 YOLO 格式数据集：

**导出流程**：
1. `_validate_annotations()` — 预校验（类别数量、索引范围、关键点数量、标注有效性）
2. 创建临时目录（同级隐藏目录）
3. 随机划分训练/验证集（默认 80%）
4. 按任务类型写入标签文件
5. 生成 `data.yaml`（含 path/train/val/nc/names，pose 任务含 kpt_shape）
6. 原子重命名临时目录为目标目录
7. 异常时清理临时目录

**任务特定处理**：
- `detect`：矩形框归一化坐标；多边形转外接矩形
- `segment`：多边形顶点归一化坐标（Douglas-Peucker 简化）
- `pose`：矩形框 + 关键点坐标（x, y, visibility=2）
- `obb`：旋转框归一化坐标 + 角度（弧度转度）
- `classify`：目录结构导出（train/val 下按类别名分子目录），不生成 labels

#### 4.2.10 `model_registry.py` — 模型注册表

集中维护训练相关常量，避免多处重复定义：

| 常量 | 用途 |
|------|------|
| `MODEL_FAMILY_TASK_MODEL_MAP` | 各模型系列（yolo26/yolov8）在不同任务下的权重文件命名模板 |
| `AUGMENTATION_PRESET_LABELS` | 增强预设中文标签映射 |
| `AUGMENTATION_PRESET_ORDER` | 预设展示顺序 |
| `AUGMENTATION_PRESETS` | 4 种预设（off/light/default/strong）的具体参数值 |

#### 4.2.11 `persistence.py` — 原子持久化

`write_json_atomic(path, data)` 使用 `QSaveFile` 实现 JSON 原子写入：写入失败时 `cancelWriting()` 不产生部分文件，只有 `commit()` 成功后文件才可见。

#### 4.2.12 `gpu_detector.py` — GPU 检测

通过子进程检测 CUDA 可用性，避免主进程因 CUDA 初始化卡死：

- `detect_gpu_subprocess(timeout)` — 在独立进程中检测 `torch.cuda.is_available()`，含超时终止和重试
- `load_gpu_cache()` / `save_gpu_cache()` — 缓存检测结果（TTL 30 分钟，超时状态 TTL 1 分钟）
- `GPUDetectWorker(QThread)` — 异步检测，优先读缓存，信号返回结果
- `load_exit_flag()` / `save_exit_flag()` — 记录上次退出是否正常

#### 4.2.13 `logger.py` — 统一日志

- `init_logging(workspace_root)` — 初始化根 logger，文件 handler 按日期切割（保留 7 天），控制台 handler 输出 WARNING+ 
- `get_logger(name)` — 获取配置好的 logger 实例
- `get_workspace_root()` — 返回初始化时的工作区根路径

#### 4.2.14 `exception_handler.py` — 全局异常处理

`install_exception_hooks(main_window)` 安装双层异常钩子：

1. **Python 层**：`sys.excepthook` → `_crash_excepthook`
   - 强制保存当前标注（`flush_autosave`）
   - 写入崩溃日志（`crash_YYYYMMDD_HHMMSS.log`，含时间戳、系统信息、traceback）
   - 弹出友好提示对话框
   - 安全退出

2. **Qt 层**：重写 `QApplication.notify` 兜底捕获 Qt 事件循环中的异常

#### 4.2.15 其他核心模块

| 模块 | 类/函数 | 职责 |
|------|---------|------|
| `label_manager.py` | `LabelManager` | 类别标签 CRUD，20 色调色板自动分配 |
| `task_manager.py` | `TaskManager`, `_TaskWorker` | 通用异步任务管理，支持超时取消和回调 |
| `realsense_camera.py` | `RealSenseCamera`, `DeviceInfo` | Intel RealSense 深度相机封装（设备枚举、流启动、帧获取、深度着色） |
| `utils/common.py` | `imread_unicode`, `imwrite_unicode` | 中文路径图像读写（`np.fromfile` + `cv2.imdecode` / `cv2.imencode` + `np.tofile`） |

### 4.3 UI 表现层 (ui/)

#### 4.3.1 `main_window.py` — 主窗口

`MainWindow(QMainWindow)` 是应用的顶层窗口，职责包括：

- **布局结构**：顶部工作区间工具栏 + 左侧导航栏（64px 宽，图标按钮）+ 右侧 `QStackedWidget` 页面区
- **页面管理**：三个页面（标注/训练/测试），懒加载模式（首次切换时创建 Widget）
- **工作区间管理**：ComboBox 选择工作区间，含 "自由空间" 选项清空配置；切换失败时回滚到上一个选择
- **状态持久化**：窗口几何（`normalGeometry` + 屏幕边界检查）、训练页面滚动位置等保存到 `~/.yolo26_app/app_state.json`
- **环境自检**：延迟 2 秒检查常见环境问题（onnxruntime 冲突、PyTorch CUDA 不匹配等）
- **资源清理**：`closeEvent` 中停止所有后台线程（`_gpu_detect_worker` 等）

**辅助类**：`NewProjectDialog` — 新建项目对话框，自动生成不冲突的默认名（project1, project2...）

#### 4.3.2 `annotation.py` — 标注页面

`AnnotateWidget(QWidget)` 是数据标注的主界面：

- **媒体导入**：单张图片、视频文件（自动提取帧）、整个目录批量导入（>50 项显示进度对话框）
- **工具栏**：选择、矩形、多边形、关键点、OBB、SAM 分割、Grounding DINO、批量检测、清除、删除、导出
- **类别管理**：添加/删除/重命名类别，颜色自动分配
- **标注交互**：委托给 `AnnotationScene` 处理所有画布交互
- **自动保存**：1500ms 防抖，切换图片/关闭窗口时强制保存（`flush_autosave`）
- **批量检测**：YOLO + SAM2 自动标注流水线，后台线程逐帧处理，支持取消
- **数据集导出**：调用 `YOLOExporter` 导出，支持任务类型选择和训练比例配置

#### 4.3.3 `training.py` — 训练页面

`TrainWidget(QWidget)` 是模型训练的配置与监控界面：

- **配置区**（可滚动）：任务类型、模型系列、模型大小、预训练模型、epochs、batch、imgsz、device、optimizer、lr0、patience、workers、cache、seed、close_mosaic
- **数据增强区**：启用开关 + 预设选择（off/light/default/strong/custom）+ 16 个增强参数滑块
- **训练控制**：开始/停止训练按钮
- **进度显示**：epoch 进度条 + 实时日志文本框
- **训练曲线**：pyqtgraph 绘制 loss 曲线（box_loss, cls_loss, dfl_loss 等），自动刷新
- **状态恢复**：`save_state()` / `restore_state()` 保存/恢复滚动位置，打开页面时平滑滚动到上次位置

#### 4.3.4 `inference.py` — 推理页面

`TestWidget(QWidget)` 是模型推理与测试界面：

- **模型加载**：选择 .pt/.onnx/.engine 模型文件，显示模型信息和 ONNX 诊断
- **推理源**：单张图片、视频文件、摄像头、Intel RealSense 深度相机
- **推理参数**：置信度阈值、IoU 阈值、图像尺寸、设备
- **结果显示**：标注图像显示区域 + 检测结果统计
- **模型验证**：选择 data.yaml 验证模型，显示 mAP/top1 等指标
- **模型导出**：打开 `ExportDialog` 选择格式和参数

#### 4.3.5 `export_dialog.py` — 导出对话框

`ExportDialog(QDialog)` 提供模型导出的完整参数配置：

- **任务预设**：8 种预设（自定义、TensorRT FP16/INT8、检测/分割/关键点/OBB/分类），自动匹配当前模型任务
- **导出格式**：10 种格式（ONNX/TorchScript/OpenVINO/TensorRT/CoreML/TFLite/NCNN/PaddlePaddle/MNN/RKNN）
- **参数控件**：根据格式动态显示/隐藏适用参数（imgsz/half/int8/dynamic/batch/opset/workspace/simplify/nms/device/data/fraction）
- **INT8 校准**：必须提供 data.yaml 校准数据集，含文件存在性和格式校验
- **任务-预设不匹配警告**：选择与模型任务不符的预设时显示提示

**关键常量**：
- `FORMAT_PARAMS` — 各格式支持的参数集合
- `EXPORT_PRESETS` — 预设的具体参数值
- `TASK_PRESET_MAP` — 任务类型到预设名的映射

#### 4.3.6 `styles.py` — 样式系统

基于设计令牌（Design Token）的 QSS 样式表生成系统：

- `DARK_TOKENS` / `LIGHT_TOKENS` — 颜色、字体、间距、圆角等设计令牌字典
- `DARK_STYLE` / `LIGHT_STYLE` — 预生成的 QSS 样式表字符串
- `get_style(theme)` — 根据主题名返回对应 QSS

采用 Catppuccin 配色方案，暗色主题为 Mocha 变体，亮色主题为 Latte 变体。

---

## 5. 关键类与函数说明

### 5.1 核心数据类

#### `ClassItem`

```python
@dataclass
class ClassItem:
    name: str = ""           # 类别名称
    color: str = "#FF0000"   # 显示颜色（十六进制）
    kpt_count: int = 0       # 关键点数量（pose 任务）
```

#### `TrainConfig`

```python
@dataclass
class TrainConfig:
    task: str = "detect"              # 任务类型: detect/segment/classify/pose
    model_family: str = "yolo26"      # 模型系列: yolo26/yolov8
    model_size: str = "n"             # 模型大小: n/s/m/l/x
    pretrained_model: str = ""        # 自定义预训练模型路径
    epochs: int = 100                 # 训练轮数
    batch: int = 16                   # 批大小
    imgsz: int = 640                  # 图像尺寸
    device: str = ""                  # 设备: auto/cpu/0/0,1
    optimizer: str = "auto"           # 优化器: auto/SGD/Adam/AdamW
    lr0: float = 0.01                 # 初始学习率
    patience: int = 100               # 早停耐心值
    workers: int = 8                  # 数据加载线程数
    cache: bool = False               # 缓存数据
    seed: int = 0                     # 随机种子
    plots: bool = True                # 生成训练图表
    close_mosaic: int = 10            # 最后 N 轮关闭 mosaic
    augmentation_enabled: bool = True # 是否启用数据增强
    augmentation_preset: str = "default"  # 增强预设
    # ... 16 个数据增强参数 (hsv_h/s/v, degrees, translate, scale, shear,
    #     perspective, flipud, fliplr, mosaic, mixup, cutmix, copy_paste,
    #     erasing, auto_augment)
```

#### `ProjectConfig`

```python
@dataclass
class ProjectConfig:
    project_name: str = ""
    project_path: str = ""
    classes: List[ClassItem] = field(default_factory=list)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    created_at: str = ""
    last_opened: str = ""

    def save(self, path) -> None     # 原子写入（临时文件 + os.replace）
    @classmethod
    def load(cls, path) -> ProjectConfig  # 从 JSON 加载
```

### 5.2 标注相关类

#### `AnnotationItem`

```python
@dataclass
class AnnotationItem:
    class_index: int                              # 类别索引
    rect: QRectF = field(default_factory=QRectF)  # 矩形区域
    polygon: QPolygonF = field(default_factory=QPolygonF)  # 多边形顶点
    item_type: str = "rect"                       # 类型: rect/polygon/keypoint/obb
    keypoints: List[QPointF] = field(default_factory=list)  # 关键点列表
    angle: float = 0.0                            # OBB 旋转角度（弧度）
```

#### `AnnotationScene`

```python
class AnnotationScene(QGraphicsScene):
    # 信号
    annotations_changed = pyqtSignal()

    # 核心属性
    @property
    def current_tool(self) -> str          # 当前工具
    @property
    def current_class_index(self) -> int   # 当前类别索引
    @property
    def annotations(self) -> list[AnnotationItem]  # 标注列表副本

    # 配置方法
    def set_tool(self, tool: str) -> None
    def set_class_colors(self, colors: list[str]) -> None
    def set_class_names(self, names: list[str]) -> None
    def set_sam_annotator(self, annotator) -> None

    # 标注操作
    def set_annotations(self, annotations: list[AnnotationItem]) -> None
    def add_annotation(self, item: AnnotationItem) -> None
    def delete_selected(self) -> None
    def clear_annotations(self) -> None

    # 撤销/重做
    def undo(self) -> None
    def redo(self) -> None
```

### 5.3 训练器

#### `YOLOTrainer`

```python
class YOLOTrainer(QThread):
    progress_signal = pyqtSignal(int, int)  # (current_epoch, total_epochs)
    log_signal = pyqtSignal(str)            # 日志文本
    finished_signal = pyqtSignal(str)       # 完成消息
    error_signal = pyqtSignal(str)          # 错误消息

    def __init__(self, config: TrainConfig, project_path: str)
    def run(self) -> None     # 训练主逻辑
    def stop(self) -> None    # 设置停止标志
```

### 5.4 推理器

#### `YOLOPredictor`

```python
class YOLOPredictor:
    def load_model(self, path: str, task: str = "") -> bool
    def predict_image(self, image_path, conf, iou, imgsz, device, max_det) -> Tuple[np.ndarray, object]
    def predict_frame(self, frame_np, conf, iou, imgsz, device, max_det) -> Tuple[np.ndarray, object]
    def validate_model(self, data: str) -> dict
    def export_model(self, format: str, output_dir: str, **kwargs) -> Tuple[str, bool, str]
    def get_model_info(self) -> dict
    def get_onnx_diag(self) -> str

    @property
    def is_onnx(self) -> bool
```

### 5.5 工具函数

#### `write_json_atomic`

```python
def write_json_atomic(path: Union[str, Path], data: Any) -> None
    # 使用 QSaveFile 原子写入 JSON，读者永远不会看到部分文件
```

#### `imread_unicode` / `imwrite_unicode`

```python
def imread_unicode(path: str) -> np.ndarray
    # np.fromfile + cv2.imdecode，绕过 OpenCV 中文路径限制

def imwrite_unicode(path: str, img: np.ndarray) -> bool
    # cv2.imencode + np.tofile，绕过 OpenCV 中文路径限制
```

#### `detect_gpu_subprocess`

```python
def detect_gpu_subprocess(timeout: float = 10.0) -> Tuple[str, str]
    # 在独立进程中检测 CUDA，返回 (status, device_name)
    # status: "gpu" / "cpu" / "timeout" / "error"
```

---

## 6. 依赖关系

### 6.1 核心依赖（必装）

| 依赖 | 最低版本 | 用途 |
|------|----------|------|
| `ultralytics` | >=8.0 | YOLO 模型训练/推理/导出引擎 |
| `PyQt6` | >=6.0 | GUI 框架 |
| `opencv-python` | >=4.6.0 | 图像处理 |
| `numpy` | >=1.20.0 | 数值计算 |
| `pyyaml` | >=5.3.1 | YAML 配置文件读写 |
| `pyqtgraph` | >=0.13.0 | 训练曲线可视化 |

此外，`PyTorch`（torch + torchvision）是 Ultralytics 的底层依赖，由安装脚本根据 CUDA 版本自动选择对应索引安装。

### 6.2 可选依赖

| 可选组 | 包 | 用途 | 安装命令 |
|--------|----|------|----------|
| `sam` | `sam2` | SAM2 交互式分割标注 | `pip install -e ".[sam]"` |
| `dino` | `groundingdino-pip` | Grounding DINO 零样本检测 | `pip install -e ".[dino]"` |
| `realsense` | `pyrealsense2>=2.50.0` | Intel RealSense 深度相机 | `pip install -e ".[realsense]"` |
| `tensorrt` | `tensorrt>=8.6` | TensorRT 极速 GPU 推理 | `pip install -e ".[tensorrt]"` |
| `all` | 以上全部 | 全部可选依赖 | `pip install -e ".[all]"` |

### 6.3 依赖配置文件

| 文件 | 用途 |
|------|------|
| `pyproject.toml` | 项目元数据 + 核心依赖 + 可选依赖组 + 构建系统配置 |
| `requirements.txt` | pip 直接安装的核心依赖 + 可选依赖安装说明 |
| `requirements-lock.txt` | 锁定版依赖（PyTorch 2.3.1 + CUDA 12.1 + Ultralytics 8.3.20 + TensorRT 10.2.0） |

### 6.4 版本兼容性说明

- **TensorRT 10.x**：`BuilderFlag` 枚举值改名（`FP16`→`kFP16`），需 Ultralytics ≥ 8.3.0。`predictor.py` 内置兼容补丁。
- **ONNX Runtime**：`onnxruntime` 与 `onnxruntime-gpu` 不可同时安装。GPU 推理异常时自动回退 CPU。
- **Python**：支持 3.9 - 3.12。

---

## 7. 项目运行方式

### 7.1 环境准备

**前置要求**：
- Python 3.9+
- （可选）NVIDIA GPU + CUDA 驱动

### 7.2 一键安装

**Windows**：
```bat
install.bat
```
脚本自动完成：创建虚拟环境 → 检测 CUDA 版本 → 选择 PyTorch 索引 → 安装 PyTorch → 安装核心依赖 → 询问可选依赖 → 环境自检 → 排障提示

**Linux/macOS**：
```bash
chmod +x install.sh
./install.sh
```

### 7.3 手动安装

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate.bat       # Windows

# 2. 安装 PyTorch（根据 CUDA 版本选择索引）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124  # CUDA 12.4
# 或 CPU 版本:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. 安装项目核心依赖
pip install -e .
# 或: pip install -r requirements.txt

# 4. （可选）安装可选依赖
pip install -e ".[sam,dino,realsense,tensorrt]"
```

### 7.4 启动应用

```bash
python main.py
```

### 7.5 典型使用流程

```
1. 启动应用 → 自动创建工作区目录结构
2. 新建工作区间（或选择已有）
3. 标注页面：
   a. 导入图片/视频/目录
   b. 添加类别
   c. 使用标注工具标注（矩形/多边形/关键点/OBB）
   d.（可选）使用 SAM2/Grounding DINO 辅助标注
   e.（可选）使用 YOLO+SAM2 批量自动标注
   f. 导出 YOLO 格式数据集
4. 训练页面：
   a. 选择任务类型和模型
   b. 配置训练参数和数据增强
   c. 选择导出的数据集 data.yaml
   d. 开始训练，监控进度和曲线
5. 测试页面：
   a. 加载训练好的模型
   b. 图片/视频/摄像头推理
   c.（可选）模型验证
   d.（可选）导出模型（ONNX/TensorRT 等）
```

---

## 8. 配置系统

### 8.1 配置层次

```
项目级配置 (project_config.json)
├── 项目元数据 (project_name, project_path, created_at, last_opened)
├── 类别配置 (classes: List[ClassItem])
└── 训练配置 (train_config: TrainConfig)
    ├── 基本训练参数 (task, model_family, epochs, batch, ...)
    └── 数据增强参数 (16 个增强参数 + preset)

应用级状态 (~/.yolo26_app/)
├── app_state.json          # 窗口几何、当前工作区间、页面状态
├── gpu_cache.json          # GPU 检测结果缓存
└── recent_projects.json    # 最近项目列表
```

### 8.2 配置模板

`core/config_template.yaml` 提供了完整的参数参考，包含训练参数、推理参数和数据增强默认值及注释说明。

### 8.3 工作区间目录结构

每个工作区间（项目）包含以下子目录：

```
my_project/<workspace_name>/
├── project_config.json    # 项目配置
├── classes.txt            # 类别列表（纯文本）
├── annotations.json       # 标注数据
├── images/                # 导入的图片
├── datasets/              # 导出的数据集
├── models/                # 项目模型
└── runs/                  # 训练输出
```

---

## 9. 数据流与持久化

### 9.1 标注数据流

```
用户交互 (鼠标事件)
    ↓
AnnotationScene (内存中的 _annotations 列表)
    ↓ annotations_changed 信号
AnnotateWidget (自动保存防抖 1500ms)
    ↓ write_json_atomic()
annotations.json (项目目录，原子写入)
    ↓ 加载时
AnnotationScene.set_annotations() (恢复到画布)
```

### 9.2 训练数据流

```
TrainWidget (配置 UI)
    ↓ 收集参数
TrainConfig (dataclass)
    ↓
YOLOTrainer(QThread)
    ↓ model.train()
Ultralytics YOLO (训练引擎)
    ↓ 回调
progress_signal / log_signal (实时反馈到 UI)
    ↓ 训练完成
runs/<experiment_name>/weights/best.pt (模型产出)
    ↓ finished_signal
TrainWidget (显示完成消息和指标)
```

### 9.3 推理数据流

```
TestWidget (选择模型和输入源)
    ↓
YOLOPredictor.load_model() → YOLOPredictor.predict_image/frame()
    ↓
Ultralytics YOLO (推理引擎)
    ↓ results.plot()
标注图像 (numpy array)
    ↓
TestWidget (显示结果)
```

### 9.4 原子写入保障

关键文件写入使用两种原子模式：

1. **`ProjectConfig.save()`**：`tempfile.mkstemp` → 写入 → `os.replace`（POSIX 原子操作）
2. **`write_json_atomic()`**：`QSaveFile` → `commit()`（Qt 原子写入，读者永远看不到部分文件）

导出数据集使用临时目录模式：先写入同级隐藏临时目录 → 全部完成后 `os.replace` 重命名 → 异常时清理临时目录。

---

## 10. 线程模型

### 10.1 线程使用概览

| 线程 | 类 | 触发场景 | 停止方式 |
|------|----|----------|----------|
| 训练线程 | `YOLOTrainer(QThread)` | 用户点击"开始训练" | `stop()` 设置 `_stop_flag`，通过回调中断 |
| GPU 检测线程 | `GPUDetectWorker(QThread)` | 应用启动 | 自动完成（子进程超时机制） |
| 推理工作线程 | `TestWidget` 内部 QThread | 实时推理/视频推理 | 停止标志 + `wait(timeout)` |
| 自动标注线程 | `AnnotateWidget` 内部 QThread | 批量检测 | 停止标志 + `wait(timeout)` |
| 通用任务线程 | `_TaskWorker(QThread)` | `TaskManager.submit()` | 超时取消 + `quit()` + `wait(1000)` |
| SAM2 编码线程 | `AnnotateWidget` 内部 | SAM 交互标注 | 标志位控制 |

### 10.2 线程安全约定

- **自定义 `run()` 的 QThread 不响应 `quit()`**：必须使用显式 `_stop_flag`，在回调中检查并设置 `trainer.stop_training = True`。
- **线程引用清理**：`finished` 信号回调中将 worker 引用设为 `None`，同步 Python 和 C++ 对象清理。避免使用 lambda 连接 `finished` 信号（竞态条件）。
- **线程 parent=None**：防止 widget 析构时过早终止运行中的线程。
- **窗口关闭清理**：`MainWindow.closeEvent` 中显式停止所有运行中的 worker（`quit()` + `wait(3000)`）。
- **GPU 缓存**：`wait(5000)` 超时等待防止无限阻塞。

---

## 11. 样式系统

### 11.1 设计令牌

样式系统基于设计令牌（Design Token）方法，将颜色、字体、间距、圆角等视觉属性抽象为字典常量：

```python
DARK_TOKENS = {
    "color_base": "#1e1e2e",       # 基础背景色
    "color_mantle": "#181825",     # 次级背景色
    "color_surface_0": "#313244",  # 表面色 0
    "color_text": "#cdd6f4",       # 主文本色
    "color_primary": "#89b4fa",    # 主题强调色
    "font_sans": '"Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif',
    "font_size_base": "13px",
    "space_4": "16px",
    "radius_md": "6px",
    # ... 更多令牌
}
```

### 11.2 QSS 生成

`DARK_STYLE` 和 `LIGHT_STYLE` 是预生成的 QSS 字符串，通过 `get_style(theme)` 返回。样式覆盖了所有自定义控件（sidebar、navButton、topBar、configCard、primaryButton、warningLabel 等）。

### 11.3 配色方案

采用 **Catppuccin** 配色方案：
- 暗色主题：Mocha 变体（深色背景 + 柔和前景）
- 亮色主题：Latte 变体（浅色背景 + 深色前景）

---

## 12. 工程约定与最佳实践

### 12.1 文件 I/O 约定

- **原子写入**：所有关键配置文件（`project_config.json`、`annotations.json`）必须使用原子写入模式。
- **中文路径**：OpenCV 的 `imread`/`imwrite` 在 Windows 上不支持中文路径，必须使用 `imread_unicode`/`imwrite_unicode`（`np.fromfile` + `cv2.imdecode`）。
- **备份机制**：关键文件写入前创建备份，导出操作使用临时目录 + 原子重命名。

### 12.2 UI 约定

- **UI 阻塞操作**：GroundingDINO 检测等耗时操作必须在 QThread 子类中执行。
- **QGraphicsScene 清理**：Pixmap items 必须在添加新 item 前显式移除，防止内存泄漏。
- **工作区间切换**：必须包含错误处理，失败时回滚到上一个选择。
- **导入对话框**：图片/视频导入对话框起始目录为项目的 images 目录。
- **大型导入**：>50 项的导入必须显示 `QProgressDialog` 并支持取消。
- **窗口几何持久化**：使用 `normalGeometry()` 保存（兼容最大化/全屏），加载时包含屏幕边界检查。

### 12.3 线程约定

- **停止方法**：`stop()` 方法必须包含基于超时的 `wait()`（如 `wait(5000)`），防止无限阻塞。
- **回调清理**：`finished` 信号回调中必须将 worker 引用设为 `None`。
- **Lambda 禁用**：`finished` 信号连接禁止使用 lambda（竞态条件），使用独立方法。

### 12.4 项目结构约定

- **代码核心隔离**：所有功能代码在 `code/` 文件夹下，用户更新只需替换此目录。
- **系统模型**：存储在 `system_model/` 文件夹，按用途分子目录。
- **用户工作区**：在 `my_project/` 文件夹下，`default` 子目录作为自由空间模式回退。
- **.gitignore 模式**：使用 `dir/*` + `!dir/.gitkeep` 模式保留目录结构但忽略内容。
- **入口点**：`main.py` 在项目根目录。
- **文档**：README 同时提供中英文版本。

### 12.5 日志约定

- 使用标准 `logging` 模块，输出到 `workspace/logs/app.log`。
- 按日期切割（`TimedRotatingFileHandler`），保留 7 天。
- 控制台输出 WARNING 及以上级别。
- 业务模块通过 `get_logger(__name__)` 获取 logger 实例。

### 12.6 ONNX Runtime 版本检测

`onnxruntime.__version__` 在不同安装中不可靠，应使用 `importlib.metadata.version('onnxruntime')` 并配合 `getattr` 回退进行版本检查。

---

*本文档由代码分析自动生成，反映项目当前状态。如需更新，请重新运行分析。*
