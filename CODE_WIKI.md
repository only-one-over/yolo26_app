# YOLO26 App Code Wiki

## 项目概述

YOLO26 App 是一个基于 Ultralytics YOLO26 的桌面端标注-训练-推理一体化应用，使用 PyQt6 构建 GUI，提供了从数据标注、模型训练到推理测试的完整工作流。

---

## 目录结构

```
yolo26_app/
├── main.py                          # 应用入口文件
├── pyproject.toml                   # 项目配置文件
├── requirements.txt                 # 依赖清单
├── LICENSE                          # MIT 许可证
├── README.md                        # 项目说明文档
└── yolo26_app/                      # 应用主包
    ├── __init__.py                  # 包初始化文件
    ├── core/                        # 核心业务逻辑模块
    │   ├── __init__.py
    │   ├── config.py                # 数据模型定义
    │   ├── project_manager.py       # 项目管理
    │   ├── label_manager.py         # 标注类别管理
    │   ├── annotation_canvas.py     # 标注画布组件
    │   ├── yolo_exporter.py         # YOLO 数据集导出
    │   ├── trainer.py               # YOLO 训练器
    │   ├── predictor.py             # YOLO 推理器
    │   ├── auto_annotator.py        # 自动标注工具
    │   └── realsense_camera.py      # RealSense 深度相机
    └── ui/                          # 用户界面模块
        ├── __init__.py
        ├── main_window.py           # 主窗口
        ├── annotate_widget.py       # 标注模块界面
        ├── train_widget.py          # 训练模块界面
        ├── test_widget.py           # 测试模块界面
        └── styles.py                # QSS 样式表
```

---

## 整体架构

### 架构分层

```
┌─────────────────────────────────────────────────────────────┐
│                      UI Layer (ui/)                         │
│  ┌──────────────┬──────────────┬──────────────┐             │
│  │AnnotateWidget│  TrainWidget  │  TestWidget │             │
│  └──────────────┴──────────────┴──────────────┘             │
│                      MainWindow                              │
├─────────────────────────────────────────────────────────────┤
│                   Business Logic Layer (core/)                │
│  ┌────────────┬────────────┬────────────┬────────────┐        │
│  │Annotation  │  Trainer   │ Predictor  │ YOLO      │        │
│  │Canvas      │  (QThread) │            │ Exporter  │        │
│  └────────────┴────────────┴────────────┴────────────┘        │
│  ┌────────────┬────────────┬────────────┬────────────┐        │
│  │Label       │  Project   │Auto        │RealSense   │        │
│  │Manager     │  Manager   │Annotator   │Camera      │        │
│  └────────────┴────────────┴────────────┴────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                    Data Model Layer (config.py)               │
│  ┌──────────────┬──────────────┬──────────────┐             │
│  │  ClassItem  │ TrainConfig  │ProjectConfig│             │
│  └──────────────┴──────────────┴──────────────┘             │
├─────────────────────────────────────────────────────────────┤
│                   External Libraries                          │
│  Ultralytics YOLO26 | PyQt6 | OpenCV | PyTorch | NumPy      │
└─────────────────────────────────────────────────────────────┘
```

### 模块职责划分

| 模块 | 职责 | 主要类 |
|------|------|--------|
| **ui/** | 用户界面层，负责GUI展示和用户交互 | MainWindow, AnnotateWidget, TrainWidget, TestWidget |
| **core/** | 核心业务逻辑层，处理数据和算法 | AnnotationCanvas, YOLOTrainer, YOLOPredictor, YOLOExporter |
| **core/config.py** | 数据模型层，定义数据结构 | ClassItem, TrainConfig, ProjectConfig |

---

## 核心模块详解

### 1. UI 模块 (yolo26_app/ui/)

#### 1.1 MainWindow (main_window.py)

主窗口类，负责应用的整体布局和导航。

**关键属性：**
- `current_project_config: Optional[ProjectConfig]` - 当前项目配置
- `annotate_widget: AnnotateWidget` - 标注模块实例
- `train_widget: TrainWidget` - 训练模块实例
- `test_widget: TestWidget` - 测试模块实例
- `nav_buttons: List[QPushButton]` - 导航按钮列表

**关键方法：**

| 方法 | 描述 | 签名 |
|------|------|------|
| `_init_ui` | 初始化UI组件和布局 | `() -> None` |
| `_switch_page` | 切换页面 | `(index: int) -> None` |
| `_set_project_config` | 设置当前项目配置 | `(config: ProjectConfig) -> None` |
| `_new_project` | 创建新项目 | `() -> None` |
| `_open_project` | 打开现有项目 | `() -> None` |
| `_save_app_state` | 保存应用状态到文件 | `() -> None` |
| `_restore_app_state` | 从文件恢复应用状态 | `() -> None` |

**信号流程：**
```
TestWidget.model_loaded --→ AnnotateWidget.set_yolo_model
MainWindow.project_config --→ TrainWidget/AnnotateWidget/TestWidget
```

#### 1.2 AnnotateWidget (annotate_widget.py)

标注模块界面，提供图像标注功能。

**关键属性：**
- `_label_manager: LabelManager` - 类别管理器
- `_annotations_dict: Dict[str, List[AnnotationItem]]` - 标注数据字典
- `_current_image_path: str` - 当前图片路径
- `_image_list: List[str]` - 图片列表
- `_scene: AnnotationScene` - 标注场景
- `_view: AnnotationView` - 标注视图
- `_yolo_annotator: YOLOPreAnnotator` - YOLO预标注器
- `_sam_annotator: SAMAnnotator` - SAM分割标注器
- `_video_tracker: VideoTracker` - 视频追踪器
- `_dino_annotator: GroundingDINOAnnotator` - 文本检测标注器

**关键方法：**

| 方法 | 描述 | 签名 |
|------|------|------|
| `_setup_ui` | 初始化标注界面UI | `() -> None` |
| `_import_images` | 导入单张或多张图片 | `() -> None` |
| `_import_video` | 导入视频并提取帧 | `() -> None` |
| `_import_directory` | 导入整个目录的图片 | `() -> None` |
| `_load_image` | 加载图片到画布 | `(image_path: str) -> None` |
| `_export_dataset` | 导出YOLO格式数据集 | `() -> None` |
| `_auto_annotate` | YOLO预标注 | `() -> None` |
| `_sam_annotate` | SAM交互式分割 | `() -> None` |
| `_video_track` | 视频帧间追踪 | `() -> None` |
| `_text_detect` | 文本驱动检测 | `() -> None` |
| `_batch_detect` | 批量图片检测 | `() -> None` |
| `set_yolo_model` | 设置YOLO模型用于预标注 | `(model) -> None` |
| `save_state` | 保存标注状态 | `() -> dict` |
| `restore_state` | 恢复标注状态 | `(state: dict) -> None` |

**工具类型：**
- `rect` - 矩形标注工具
- `polygon` - 多边形标注工具
- `select` - 选择工具
- `sam` - SAM分割工具

#### 1.3 TrainWidget (train_widget.py)

训练模块界面，配置和启动YOLO模型训练。

**关键属性：**
- `_trainer: Optional[YOLOTrainer]` - 训练器实例
- `_project_path: str` - 项目路径

**支持的模型配置：**

| 模型大小 | 参数量 | 显存需求 | 速度评级 |
|---------|--------|---------|---------|
| n (Nano) | 3.2M | ≥2GB | ★★★★★ |
| s (Small) | 11.2M | ≥4GB | ★★★★ |
| m (Medium) | 25.9M | ≥8GB | ★★★ |
| l (Large) | 43.7M | ≥12GB | ★★ |
| x (XLarge) | 68.4M | ≥16GB | ★ |

**支持的训练任务：**

| 任务类型 | 描述 | 模型后缀 |
|---------|------|---------|
| detect | 目标检测 | yolo26{n/s/m/l/x}.pt |
| segment | 实例分割 | yolo26{n/s/m/l/x}-seg.pt |
| classify | 图像分类 | yolo26{n/s/m/l/x}-cls.pt |
| pose | 姿态估计 | yolo26{n/s/m/l/x}-pose.pt |

**关键方法：**

| 方法 | 描述 | 签名 |
|------|------|------|
| `_setup_ui` | 初始化训练界面UI | `() -> None` |
| `_build_config` | 构建训练配置对象 | `() -> TrainConfig` |
| `_validate_dataset` | 验证数据集配置 | `() -> bool` |
| `_on_start` | 开始训练 | `() -> None` |
| `_on_stop` | 停止训练 | `() -> None` |
| `_on_progress` | 处理训练进度信号 | `(current: int, total: int) -> None` |
| `_on_log` | 处理日志消息 | `(message: str) -> None` |
| `_on_finished` | 处理训练完成 | `(message: str) -> None` |

#### 1.4 TestWidget (test_widget.py)

测试模块界面，模型推理和验证。

**关键属性：**
- `predictor: YOLOPredictor` - 推理器实例
- `cap: Optional[cv2.VideoCapture]` - 视频捕获对象
- `rs_camera: RealSenseCamera` - RealSense相机
- `_batch_images: List[str]` - 批量图片列表
- `_show_depth: bool` - 是否显示深度图

**支持的推理输入：**
- 单张图片
- 图片目录
- 视频文件
- USB摄像头
- Intel RealSense深度相机

**关键方法：**

| 方法 | 描述 | 签名 |
|------|------|------|
| `_init_ui` | 初始化测试界面UI | `() -> None` |
| `_on_load_model` | 加载YOLO模型 | `() -> None` |
| `_on_select_image` | 选择单张图片推理 | `() -> None` |
| `_select_image_directory` | 选择图片目录批量推理 | `() -> None` |
| `_start_capture` | 启动视频捕获 | `(source: Union[str, int]) -> None` |
| `_on_timer_timeout` | 定时器回调，处理视频帧 | `() -> None` |
| `_on_validate` | 验证模型指标 | `() -> None` |
| `_on_export_clicked` | 显示导出设置 | `() -> None` |
| `_on_confirm_export` | 确认模型导出 | `() -> None` |
| `_display_np_image` | 显示推理结果图像 | `(image_np: np.ndarray) -> None` |

**导出的模型格式：**
- `onnx` - ONNX格式
- `torchscript` - TorchScript格式
- `openvino` - Intel OpenVINO格式
- `engine` - NVIDIA TensorRT格式

#### 1.5 Styles (styles.py)

定义应用的主题样式，支持深色和亮色主题。

**主题变量：**
- `DARK_STYLE` - Catppuccin Mocha 深色主题
- `LIGHT_STYLE` - Catppuccin Latte 亮色主题

**配色方案（深色主题）：**
- 背景色: `#1e1e2e`
- 表面色: `#313244`
- 边框色: `#45475a`
- 主色调: `#89b4fa` (蓝色)
- 文字色: `#cdd6f4`

---

### 2. Core 模块 (yolo26_app/core/)

#### 2.1 Config (config.py)

数据模型定义模块，包含所有核心数据类。

##### ClassItem

标注类别数据类。

```python
@dataclass
class ClassItem:
    name: str = ""           # 类别名称
    color: str = "#FF0000"   # 十六进制颜色值
```

**方法：**
- `to_dict() -> dict` - 序列化为字典
- `from_dict(data: dict) -> ClassItem` - 从字典反序列化

##### TrainConfig

训练配置数据类。

```python
@dataclass
class TrainConfig:
    task: str = "detect"           # 任务类型
    model_size: str = "n"          # 模型大小
    data: str = ""                 # 数据集配置文件路径
    epochs: int = 100              # 训练轮数
    batch: int = 16                # 批大小
    imgsz: int = 640               # 输入图像尺寸
    device: str = ""               # 设备 (auto/cpu/0/0,1)
    optimizer: str = "auto"        # 优化器
    lr0: float = 0.01              # 初始学习率
    patience: int = 100            # 早停耐心值
    project: str = ""               # 项目目录
    name: str = ""                 # 实验名称
```

**方法：**
- `to_dict() -> dict` - 序列化为字典
- `from_dict(data: dict) -> TrainConfig` - 从字典反序列化

##### ProjectConfig

项目配置数据类，包含项目的完整配置信息。

```python
@dataclass
class ProjectConfig:
    project_name: str = ""                    # 项目名称
    project_path: str = ""                    # 项目路径
    classes: List[ClassItem] = field(...)     # 类别列表
    train_config: TrainConfig = field(...)      # 训练配置
    created_at: str = ""                       # 创建时间 (ISO格式)
    last_opened: str = ""                     # 最后打开时间
```

**方法：**
- `to_dict() -> dict` - 序列化为字典
- `from_dict(data: dict) -> ProjectConfig` - 从字典反序列化
- `save(path: Union[str, Path]) -> None` - 保存到文件
- `load(path: Union[str, Path]) -> ProjectConfig` - 从文件加载

#### 2.2 ProjectManager (project_manager.py)

项目管理器，处理项目的创建、打开和配置管理。

**主要功能：**
- 创建新项目并初始化目录结构
- 打开现有项目
- 管理最近项目列表
- 提供数据集和模型目录路径

**关键方法：**

| 方法 | 描述 | 签名 |
|------|------|------|
| `create_project` | 创建新项目 | `(name: str, path: str) -> ProjectConfig` |
| `open_project` | 打开现有项目 | `(path: str) -> ProjectConfig` |
| `get_recent_projects` | 获取最近项目列表 | `() -> List[str]` |
| `add_recent_project` | 添加到最近项目 | `(path: str) -> None` |
| `get_dataset_dir` | 获取数据集目录 | `(config: ProjectConfig) -> Path` |
| `get_models_dir` | 获取模型目录 | `(config: ProjectConfig) -> Path` |

**项目目录结构：**
```
project_path/
├── project_config.json      # 项目配置文件
├── classes.txt             # 类别列表
├── datasets/               # 数据集目录
├── models/                 # 模型目录
└── runs/                   # 训练运行记录
```

#### 2.3 LabelManager (label_manager.py)

标注类别管理器，处理类别的增删改查。

**颜色调色板：**
预定义20种颜色用于类别自动分配。

**关键方法：**

| 方法 | 描述 | 签名 |
|------|------|------|
| `load_from_project` | 从项目配置加载类别 | `(config: ProjectConfig) -> None` |
| `save_to_project` | 保存类别到项目配置 | `(config: ProjectConfig) -> None` |
| `add_class` | 添加类别 | `(name: str) -> ClassItem` |
| `remove_class` | 移除类别 | `(index: int) -> None` |
| `update_class` | 更新类别名称 | `(index: int, name: str) -> None` |
| `get_class_by_index` | 按索引获取类别 | `(index: int) -> Optional[ClassItem]` |
| `get_class_index` | 按名称获取索引 | `(name: str) -> int` |
| `get_all_classes` | 获取所有类别 | `() -> List[ClassItem]` |

#### 2.4 AnnotationCanvas (annotation_canvas.py)

标注画布模块，提供图像标注的交互功能。

##### AnnotationItem

标注项数据类。

```python
@dataclass
class AnnotationItem:
    class_index: int              # 类别索引
    rect: QRectF = field(...)    # 矩形区域
    polygon: QPolygonF = field(...)  # 多边形点集
    item_type: str = "rect"      # 类型: "rect" 或 "polygon"
```

##### AnnotationScene

标注场景类，继承自 QGraphicsScene，处理标注交互逻辑。

**主要功能：**
- 矩形标注绘制
- 多边形标注绘制
- SAM 交互式分割
- 标注选择和删除
- 缩放和拖拽

**关键属性：**
- `_current_tool: str` - 当前工具
- `_current_class_index: int` - 当前类别索引
- `_annotations: list[AnnotationItem]` - 标注列表
- `_drawing: bool` - 是否正在绘制
- `_selected_index: int` - 选中标注索引
- `_class_colors: list[str]` - 类别颜色列表

**关键方法：**

| 方法 | 描述 | 签名 |
|------|------|------|
| `set_tool` | 设置工具 | `(tool: str) -> None` |
| `set_current_class` | 设置当前类别 | `(index: int) -> None` |
| `set_class_colors` | 设置类别颜色 | `(colors: list[str]) -> None` |
| `mousePressEvent` | 鼠标按下事件 | `(event) -> None` |
| `mouseMoveEvent` | 鼠标移动事件 | `(event) -> None` |
| `mouseReleaseEvent` | 鼠标释放事件 | `(event) -> None` |
| `mouseDoubleClickEvent` | 双击事件(完成多边形) | `(event) -> None` |
| `run_sam_prediction` | 运行SAM预测 | `() -> None` |
| `apply_sam_result` | 应用SAM结果 | `(masks, scores) -> None` |
| `delete_selected` | 删除选中标注 | `() -> None` |
| `clear_annotations` | 清除所有标注 | `() -> None` |
| `get_annotations` | 获取标注列表 | `() -> list[AnnotationItem]` |
| `load_annotations` | 加载标注列表 | `(annotations: list) -> None` |

**信号：**
- `annotations_changed: pyqtSignal` - 标注变化信号

##### AnnotationView

标注视图类，继承自 QGraphicsView，提供画布的显示和交互。

**关键属性：**
- `_scale_factor: float` - 缩放因子

**关键方法：**

| 方法 | 描述 | 签名 |
|------|------|------|
| `fit_to_item` | 适应项目尺寸 | `() -> None` |
| `wheelEvent` | 鼠标滚轮缩放 | `(event) -> None` |
| `mousePressEvent` | 中键拖拽 | `(event) -> None` |

#### 2.5 Trainer (trainer.py)

YOLO模型训练器，在后台线程中运行训练。

##### YOLOTrainer

训练器类，继承自 QThread。

**信号：**
- `progress_signal: pyqtSignal(int, int)` - 进度信号 (当前, 总数)
- `log_signal: pyqtSignal(str)` - 日志消息信号
- `finished_signal: pyqtSignal(str)` - 训练完成信号
- `error_signal: pyqtSignal(str)` - 错误信号

**任务模型映射：**
```python
TASK_MODEL_MAP = {
    "detect": "yolo26{size}.pt",
    "segment": "yolo26{size}-seg.pt",
    "classify": "yolo26{size}-cls.pt",
    "pose": "yolo26{size}-pose.pt",
}
```

**关键方法：**

| 方法 | 描述 | 签名 |
|------|------|------|
| `run` | 训练主循环 | `() -> None` |
| `stop` | 停止训练 | `() -> None` |

##### _QtLogHandler

日志处理器，将日志消息转发到Qt信号。

```python
class _QtLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None
```

#### 2.6 Predictor (predictor.py)

YOLO模型推理器，处理模型加载和推理。

##### YOLOPredictor

推理器类，封装YOLO模型的加载和推理功能。

**关键属性：**
- `model: Optional[YOLO]` - YOLO模型实例
- `model_path: str` - 模型文件路径

**关键方法：**

| 方法 | 描述 | 签名 |
|------|------|------|
| `load_model` | 加载模型 | `(path: str) -> bool` |
| `predict_image` | 预测单张图片 | `(image_path: str, conf: float) -> Tuple[np.ndarray, object]` |
| `predict_frame` | 预测视频帧 | `(frame_np: np.ndarray, conf: float) -> Tuple[np.ndarray, object]` |
| `validate_model` | 验证模型指标 | `(data: str) -> dict` |
| `export_model` | 导出模型 | `(format: str, output_dir: str) -> str` |
| `get_model_info` | 获取模型信息 | `() -> dict` |
| `_draw_results` | 绘制推理结果 | `(image_np: np.ndarray, results: object) -> np.ndarray` |

**返回格式：**
- 成功：`(annotated_image, results_object)`
- 失败：`(np.array([]), None)`

#### 2.7 YOLOExporter (yolo_exporter.py)

YOLO数据集导出器，将标注数据导出为YOLO格式。

##### YOLOExporter

导出器类，静态方法实现导出逻辑。

**关键方法：**

| 方法 | 描述 | 签名 |
|------|------|------|
| `export_dataset` | 导出数据集 | `(annotations_dict, classes, output_dir, train_ratio) -> Tuple[str, Dict]` |

**导出目录结构：**
```
output_dir/
├── images/
│   ├── train/           # 训练集图片
│   └── val/             # 验证集图片
├── labels/
│   ├── train/           # 训练集标签
│   └── val/             # 验证集标签
└── data.yaml            # 数据集配置文件
```

**YOLO标签格式：**

矩形框：
```
<class_index> <center_x> <center_y> <width> <height>
```
所有值都是相对于图像尺寸的归一化值 (0-1)。

多边形：
```
<class_index> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

#### 2.8 AutoAnnotator (auto_annotator.py)

自动标注工具集，支持多种预标注方式。

##### YOLOPreAnnotator

使用YOLO模型进行预标注。

```python
class YOLOPreAnnotator:
    def __init__(self) -> None
    def set_model(self, model) -> None
    def annotate(self, image_path: str, conf: float = 0.25, 
                 class_mapping: Optional[Dict[int, int]] = None) -> List[AnnotationItem]
```

**功能：**
- 处理检测框
- 处理分割掩码
- 支持类别映射

##### SAMAnnotator

使用SAM (Segment Anything Model) 进行交互式分割标注。

```python
class SAMAnnotator:
    def __init__(self) -> None
    @property
    def available(self) -> bool
    @staticmethod
    def scan_model_file(directory: str) -> Optional[Tuple[str, str]]
    def load_model(self, model_path: str, model_type: str = "vit_b", 
                   device: str = "cuda") -> bool
    def set_image(self, image: np.ndarray) -> None
    def predict(self, point_coords: np.ndarray, point_labels: np.ndarray) -> List[AnnotationItem]
```

**支持的SAM模型类型：**
- `vit_h` - ViT-Huge
- `vit_l` - ViT-Large
- `vit_b` - ViT-Base (默认)
- `vit_t` - MobileSAM

##### VideoTracker

使用OpenCV追踪器进行视频帧间标注传播。

```python
class VideoTracker:
    @staticmethod
    def get_tracker_name() -> str
    def track_frames(self, image_paths: List[str], 
                    initial_annotations: List[AnnotationItem],
                    max_frames: int = 0) -> Dict[int, List[AnnotationItem]]
```

**追踪器优先级：** CSRT > KCF > MIL

##### GroundingDINOAnnotator

使用Grounding DINO进行文本驱动的零样本检测。

```python
class GroundingDINOAnnotator:
    def __init__(self) -> None
    @property
    def available(self) -> bool
    def load_model(self, config_path: str, weights_path: str) -> bool
    def detect(self, image_path: str, text_prompt: str,
               box_threshold: float = 0.35,
               text_threshold: float = 0.25,
               class_mapping: Optional[Dict[str, int]] = None) -> List[AnnotationItem]
```

#### 2.9 RealSenseCamera (realsense_camera.py)

Intel RealSense深度相机支持模块。

##### DeviceInfo

设备信息数据类。

```python
@dataclass
class DeviceInfo:
    name: str           # 设备名称
    serial: str         # 序列号
    usb_type: str       # USB类型
    product_line: str    # 产品线
```

##### RealSenseCamera

RealSense相机控制类。

**关键方法：**

| 方法 | 描述 | 签名 |
|------|------|------|
| `is_available` (静态) | 检查pyrealsense2是否可用 | `() -> bool` |
| `list_devices` (静态) | 列出可用设备 | `() -> List[DeviceInfo]` |
| `start` | 启动相机流 | `(device_serial, width, height, fps) -> bool` |
| `get_frames` | 获取彩色和深度帧 | `() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]` |
| `colorize_depth` | 深度图着色 | `(depth_np: np.ndarray) -> Optional[np.ndarray]` |
| `stop` | 停止相机流 | `() -> None` |
| `running` (属性) | 是否正在运行 | `-> bool` |

---

## 依赖关系

### 核心依赖

| 依赖包 | 版本要求 | 用途 |
|--------|---------|------|
| ultralytics | >=8.0 | YOLO模型框架 |
| PyQt6 | >=6.0 | GUI框架 |
| opencv-python | >=4.6.0 | 图像处理 |
| numpy | >=1.20.0 | 数值计算 |
| pyyaml | >=5.3.1 | YAML配置解析 |

### 可选依赖

| 依赖包 | 版本要求 | 用途 |
|--------|---------|------|
| pyrealsense2 | >=2.50.0 | RealSense深度相机 |
| torch | - | PyTorch深度学习框架 (GPU支持) |

### 间接依赖（通过ultralytics）

| 依赖包 | 用途 |
|--------|------|
| torch | 深度学习框架 |
| torchvision | 计算机视觉工具 |
| pillow | 图像处理 |
| matplotlib | 可视化 |
| seaborn | 统计图形 |
| scipy | 科学计算 |

---

## 项目运行方式

### 环境准备

#### 1. 克隆仓库

```bash
git clone https://github.com/<your-username>/yolo26-app.git
cd yolo26-app
```

#### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

#### 3. 安装依赖

```bash
pip install -r requirements.txt
```

#### 4. 安装PyTorch（GPU支持）

访问 [PyTorch官网](https://pytorch.org/get-started/locally/) 安装对应CUDA版本的PyTorch。

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### 5. 安装可选依赖

```bash
# RealSense深度相机支持
pip install pyrealsense2
```

### 启动应用

#### 方式一：模块方式运行（推荐）

```bash
python -m yolo26_app.main
```

#### 方式二：直接运行入口文件

```bash
python main.py
```

#### 方式三：安装后运行

```bash
pip install -e .
yolo26-app
```

### 完整工作流程

```
1. 新建项目
   文件 → 新建项目 → 输入名称和路径

2. 导入数据
   标注页面 → 导入图片/视频/目录

3. 定义类别
   左侧类别面板 → 点击 "+" 添加类别

4. 绘制标注
   选择工具（矩形/多边形）→ 在画布上标注

5. 导出数据集
   点击"导出数据集" → 选择输出目录

6. 训练模型
   切换到训练页面 → 选择data.yaml → 配置参数 → 开始训练

7. 测试推理
   切换到测试页面 → 加载best.pt模型 → 选择推理源
```

### GPU配置

应用会自动检测CUDA可用性，状态栏显示：
- 🟢 GPU: [设备名称] - 表示GPU可用
- 🔴 CPU - 表示仅使用CPU

手动指定设备：
- `cpu` - 强制使用CPU
- `0` - 使用第一块GPU
- `0,1` - 使用多块GPU

---

## 信号与槽连接

### 跨模块通信

```
MainWindow
├── TestWidget.model_loaded --→ AnnotateWidget.set_yolo_model
├── _set_project_config
│   ├── --→ AnnotateWidget.set_project_config
│   ├── --→ TrainWidget.set_project_config
│   └── --→ TestWidget.set_project_config
└── AnnotatorWidget
    └── YOLOTrainer signals
        ├── progress_signal
        ├── log_signal
        ├── finished_signal
        └── error_signal
```

### 标注模块内部信号

```
AnnotationScene.annotations_changed --→ (通知标注变化)
_SamWorker signals (QThread)
├── encoding_done
├── prediction_done
└── error_occurred
```

---

## 数据流

### 标注数据流

```
用户输入 (鼠标事件)
    ↓
AnnotationScene.mousePressEvent
    ↓
创建 AnnotationItem
    ↓
存储到 _annotations
    ↓
绘制到画布 (_draw_annotation)
    ↓
触发 annotations_changed 信号
    ↓
AnnotationsWidget 保存到 _annotations_dict
```

### 训练数据流

```
TrainWidget._on_start
    ↓
创建 YOLOTrainer (QThread)
    ↓
YOLOTrainer.run()
    ├─ 加载 YOLO 模型
    ├─ 调用 model.train()
    └─ 发射信号 (progress/log/finished/error)
    ↓
TrainWidget 处理信号
    ├─ _on_progress → 更新进度条
    ├─ _on_log → 显示日志
    ├─ _on_finished → 显示结果
    └─ _on_error → 显示错误
```

### 推理数据流

```
TestWidget._on_timer_timeout (视频流)
TestWidget._on_select_image (单张图片)
    ↓
YOLOPredictor.predict_frame / predict_image
    ↓
YOLO 模型推理
    ↓
YOLOPredictor._draw_results (绘制结果)
    ↓
TestWidget._display_np_image (显示)
```

---

## 配置文件格式

### project_config.json

```json
{
  "project_name": "my_project",
  "project_path": "/path/to/project",
  "classes": [
    {"name": "person", "color": "#FF6B6B"},
    {"name": "car", "color": "#4ECDC4"}
  ],
  "train_config": {
    "task": "detect",
    "model_size": "n",
    "data": "/path/to/data.yaml",
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "device": "",
    "optimizer": "auto",
    "lr0": 0.01,
    "patience": 100,
    "project": "",
    "name": ""
  },
  "created_at": "2024-01-01T00:00:00",
  "last_opened": "2024-01-02T00:00:00"
}
```

### data.yaml (YOLO数据集配置)

```yaml
path: /path/to/exported/dataset
train: images/train
val: images/val
nc: 2
names: ['person', 'car']
```

---

## 错误处理

### 常见错误及解决方案

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| "SAM 模型加载失败" | SAM未安装或模型路径错误 | 安装segment-anything，下载模型权重 |
| "无法打开视频源" | 摄像头被占用或视频文件损坏 | 关闭其他应用，检查视频文件 |
| "模型加载失败" | 模型文件不存在或格式错误 | 检查文件路径，使用正确的模型格式 |
| "数据集配置文件不存在" | data.yaml路径错误 | 检查并更正数据集配置文件路径 |

### 日志查看

训练过程的日志通过 `YOLOTrainer.log_signal` 实时显示在训练界面的日志文本框中。

---

## 扩展开发

### 添加新的自动标注器

1. 在 `auto_annotator.py` 中创建新的标注器类
2. 实现 `annotate` 方法返回 `List[AnnotationItem]`
3. 在 `annotate_widget.py` 中添加对应的UI和调用逻辑

### 添加新的导出格式

1. 在 `yolo_exporter.py` 中添加新的导出方法
2. 实现数据转换逻辑
3. 在UI中添加导出选项

### 添加新的推理输入源

1. 在 `test_widget.py` 中添加新的输入处理方法
2. 实现帧获取和推理逻辑
3. 更新UI添加对应的输入按钮

---

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

**注意**：本项目依赖 Ultralytics YOLO26，其采用 [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 许可证。如果修改并分发 Ultralytics 源码，需遵守 AGPL-3.0 的要求。
