"""统一管理模型家族命名模板与增强预设相关常量。

本模块集中维护以下常量,供 trainer / train_widget / config 等模块共享引用,
避免在多处重复定义导致不一致:

- MODEL_FAMILY_TASK_MODEL_MAP: 各模型系列在不同任务下的权重文件命名模板
- AUGMENTATION_PRESET_LABELS:  增强预设的中文标签映射
- CUSTOM_AUGMENTATION_PRESET:  自定义预设标识
- AUGMENTATION_PRESET_ORDER:   预设的展示顺序
- AUGMENTATION_PRESETS:        各预设的具体参数取值
"""

MODEL_FAMILY_TASK_MODEL_MAP = {
    "yolo26": {
        "detect": "yolo26{size}.pt",
        "segment": "yolo26{size}-seg.pt",
        "classify": "yolo26{size}-cls.pt",
        "pose": "yolo26{size}-pose.pt",
    },
    "yolov8": {
        "detect": "yolov8{size}.pt",
        "segment": "yolov8{size}-seg.pt",
        "classify": "yolov8{size}-cls.pt",
        "pose": "yolov8{size}-pose.pt",
    },
}

CUSTOM_AUGMENTATION_PRESET = "custom"
AUGMENTATION_PRESET_ORDER = ["off", "light", "default", "strong"]
AUGMENTATION_PRESET_LABELS = {
    "off": "关闭",
    "light": "轻度",
    "default": "默认",
    "strong": "强增强",
    CUSTOM_AUGMENTATION_PRESET: "自定义",
}

AUGMENTATION_PRESETS = {
    "off": {
        "hsv_h": 0, "hsv_s": 0, "hsv_v": 0,
        "degrees": 0, "translate": 0, "scale": 0, "shear": 0, "perspective": 0,
        "flipud": 0, "fliplr": 0,
        "mosaic": 0, "mixup": 0, "cutmix": 0, "copy_paste": 0, "erasing": 0,
        "auto_augment": "",
    },
    "light": {
        "hsv_h": 0.01, "hsv_s": 0.3, "hsv_v": 0.2,
        "degrees": 5, "translate": 0.05, "scale": 0.3, "shear": 0, "perspective": 0,
        "flipud": 0, "fliplr": 0.3,
        "mosaic": 0.5, "mixup": 0, "cutmix": 0, "copy_paste": 0, "erasing": 0.2,
        "auto_augment": "randaugment",
    },
    "default": {
        "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4,
        "degrees": 0, "translate": 0.1, "scale": 0.5, "shear": 0, "perspective": 0,
        "flipud": 0, "fliplr": 0.5,
        "mosaic": 1.0, "mixup": 0, "cutmix": 0, "copy_paste": 0, "erasing": 0.4,
        "auto_augment": "randaugment",
    },
    "strong": {
        "hsv_h": 0.02, "hsv_s": 0.8, "hsv_v": 0.6,
        "degrees": 15, "translate": 0.2, "scale": 0.7, "shear": 10, "perspective": 0.001,
        "flipud": 0.2, "fliplr": 0.5,
        "mosaic": 1.0, "mixup": 0.2, "cutmix": 0.1, "copy_paste": 0.1, "erasing": 0.4,
        "auto_augment": "randaugment",
    },
}
