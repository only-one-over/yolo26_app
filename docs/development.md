# 开发指南

## 项目结构

```text
main.py
code/yolo26_app/
├── core/     # 项目、标注、训练、预测、导出、持久化与设备逻辑
└── ui/       # 主窗口、标注页、训练页、推理页、样式和图标
tests/        # 单元与 UI 可靠性测试
```

核心模块：

| 模块 | 职责 |
| --- | --- |
| `core/project_manager.py` | 项目创建、打开与安全路径校验 |
| `core/annotation_canvas.py` | 标注绘制、选择、撤销与重做 |
| `core/auto_annotator.py` | YOLO、SAM2、Grounding DINO 辅助标注 |
| `core/trainer.py` | Ultralytics 训练与实时进度回调 |
| `core/predictor.py` | 加载、预测、验证、模型导出 |
| `core/model_session.py` | 同一 YOLO 模型的线程安全共享访问 |
| `ui/annotation.py` | 标注工作流 |
| `ui/training.py` | 训练配置和可视化 |
| `ui/inference.py` | 图片、视频、相机推理与导出 |

## 本地开发

```bash
pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m mypy
```

UI 测试通过 `QT_QPA_PLATFORM=offscreen` 在无显示环境运行。涉及 GPU、TensorRT、相机或真实模型的测试应增加对应标记，并保持 CPU 环境可运行的单元测试。

## 扩展方式

### 新增辅助标注器

在 `core/auto_annotator.py` 中实现标注器，统一返回 `AnnotationItem`；通过共享 `ModelSession` 调用 YOLO，避免同一模型并发访问。

### 新增模型导出格式

在 `core/predictor.py` 增加格式参数处理、导出与验证逻辑，再在 `ui/export_dialog.py` 声明可见参数和预设。导出失败必须保留已有文件并恢复备份。

### 新增推理输入源

在 `ui/inference.py` 添加采集适配层。采集、推理和 UI 更新必须分离；停止操作应等待工作线程结束或明确阻止窗口关闭。

## 文档维护

首页只保留安装和基本使用。功能、环境、排障和开发改动应同步更新本目录的对应文件，并确保 [docs/README.md](README.md) 的导航可用。
