# 诊断与排障

## 获取诊断信息

应用日志会写入项目根目录的 `logs/`：

- `app.log`：常规运行日志。
- `crash_YYYYMMDD_HHMMSS.log`：异常退出前生成的崩溃快照。

反馈问题前，请通过“帮助 -> 导出诊断报告”生成 ZIP。报告包含系统、Python、PyQt6、OpenCV、PyTorch、CUDA、Ultralytics、TensorRT 版本，GPU 状态和近期日志。

## 常见问题

| 现象 | 排查方式 |
| --- | --- |
| 状态栏显示 CPU | 执行 PyTorch GPU 检查；确认安装的是 GPU 版 Torch |
| CUDA out of memory | 减小 batch、imgsz，或选择更小模型 |
| GPU 检测较慢 | 首次 CUDA 初始化可能耗时，应用会在后台检测并缓存结果 |
| 训练进度不更新 | 检查训练日志和 `runs/<实验名>/results.csv` 是否生成 |
| 标注似乎丢失 | 重新打开项目检查 `annotations.json`；不要手动删除该文件 |
| ONNX 无检测结果 | 尝试 CPU ONNX Runtime，检查模型输入尺寸和类别配置 |
| TensorRT 导出失败 | 核对 GPU、驱动、CUDA、TensorRT、Ultralytics 版本；保留 ONNX 回退 |
| TensorRT 模块找不到 | 确认应用使用的虚拟环境已安装 `.[tensorrt]` |
| SAM2 加载失败 | 安装 `sam2` 并确认权重在 `system_model/sam2/` |

## 提交 Issue 时应附带

1. 操作系统和 GPU 型号。
2. 应用版本、Python、PyTorch、CUDA、Ultralytics、TensorRT 版本。
3. 导出的诊断报告。
4. 可复现步骤、输入模型格式和完整错误文本。

请勿上传包含敏感图片、项目数据或模型权重的诊断资料。
