# 推理与模型导出

## 推理输入

测试页支持以下输入源：

| 输入源 | 用途 |
| --- | --- |
| 单张图片 | 快速检查检测结果 |
| 图片目录 | 批量处理图片 |
| 视频文件 | 实时逐帧推理 |
| USB 摄像头 | 在线摄像头推理 |
| Intel RealSense | RGB 与深度图推理 |

视频和摄像头推理在后台线程中执行；当推理速度低于采集速度时会跳过过期帧，避免延迟不断累积。停止后再次打开视频或相机，会创建新的推理工作线程。

## 模型验证

“验证模型”会对已加载的 `.pt` 模型执行验证并显示 mAP 指标。ONNX、TensorRT 等导出格式用于推理，通常不作为应用内验证模型。

模型会通过共享会话串行访问，因此标注页批量检测、测试页推理、验证与导出不会同时占用同一个 YOLO 模型或 GPU。

## 导出格式

| 格式 | 典型用途 |
| --- | --- |
| ONNX | 通用跨平台推理 |
| TorchScript | PyTorch 生态 |
| OpenVINO | Intel CPU/GPU |
| TensorRT | NVIDIA GPU 高性能推理 |
| CoreML | Apple 设备 |
| TFLite / NCNN | 移动端与轻量设备 |
| Paddle / MNN / RKNN | 对应推理生态与 NPU |

导出页会根据格式显示可用参数，例如输入尺寸、FP16、INT8、动态尺寸、batch、opset、workspace 和 simplify。

## 导出后的验证

ONNX 和 TensorRT 导出后会进行可加载性检查。导出失败时，应用会保留已有模型文件并恢复备份，避免覆盖可用产物。

建议交付模型时同时保留：

- 原始 `.pt` 权重，用于继续训练和重新导出。
- ONNX，作为跨设备的通用回退格式。
- 类别名称、输入尺寸、置信度和 NMS 参数。
- 一张示例图片及其预期输出。

## TensorRT 注意事项

TensorRT 仅适用于 NVIDIA GPU，需要单独安装运行库。生成的 `.engine` 与导出设备、GPU 架构、CUDA 和 TensorRT 版本有关，不建议在未知硬件之间直接复用。

推荐流程：

1. 在目标部署机器安装并验证 TensorRT。
2. 使用同一台机器导出或构建 `.engine`。
3. 保留 ONNX 作为回退模型。
4. 记录 GPU、驱动、CUDA、TensorRT 和 Ultralytics 版本。

环境安装与故障处理见[环境与模型](environment-and-models.md)和[诊断与排障](troubleshooting.md)。
