from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None


@dataclass
class DeviceInfo:
    name: str
    serial: str
    usb_type: str
    product_line: str


class RealSenseCamera:

    @staticmethod
    def list_devices() -> List[DeviceInfo]:
        if rs is None:
            return []
        try:
            ctx = rs.context()
            devices = []
            for dev in ctx.query_devices():
                devices.append(
                    DeviceInfo(
                        name=dev.get_info(rs.camera_info.name),
                        serial=dev.get_info(rs.camera_info.serial_number),
                        usb_type=dev.get_info(rs.camera_info.usb_type_descriptor),
                        product_line=dev.get_info(rs.camera_info.product_line),
                    )
                )
            return devices
        except Exception:
            return []

    @staticmethod
    def is_available() -> bool:
        return rs is not None

    def __init__(self):
        self.pipeline = None
        self.align = None
        self.colorizer = None
        self._running = False
        self._last_depth_frame = None

    def start(self, device_serial: str = "", width: int = 640, height: int = 480, fps: int = 30) -> bool:
        if rs is None:
            return False
        try:
            self.pipeline = rs.pipeline()
            config = rs.config()
            if device_serial:
                config.enable_device(device_serial)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            self.colorizer = rs.colorizer()
            self._running = True
            return True
        except Exception:
            return False

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self._running or self.pipeline is None:
            return (None, None)
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            self._last_depth_frame = depth_frame
            color_np = np.asanyarray(color_frame.get_data())
            depth_np = np.asanyarray(depth_frame.get_data())
            return (color_np, depth_np)
        except Exception:
            return (None, None)

    def colorize_depth(self, depth_np: np.ndarray) -> Optional[np.ndarray]:
        if rs is None or self.colorizer is None:
            return None
        try:
            colorized = self.colorizer.colorize(self._last_depth_frame)
            return np.asanyarray(colorized.get_data())
        except Exception:
            return None

    def stop(self):
        if self.pipeline is not None and self._running:
            self.pipeline.stop()
        self.pipeline = None
        self.align = None
        self.colorizer = None
        self._running = False
        self._last_depth_frame = None

    @property
    def running(self) -> bool:
        return self._running
