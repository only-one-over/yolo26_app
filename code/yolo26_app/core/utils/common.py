"""通用工具函数:图像读写、路径处理等跨模块共享功能。"""
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def imread_unicode(path: str) -> np.ndarray:
    """读取含中文路径的图像,np.fromfile + cv2.imdecode 绕过 OpenCV 中文路径限制。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path: str, img: np.ndarray) -> bool:
    """写入含中文路径的图像,cv2.imencode + np.tofile 绕过 OpenCV 中文路径限制。"""
    try:
        ext = Path(path).suffix
        result, encoded = cv2.imencode(ext, img)
        if result:
            encoded.tofile(path)
            return True
        return False
    except Exception:
        return False
