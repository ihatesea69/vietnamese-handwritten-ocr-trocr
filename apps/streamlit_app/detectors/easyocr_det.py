"""
EasyOCR detector - chỉ dùng phần phát hiện (recognizer=False).
Trả về danh sách (x1, y1, x2, y2) theo tọa độ pixel.
"""
from __future__ import annotations
import numpy as np

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["vi", "en"], recognizer=False, verbose=False)
    return _reader


def detect(image_np: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Parameters
    ----------
    image_np : np.ndarray  shape (H, W, 3) RGB uint8

    Returns
    -------
    list of (x1, y1, x2, y2)
    """
    reader = _get_reader()
    bounds = reader.detect(image_np)
    # bounds[0][0] là horizontal_list: mỗi phần tử [x_min, x_max, y_min, y_max]
    boxes: list[tuple[int, int, int, int]] = []
    if bounds and bounds[0] and bounds[0][0]:
        for b in bounds[0][0]:
            x1, x2, y1, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
    return boxes
