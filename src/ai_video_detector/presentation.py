from __future__ import annotations

import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image


def frame_to_data_url(frame_bgr: np.ndarray, max_size: tuple[int, int] = (320, 320)) -> str:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    image.thumbnail(max_size)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=82)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"
