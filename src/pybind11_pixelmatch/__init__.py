from __future__ import annotations

from pathlib import Path

from ._core import (
    Color,
    Options,
    __doc__,
    __version__,
    pixelmatch,
    rgb2yiq,
)


def read_image(path):
    import cv2
    import numpy as np

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.shape[2] == 3:
        B, G, R = cv2.split(img)
        A = np.ones(B.shape, dtype=B.dtype) * 255
        img = cv2.merge((R, G, B, A))
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    return img


def write_image(path, img):
    import cv2

    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    Path(path).resolve().parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(path, img)


__all__ = [
    "__doc__",
    "__version__",
    "Color",
    "Options",
    "rgb2yiq",
    "pixelmatch",
    "read_image",
    "write_image",
]
