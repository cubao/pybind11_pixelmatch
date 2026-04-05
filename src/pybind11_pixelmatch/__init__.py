from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np

from ._core import (
    Color,
    Options,
    __doc__,
    __version__,
    pixelmatch,
    rgb2yiq,
)

Backend = Literal["cv2", "pillow"]


def read_image(
    path: str,
    *,
    backend: Backend | None = None,
) -> np.ndarray:
    assert Path(path).is_file(), f"{path} does not exist"

    if backend is None:
        try:
            import PIL  # noqa: F401

            backend = "pillow"
        except ImportError:
            backend = "cv2"

    if backend == "cv2":
        import cv2
        import numpy as np

        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img.shape[2] == 3:
            B, G, R = cv2.split(img)
            A = np.ones(B.shape, dtype=B.dtype) * 255
            img = cv2.merge((R, G, B, A))
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        return img

    # pillow backend
    import numpy as np
    from PIL import Image

    return np.array(Image.open(path).convert("RGBA"))


def write_image(
    path: str,
    img: np.ndarray,
    *,
    backend: Backend | None = None,
) -> None:
    Path(path).resolve().parent.mkdir(parents=True, exist_ok=True)

    if backend is None:
        try:
            import PIL  # noqa: F401

            backend = "pillow"
        except ImportError:
            backend = "cv2"

    if backend == "cv2":
        import cv2

        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(path), img)
        return

    # pillow backend
    from PIL import Image

    mode = "RGB" if img.shape[2] == 3 else "RGBA"
    Image.fromarray(img, mode).save(path)


def normalize_color(rgba):
    if rgba is None:
        return rgba
    if isinstance(rgba, Color):
        return rgba.clone()
    if isinstance(rgba, str):
        if not rgba.startswith(("rgba(", "rgb(")):
            # 0xrrggbb, 0xrrggbbaa, #rrggbb, #rrggbbaa
            return Color(rgba)
        rgba = rgba.split("(", 1)[-1].split(")", 1)[0].split(",")
        rgba = [int(x) for x in rgba]
    r, g, b = rgba[:3]
    a = rgba[3] if len(rgba) > 3 else 255
    return Color(r, g, b, a)


__all__ = [
    "__doc__",
    "__version__",
    "Color",
    "normalize_color",
    "Options",
    "rgb2yiq",
    "pixelmatch",
    "read_image",
    "write_image",
]
