from __future__ import annotations

from typing import Optional

import numpy as np

from . import Color, Options, pixelmatch, read_image, write_image

__aaColor = Color(255, 255, 0, 255)
__diffColor = Color(255, 0, 0, 255)


def main(
    img1: str,
    img2: str,
    output: str,
    *,
    threshold: float = 0.1,
    includeAA: bool = False,
    alpha: float = 0.1,
    aaColor: Color = __aaColor,
    diffColor: Color = __diffColor,
    diffColorAlt: Optional[Color] = None,  # noqa: UP007
    diffMask: bool = False,
):
    """
    Compares two images and generates a difference image.

    Parameters
    ----------
    img1 : str
        Path to the first image.
    img2 : str
        Path to the second image.
    output : str
        Path to write out the difference image.
    threshold : float, optional
        Matching threshold (0 to 1); smaller is more sensitive. Defaults to 0.1.
    includeAA : bool, optional
        Whether to include anti-aliased pixels in the diff,
        which disables anti-aliasing detection. Defaults to False.
    alpha : float, optional
        Opacity of original image in diff output. Defaults to 0.1.
    aaColor : Color, optional
        Color of anti-aliased pixels in diff output. Defaults to __aaColor.
    diffColor : Color, optional
        Color of different pixels in diff output. Defaults to __diffColor.
    diffColorAlt : Optional[Color], optional
        If set, detects dark on light differences
        between img1 and img2 and uses this as an alternative color to differentiate between the two.
        Defaults to None.
    diffMask : bool, optional
        Whether to draw the diff over a transparent background (a mask).
        Defaults to False.
    """
    options = Options()
    options.threshold = threshold
    options.includeAA = includeAA
    options.alpha = alpha
    options.aaColor = aaColor
    options.diffColor = diffColor
    options.diffColorAlt = diffColorAlt
    options.diffMask = diffMask
    print(f"options: {options}")  # noqa: T201

    i1 = read_image(img1)
    i2 = read_image(img2)
    assert i1.shape == i2.shape, f"image size mismatch: {i1.shape} != {i2.shape}"
    diff = np.zeros(i1.shape, dtype=i1.dtype)
    num = pixelmatch(i1, i2, output=diff, options=options)
    write_image(output, diff)
    print(f"wrote to {output}")  # noqa: T201
    print(f"#differente_pixels: {num}")  # noqa: T201


if __name__ == "__main__":
    import fire

    fire.core.Display = lambda lines, out: print(*lines, file=out)
    fire.Fire(main)
