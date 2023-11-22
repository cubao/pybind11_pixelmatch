from __future__ import annotations

from pybind11_pixelmatch import Color

c = Color()
print(c.to_python())
print(c.r)
print(c.g)
print(c.b)
print(c.a)
