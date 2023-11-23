from __future__ import annotations

from pybind11_pixelmatch import Color

c = Color()
assert c.to_python() == [0, 0, 0, 0]
assert id(c) == id(c.from_python([1, 3, 5, 7]))
assert c.r == 1
assert c.g == 3
assert c.b == 5
assert c.a == 7
