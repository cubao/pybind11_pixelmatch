from __future__ import annotations

from pybind11_pixelmatch import Color, Options

c = Color()
assert c.to_python() == [0, 0, 0, 0]
assert id(c) == id(c.from_python([1, 3, 5, 7]))
assert c.r == 1
assert c.g == 3
assert c.b == 5
assert c.a == 7


opt = Options()
assert abs(opt.threshold - 0.1) < 1e-8
assert not opt.includeAA
assert abs(opt.alpha - 0.1) < 1e-8
assert opt.aaColor.to_python() == [255, 255, 0, 255]
assert opt.diffColor.to_python() == [255, 0, 0, 255]
assert opt.diffColorAlt is None
assert not opt.diffMask

opt.threshold = 0.5
assert opt.threshold == 0.5
opt.includeAA = True
assert opt.includeAA
opt.alpha = 0.5
assert opt.alpha == 0.5
opt.aaColor.r = 123
assert opt.aaColor.to_python() == [123, 255, 0, 255]
opt.diffColor.r = 231
assert opt.diffColor.to_python() == [231, 0, 0, 255]
opt.diffColorAlt = Color(23, 45, 6, 7)
assert opt.diffColorAlt is not None
assert opt.diffColorAlt.to_python() == [23, 45, 6, 7]
