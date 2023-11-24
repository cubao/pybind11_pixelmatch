#include <pixelmatch/pixelmatch.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace pybind11::literals;
using rvp = py::return_value_policy;

inline bool validate_buffer_info(const py::buffer_info& buf1, const py::buffer_info& buf2) {
  // https://github.com/pybind/pybind11/blob/master/tests/test_buffers.cpp
  // should be RGBA
  if (buf1.ndim != 3 || buf2.ndim != 3) {
    return false;
  }
  if (buf1.shape[0] != buf2.shape[0] || buf1.shape[1] != buf2.shape[1] ||
      buf1.shape[2] != buf2.shape[2]) {
    return false;
  }
  if (buf1.shape[2] != 4) {
    return false;
  }
  return true;
}

using Color = pixelmatch::Color;
using Options = pixelmatch::Options;
inline std::string stringify(const Color& self) {
  return "rgba(" + std::to_string(self.r) + "," + std::to_string(self.g) + "," +
         std::to_string(self.b) + "," + std::to_string(self.a) + ")";
}
inline std::string stringify(const Options& self) {
  return std::string("{threshold=") + std::to_string(self.threshold) +       //
         std::string(",includeAA=") + (self.includeAA ? "true" : "false") +  //
         std::string(",alpha=") + std::to_string(self.alpha) +               //
         std::string(",aaColor=") + stringify(self.aaColor) +                //
         std::string(",diffColor=") + stringify(self.diffColor) +            //
         std::string(",diffColorAlt=") +
         (self.diffColorAlt ? stringify(*self.diffColorAlt) : std::string("None")) +  //
         std::string(",diffMask=") + (self.diffMask ? "true" : "false") + "}";
}

inline int pixelmatch_fn(const py::buffer& img1, const py::buffer& img2,
                         const py::buffer* out = nullptr,
                         const pixelmatch::Options& options = pixelmatch::Options()) {
  auto buf1 = img1.request();
  auto buf2 = img2.request();
  if (!validate_buffer_info(buf1, buf2)) {
    return -1;
  }
  pixelmatch::span<const uint8_t> image1(reinterpret_cast<const uint8_t*>(buf1.ptr), buf1.size);
  pixelmatch::span<const uint8_t> image2(reinterpret_cast<const uint8_t*>(buf2.ptr), buf2.size);
  pixelmatch::span<uint8_t> output(nullptr, 0);
  if (out) {
    auto buf = out->request(true);
    if (buf.readonly || !validate_buffer_info(buf, buf1)) {
      return -1;
    }
    output = pixelmatch::span<uint8_t>(reinterpret_cast<uint8_t*>(buf.ptr), buf.size);
  }
  int height = buf1.shape[0];
  int width = buf1.shape[1];
  int stride_in_pixels = width;
  return pixelmatch::pixelmatch(image1, image2, output, width, height, stride_in_pixels, options);
}

PYBIND11_MODULE(_core, m) {
  m.doc() = R"pbdoc(
    )pbdoc";

  py::class_<Color>(m, "Color", py::module_local())  //
      .def(py::init<>())
      .def(py::init<uint8_t, uint8_t, uint8_t, uint8_t>(), "r"_a, "g"_a, "b"_a, "a"_a)
      .def_readwrite("r", &Color::r)
      .def_readwrite("g", &Color::g)
      .def_readwrite("b", &Color::b)
      .def_readwrite("a", &Color::a)
      .def(
          "from_python",
          [](Color& self, const std::vector<uint8_t>& rgba) -> Color& {
            self.r = rgba[0];
            self.g = rgba[1];
            self.b = rgba[2];
            self.a = rgba[3];
            return self;
          },
          "rgba"_a, rvp::reference_internal)
      .def("to_python",
           [](const Color& self) -> std::vector<uint8_t> {
             return {self.r, self.g, self.b, self.a};
           })
      .def("clone", [](const Color& self) -> Color { return self; })
      //
      .def("__str__", [](const Color& self) -> std::string { return stringify(self); });

  py::class_<Options>(m, "Options", py::module_local())  //
      .def(py::init<>())
      .def_readwrite("threshold", &Options::threshold)
      .def_readwrite("includeAA", &Options::includeAA)
      .def_readwrite("alpha", &Options::alpha)
      .def_readwrite("aaColor", &Options::aaColor, rvp::reference_internal)
      .def_readwrite("diffColor", &Options::diffColor, rvp::reference_internal)
      .def_readwrite("diffColorAlt", &Options::diffColorAlt, rvp::reference_internal)
      .def_readwrite("diffMask", &Options::diffMask)
      .def("clone", [](const Options& self) -> Options { return self; })
      //
      .def("__str__", [](const Options& self) -> std::string { return stringify(self); });

  m.def(
      "rgb2yiq",
      [](uint8_t r, uint8_t g, uint8_t b) -> std::vector<float> {
        float y = r * 0.29889531f + g * 0.58662247f + b * 0.11448223f;
        float i = r * 0.59597799f - g * 0.27417610f - b * 0.32180189f;
        float q = r * 0.21147017f - g * 0.52261711f + b * 0.31114694f;
        return {y, i, q};
      },
      "r"_a, "g"_a, "b"_a);

  m.def(
      "pixelmatch",
      [](const py::buffer& img1, const py::buffer& img2, const py::buffer& out,
         const Options& options) -> int { return pixelmatch_fn(img1, img2, &out, options); },
      "img1"_a, "img2"_a, py::kw_only(),  //
      "output"_a,                         //
      "options"_a = Options());
  m.def(
      "pixelmatch",
      [](const py::buffer& img1, const py::buffer& img2, const Options& options) -> int {
        return pixelmatch_fn(img1, img2, nullptr, options);
      },
      "img1"_a, "img2"_a, py::kw_only(),  //
      "options"_a = Options());

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
