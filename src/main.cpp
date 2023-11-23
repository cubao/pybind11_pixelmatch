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

PYBIND11_MODULE(_core, m) {
  m.doc() = R"pbdoc(
    )pbdoc";

  using Color = pixelmatch::Color;
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
      ;

  using Options = pixelmatch::Options;
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
      ;

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
      [](const py::buffer& img1, const py::buffer& img2, const py::buffer* out,
         const Options& options, size_t stride_in_pixels) -> int {
        auto buf1 = img1.request();
        auto buf2 = img2.request();
        if (!validate_buffer_info(buf1, buf2)) {
          return -1;
        }
        uint8_t* out_ptr = nullptr;
        if (out) {
          auto buf3 = out->request(true);
          if (buf3.readonly || !validate_buffer_info(buf1, buf3)) {
            return -1;
          }
          out_ptr = reinterpret_cast<uint8_t*>(buf3.ptr);
        }
        return -1;
        // return pixelmatch::pixelmatch();
      },
      "img1"_a, "img2"_a, py::kw_only(),  //
      "output"_a = nullptr,               //
      "options"_a = Options(),            //
      "stride_in_pixels"_a = 0);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
