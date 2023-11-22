#include <pixelmatch/pixelmatch.h>
#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;
using namespace pybind11::literals;
using rvp = py::return_value_policy;

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
      //
      ;

  // using Options = pixelmatch::Options;
  //  py::class_<Options>(m, "Options", py::module_local()) //
  //     .def(py::init<>())
  //     //
  //     ;

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
