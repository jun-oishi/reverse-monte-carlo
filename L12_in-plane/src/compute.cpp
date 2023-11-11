#include "compute.hpp"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(compute, m) {
  m.doc() = "compute module";
  m.def("linspace", &linspace, "linspace");
  m.def("computeIq", &computeIq, "computeIq");
}