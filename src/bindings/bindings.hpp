#pragma once
#include <pybind11/pybind11.h>

namespace py = pybind11;

void bind_core(py::module& m);
void bind_engine(py::module& m);
void bind_ai(py::module& m);
