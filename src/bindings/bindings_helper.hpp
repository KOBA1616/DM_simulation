#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <nlohmann/json.hpp>

namespace py = pybind11;

inline nlohmann::json py_to_json(py::handle obj) {
    if (obj.is_none()) return nullptr;
    if (py::isinstance<py::bool_>(obj)) return obj.cast<bool>();
    if (py::isinstance<py::int_>(obj)) return obj.cast<int>();
    if (py::isinstance<py::float_>(obj)) return obj.cast<double>();
    if (py::isinstance<py::str>(obj)) return obj.cast<std::string>();
    if (py::isinstance<py::list>(obj)) {
        nlohmann::json j = nlohmann::json::array();
        for (auto item : obj.cast<py::list>()) j.push_back(py_to_json(item));
        return j;
    }
    if (py::isinstance<py::dict>(obj)) {
        nlohmann::json j = nlohmann::json::object();
        for (auto item : obj.cast<py::dict>()) {
            j[item.first.cast<std::string>()] = py_to_json(item.second);
        }
        return j;
    }
    return nullptr;
}
