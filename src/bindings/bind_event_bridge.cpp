#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>

namespace py = pybind11;

namespace dm::bindings {

void forward_event_to_python(const std::string& event_type, const std::string& payload_json) {
    py::gil_scoped_acquire acquire;
    try {
        py::module bridge = py::module::import("dm_toolkit.native_event_bridge");
        py::object native_emit = bridge.attr("native_emit");
        if (payload_json.empty()) {
            native_emit(event_type, py::none());
        } else {
            py::module json = py::module::import("json");
            py::object obj = json.attr("loads")(payload_json);
            native_emit(event_type, obj);
        }
    } catch (const std::exception &e) {
        // Swallow to avoid crashing native code; real code should log
    }
}

void bind_event_bridge(py::module &m) {
    m.def("forward_event_to_python", &forward_event_to_python,
          py::arg("event_type"), py::arg("payload_json") = std::string());
}

} // namespace dm::bindings
