#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

py::dict index_to_command(int idx) {
    py::dict d;
    if (idx == 0) {
        d["type"] = "PASS";
    } else if (idx > 0 && idx < 20) {
        d["type"] = "MANA_CHARGE";
        d["slot_index"] = idx;
    } else {
        d["type"] = "PLAY_FROM_ZONE";
        d["slot_index"] = idx - 20;
    }
    return d;
}

PYBIND11_MODULE(index_to_command_native, m) {
    m.doc() = "Prototype native index->command mapper for MCTS";
    m.def("index_to_command", &index_to_command, "Map action index to command dict");
}
