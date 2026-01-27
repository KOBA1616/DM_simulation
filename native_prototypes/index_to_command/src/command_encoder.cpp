// Minimal native CommandEncoder prototype
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace py = pybind11;

int TOTAL_COMMAND_SIZE() {
    // Mirror Python CommandEncoder constants
    const int PASS_INDEX = 0;
    const int MANA_CHARGE_SLOTS = 19;
    const int PLAY_FROM_ZONE_SLOTS = 256;
    const int PLAY_FROM_ZONE_BASE = 1 + MANA_CHARGE_SLOTS;
    return PLAY_FROM_ZONE_BASE + PLAY_FROM_ZONE_SLOTS;
}

py::dict index_to_command(int idx) {
    if (idx == 0) {
        py::dict d;
        d["type"] = "PASS";
        return d;
    }
    if (idx >= 1 && idx < 20) {
        py::dict d;
        d["type"] = "MANA_CHARGE";
        d["slot_index"] = idx;
        return d;
    }
    int play_base = 20;
    if (idx >= play_base && idx < TOTAL_COMMAND_SIZE()) {
        py::dict d;
        d["type"] = "PLAY_FROM_ZONE";
        d["slot_index"] = idx - play_base;
        return d;
    }
    throw std::out_of_range("index out of range");
}

int command_to_index(py::dict cmd) {
    std::string t = py::str(cmd["type"]);
    if (t == "PASS") return 0;
    if (t == "MANA_CHARGE") {
        int si = py::int_(cmd["slot_index"]);
        if (si < 1 || si >= 20) throw std::invalid_argument("slot_index out of range for MANA_CHARGE");
        return si;
    }
    if (t == "PLAY_FROM_ZONE") {
        int si = py::int_(cmd["slot_index"]);
        if (si < 0 || si >= 256) throw std::invalid_argument("slot_index out of range for PLAY_FROM_ZONE");
        return 20 + si;
    }
    throw std::invalid_argument("unsupported command type");
}

PYBIND11_MODULE(command_encoder_native, m) {
    m.def("index_to_command", &index_to_command);
    m.def("command_to_index", &command_to_index);
    m.def("TOTAL_COMMAND_SIZE", &TOTAL_COMMAND_SIZE);
}
