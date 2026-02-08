#include "bindings/bind_command_generator.hpp"
#include "engine/commands/command_generator.hpp"
#include "core/card_json_types.hpp"
#include <pybind11/stl.h>

using namespace dm;
using namespace dm::core;

void bind_command_generator(py::module& m) {
    m.def("generate_commands", [](const GameState& gs, const std::map<CardID, CardDefinition>& db){
        return dm::engine::CommandGenerator::generate_legal_commands(gs, db);
    }, py::arg("state"), py::arg("card_db"));

    // Alias for convenience to match guide (dm_ai_module.generate_commands)
}
