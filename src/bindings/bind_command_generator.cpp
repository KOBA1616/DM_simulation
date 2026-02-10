#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "src/bindings/bind_command_generator.hpp"
#include "src/engine/commands/command_generator.hpp"

namespace py = pybind11;
using engine::commands::CommandDef;
using engine::commands::CommandGenerator;

static py::dict commanddef_to_dict(const CommandDef &c) {
    py::dict d;
    d["type"] = c.type;
    d["uid"] = c.uid;
    if (c.instance_id > 0) d["instance_id"] = c.instance_id;
    if (c.source_instance_id > 0) d["source_instance_id"] = c.source_instance_id;
    if (c.target_instance_id > 0) d["target_instance_id"] = c.target_instance_id;
    if (!c.from_zone.empty()) d["from_zone"] = c.from_zone;
    if (!c.to_zone.empty()) d["to_zone"] = c.to_zone;
    if (c.amount != 0) d["amount"] = c.amount;
    if (c.optional) d["optional"] = true;
    if (c.up_to) d["up_to"] = true;
    return d;
}

void bind_command_generator(py::module_ &m) {
    py::class_<CommandDef>(m, "CommandDef")
        .def_readwrite("type", &CommandDef::type)
        .def_readwrite("uid", &CommandDef::uid)
        .def("to_dict", [](const CommandDef &c){ return commanddef_to_dict(c); });

    py::class_<CommandGenerator>(m, "CommandGenerator")
        .def(py::init<>())
        .def("generate_commands", [](CommandGenerator &cg, py::object /*state*/, py::object /*card_db*/) {
            // Placeholder bridge. Real implementation should accept native GameState/CardDB.
            auto cmds = cg.generate_commands();
            py::list out;
            for (const auto &c : cmds) out.append(commanddef_to_dict(c));
            return out;
        });

    m.def("generate_commands", [](py::object state, py::object card_db) {
        CommandGenerator cg;
        auto cmds = cg.generate_commands();
        py::list out;
        for (const auto &c : cmds) out.append(commanddef_to_dict(c));
        return out;
    }, "Generate CommandDef list from state and card_db (placeholder)");
}
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
