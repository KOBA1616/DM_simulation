#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "bindings/bind_command_generator.hpp"
#include "engine/commands/command_generator.hpp"
#include "engine/actions/intent_generator.hpp"
#include "core/action.hpp"

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
    if (!c.str_param.empty()) d["str_param"] = c.str_param;
    return d;
}

void bind_command_generator(py::module_ &m) {
    py::class_<CommandDef>(m, "CommandDef")
        .def_readwrite("type", &CommandDef::type)
        .def_readwrite("uid", &CommandDef::uid)
        .def("to_dict", [](const CommandDef &c){ return commanddef_to_dict(c); });

    // expose a simple module-level helper that returns dicts for now
    m.def("generate_commands", [](const dm::core::GameState& gs, py::object card_db_obj){
        // Try to resolve Python-side card_db to native CardDatabase using EngineCompat
        std::map<dm::core::CardID, dm::core::CardDefinition> db_resolved;
        try {
            py::object engine_compat = py::module::import("dm_toolkit.engine.compat");
            py::object resolved = engine_compat.attr("_resolve_db")(card_db_obj);
            db_resolved = resolved.cast<std::map<dm::core::CardID, dm::core::CardDefinition>>();
        } catch (...) {
            // If resolution fails, attempt to cast directly; empty map on failure
            try {
                db_resolved = card_db_obj.cast<std::map<dm::core::CardID, dm::core::CardDefinition>>();
            } catch (...) {
                db_resolved.clear();
            }
        }

        CommandGenerator cg;
        auto cmds = cg.generate_commands(gs, db_resolved);
        py::list out;
        if (!cmds.empty()) {
            for (const auto &c : cmds) out.append(commanddef_to_dict(c));
            return out;
        }

        // Fallback: call legacy IntentGenerator and map Actions -> command dicts
        try {
            auto actions = dm::engine::IntentGenerator::generate_legal_actions(gs, db_resolved);
            for (const auto &a : actions) {
                py::dict d;
                using PI = dm::core::PlayerIntent;
                std::string t = "UNKNOWN";
                switch (a.type) {
                    case PI::PLAY_CARD: t = "PLAY_FROM_ZONE"; break;
                    case PI::DECLARE_PLAY: t = "PLAY_FROM_ZONE"; break;
                    case PI::MANA_CHARGE: t = "MANA_CHARGE"; break;
                    case PI::PASS: t = "PASS_TURN"; break;
                    case PI::ATTACK_PLAYER: t = "ATTACK"; break;
                    case PI::ATTACK_CREATURE: t = "ATTACK"; break;
                    case PI::BLOCK: t = "BLOCK"; break;
                    case PI::SELECT_TARGET: t = "SELECT_TARGET"; break;
                    default: t = "UNKNOWN"; break;
                }
                d["type"] = t;
                // Map ids
                if (a.card_id != 0) d["instance_id"] = static_cast<int>(a.card_id);
                if (a.source_instance_id >= 0) d["source_id"] = a.source_instance_id;
                if (a.target_instance_id >= 0) d["target_id"] = a.target_instance_id;
                if (a.target_player >= 0) d["target_player"] = static_cast<int>(a.target_player);
                if (a.slot_index >= 0) d["slot_index"] = a.slot_index;
                out.append(d);
            }
            return out;
        } catch (...) {
            return out; // empty
        }
    }, py::arg("state"), py::arg("card_db"));
}
