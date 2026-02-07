#include "bindings/bindings.hpp"
#include "bindings/bindings_helper.hpp"
#include "bindings/types.hpp"
#include "engine/game_instance.hpp"
#include <pybind11/stl_bind.h>
#include "engine/actions/intent_generator.hpp"
#include "engine/systems/card/card_registry.hpp"
#include "engine/systems/game_logic_system.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/systems/pipeline_executor.hpp"
#include "engine/systems/card/json_loader.hpp"
#include "engine/systems/flow/phase_manager.hpp"
#include "engine/systems/trigger_system/trigger_manager.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/game_command/action_commands.hpp"
#include "engine/utils/dev_tools.hpp"
#include <pybind11/stl.h>
#include <fstream>

using namespace dm;
using namespace dm::core;
using namespace dm::engine;

// Helper to access pipeline
std::shared_ptr<dm::engine::systems::PipelineExecutor> get_active_pipeline(GameState& state) {
    return std::static_pointer_cast<dm::engine::systems::PipelineExecutor>(state.active_pipeline);
}

void bind_engine(py::module& m) {
     // GameCommand bindings
    py::class_<dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::GameCommand>>(m, "GameCommand")
        .def("get_type", &dm::engine::game_command::GameCommand::get_type)
        .def("invert", &dm::engine::game_command::GameCommand::invert);

    // Removed CommandType binding here to avoid conflict with bind_core.cpp
    // The core CommandType (card_json_types.hpp) is now the primary one exposed.
    // If Engine uses a different CommandType enum, we should map or expose it differently.
    // However, GameCommand uses `dm::engine::game_command::CommandType` which seems to be different
    // or aliased.
    // Let's check `game_command.hpp`.

    // Assuming dm::engine::game_command::CommandType needs to be exposed for Engine Commands
    // but the conflict suggests they might share name or scope in python.
    // Let's rename this one to EngineCommandType if needed, or remove if unused by python side
    // except for return types.

    py::enum_<dm::engine::game_command::CommandType>(m, "EngineCommandType")
        .value("TRANSITION", dm::engine::game_command::CommandType::TRANSITION)
        .value("MUTATE", dm::engine::game_command::CommandType::MUTATE)
        .value("ATTACH", dm::engine::game_command::CommandType::ATTACH)
        .value("FLOW", dm::engine::game_command::CommandType::FLOW)
        .value("QUERY", dm::engine::game_command::CommandType::QUERY)
        .value("DECIDE", dm::engine::game_command::CommandType::DECIDE)
        .value("DECLARE_REACTION", dm::engine::game_command::CommandType::DECLARE_REACTION)
        .value("STAT", dm::engine::game_command::CommandType::STAT)
        .value("GAME_RESULT", dm::engine::game_command::CommandType::GAME_RESULT)
        .value("SHUFFLE", dm::engine::game_command::CommandType::SHUFFLE)
        .value("ADD_CARD", dm::engine::game_command::CommandType::ADD_CARD)
        .value("PLAY_CARD", dm::engine::game_command::CommandType::PLAY_CARD)
        .value("ATTACK", dm::engine::game_command::CommandType::ATTACK)
        .value("BLOCK", dm::engine::game_command::CommandType::BLOCK)
        .value("USE_ABILITY", dm::engine::game_command::CommandType::USE_ABILITY)
        .value("MANA_CHARGE", dm::engine::game_command::CommandType::MANA_CHARGE)
        .value("RESOLVE_PENDING_EFFECT", dm::engine::game_command::CommandType::RESOLVE_PENDING_EFFECT)
        .value("PASS_TURN", dm::engine::game_command::CommandType::PASS_TURN)
        .export_values();

    py::class_<dm::engine::game_command::TransitionCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::TransitionCommand>>(m, "TransitionCommand")
        .def(py::init<int, Zone, Zone, PlayerID, int>(),
             py::arg("instance_id"), py::arg("from"), py::arg("to"), py::arg("owner"), py::arg("dest_idx") = -1)
        .def_readwrite("card_instance_id", &dm::engine::game_command::TransitionCommand::card_instance_id)
        .def_readwrite("from_zone", &dm::engine::game_command::TransitionCommand::from_zone)
        .def_readwrite("to_zone", &dm::engine::game_command::TransitionCommand::to_zone)
        .def_readwrite("owner_id", &dm::engine::game_command::TransitionCommand::owner_id)
        .def_readwrite("destination_index", &dm::engine::game_command::TransitionCommand::destination_index);

    py::class_<dm::engine::game_command::MutateCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::MutateCommand>>(m, "MutateCommand")
        .def(py::init<int, dm::engine::game_command::MutateCommand::MutationType, int, std::string>(),
             py::arg("instance_id"), py::arg("type"), py::arg("val") = 0, py::arg("str") = "")
        .def_readwrite("target_instance_id", &dm::engine::game_command::MutateCommand::target_instance_id)
        .def_readwrite("mutation_type", &dm::engine::game_command::MutateCommand::mutation_type)
        .def_readwrite("int_value", &dm::engine::game_command::MutateCommand::int_value)
        .def_readwrite("str_value", &dm::engine::game_command::MutateCommand::str_value);

    py::enum_<dm::engine::game_command::MutateCommand::MutationType>(m, "MutationType")
        .value("TAP", dm::engine::game_command::MutateCommand::MutationType::TAP)
        .value("UNTAP", dm::engine::game_command::MutateCommand::MutationType::UNTAP)
        .value("POWER_MOD", dm::engine::game_command::MutateCommand::MutationType::POWER_MOD)
        .value("ADD_KEYWORD", dm::engine::game_command::MutateCommand::MutationType::ADD_KEYWORD)
        .value("REMOVE_KEYWORD", dm::engine::game_command::MutateCommand::MutationType::REMOVE_KEYWORD)
        .value("ADD_PASSIVE_EFFECT", dm::engine::game_command::MutateCommand::MutationType::ADD_PASSIVE_EFFECT)
        .value("ADD_COST_MODIFIER", dm::engine::game_command::MutateCommand::MutationType::ADD_COST_MODIFIER)
        .value("ADD_PENDING_EFFECT", dm::engine::game_command::MutateCommand::MutationType::ADD_PENDING_EFFECT)
        .export_values();

    py::class_<dm::engine::game_command::AttachCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::AttachCommand>>(m, "AttachCommand")
        .def(py::init<int, int, Zone>())
        .def_readwrite("card_to_attach_id", &dm::engine::game_command::AttachCommand::card_to_attach_id)
        .def_readwrite("target_base_card_id", &dm::engine::game_command::AttachCommand::target_base_card_id)
        .def_readwrite("source_zone", &dm::engine::game_command::AttachCommand::source_zone);

    py::class_<dm::engine::game_command::FlowCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::FlowCommand>>(m, "FlowCommand")
        .def(py::init<dm::engine::game_command::FlowCommand::FlowType, int>())
        .def_readwrite("flow_type", &dm::engine::game_command::FlowCommand::flow_type)
        .def_readwrite("new_value", &dm::engine::game_command::FlowCommand::new_value);

    py::enum_<dm::engine::game_command::FlowCommand::FlowType>(m, "FlowType")
        .value("PHASE_CHANGE", dm::engine::game_command::FlowCommand::FlowType::PHASE_CHANGE)
        .value("TURN_CHANGE", dm::engine::game_command::FlowCommand::FlowType::TURN_CHANGE)
        .value("STEP_CHANGE", dm::engine::game_command::FlowCommand::FlowType::STEP_CHANGE)
        .value("SET_ATTACK_SOURCE", dm::engine::game_command::FlowCommand::FlowType::SET_ATTACK_SOURCE)
        .value("SET_ATTACK_TARGET", dm::engine::game_command::FlowCommand::FlowType::SET_ATTACK_TARGET)
        .value("SET_ATTACK_PLAYER", dm::engine::game_command::FlowCommand::FlowType::SET_ATTACK_PLAYER)
        .value("SET_ACTIVE_PLAYER", dm::engine::game_command::FlowCommand::FlowType::SET_ACTIVE_PLAYER)
        .export_values();

    py::class_<dm::engine::game_command::QueryCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::QueryCommand>>(m, "QueryCommand")
        .def(py::init<std::string, std::vector<int>, std::map<std::string, int>>())
        .def_readwrite("query_type", &dm::engine::game_command::QueryCommand::query_type)
        .def_readwrite("valid_targets", &dm::engine::game_command::QueryCommand::valid_targets)
        .def_readwrite("params", &dm::engine::game_command::QueryCommand::params);

    py::class_<dm::engine::game_command::DecideCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::DecideCommand>>(m, "DecideCommand")
        .def(py::init<int, std::vector<int>, int>())
        .def_readwrite("query_id", &dm::engine::game_command::DecideCommand::query_id)
        .def_readwrite("selected_indices", &dm::engine::game_command::DecideCommand::selected_indices)
        .def_readwrite("selected_option_index", &dm::engine::game_command::DecideCommand::selected_option_index);

    py::class_<dm::engine::game_command::DeclareReactionCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::DeclareReactionCommand>>(m, "DeclareReactionCommand")
        .def(py::init<PlayerID, bool, int>())
        .def_readwrite("pass", &dm::engine::game_command::DeclareReactionCommand::pass)
        .def_readwrite("reaction_index", &dm::engine::game_command::DeclareReactionCommand::reaction_index)
        .def_readwrite("player_id", &dm::engine::game_command::DeclareReactionCommand::player_id);

    py::class_<dm::engine::game_command::StatCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::StatCommand>>(m, "StatCommand")
        .def(py::init<dm::engine::game_command::StatCommand::StatType, int>())
        .def_readwrite("stat", &dm::engine::game_command::StatCommand::stat)
        .def_readwrite("amount", &dm::engine::game_command::StatCommand::amount);

    py::enum_<dm::engine::game_command::StatCommand::StatType>(m, "StatType")
        .value("CARDS_DRAWN", dm::engine::game_command::StatCommand::StatType::CARDS_DRAWN)
        .value("CARDS_DISCARDED", dm::engine::game_command::StatCommand::StatType::CARDS_DISCARDED)
        .value("CREATURES_PLAYED", dm::engine::game_command::StatCommand::StatType::CREATURES_PLAYED)
        .value("SPELLS_CAST", dm::engine::game_command::StatCommand::StatType::SPELLS_CAST)
        .export_values();

    py::class_<dm::engine::game_command::GameResultCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::GameResultCommand>>(m, "GameResultCommand")
        .def(py::init<GameResult>())
        .def_readwrite("result", &dm::engine::game_command::GameResultCommand::result);

    py::class_<dm::engine::game_command::ShuffleCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::ShuffleCommand>>(m, "ShuffleCommand")
        .def(py::init<PlayerID>())
        .def_readwrite("player_id", &dm::engine::game_command::ShuffleCommand::player_id);

    // High-level action command bindings (used by Python-side execution helpers)
    py::class_<dm::engine::game_command::PlayCardCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::PlayCardCommand>>(m, "PlayCardCommand")
        .def(py::init<int>())
        .def_readwrite("card_instance_id", &dm::engine::game_command::PlayCardCommand::card_instance_id)
        .def_readwrite("target_slot_index", &dm::engine::game_command::PlayCardCommand::target_slot_index)
        .def_readwrite("is_spell_side", &dm::engine::game_command::PlayCardCommand::is_spell_side)
        .def_readwrite("spawn_source", &dm::engine::game_command::PlayCardCommand::spawn_source);

    py::class_<dm::engine::game_command::AttackCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::AttackCommand>>(m, "AttackCommand")
        .def(py::init<int, int, dm::core::PlayerID>())
        .def_readwrite("source_id", &dm::engine::game_command::AttackCommand::source_id)
        .def_readwrite("target_id", &dm::engine::game_command::AttackCommand::target_id)
        .def_readwrite("target_player_id", &dm::engine::game_command::AttackCommand::target_player_id);

    py::class_<dm::engine::game_command::BlockCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::BlockCommand>>(m, "BlockCommand")
        .def(py::init<int, int>())
        .def_readwrite("blocker_id", &dm::engine::game_command::BlockCommand::blocker_id)
        .def_readwrite("attacker_id", &dm::engine::game_command::BlockCommand::attacker_id);

    py::class_<dm::engine::game_command::UseAbilityCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::UseAbilityCommand>>(m, "UseAbilityCommand")
        .def(py::init<int, int>())
        .def_readwrite("source_id", &dm::engine::game_command::UseAbilityCommand::source_id)
        .def_readwrite("target_id", &dm::engine::game_command::UseAbilityCommand::target_id);

    py::class_<dm::engine::game_command::ManaChargeCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::ManaChargeCommand>>(m, "ManaChargeCommand")
        .def(py::init<int>())
        .def_readwrite("card_id", &dm::engine::game_command::ManaChargeCommand::card_id);

    py::class_<dm::engine::game_command::PassCommand, dm::engine::game_command::GameCommand, std::shared_ptr<dm::engine::game_command::PassCommand>>(m, "PassCommand")
        .def(py::init<>());

    py::class_<dm::engine::systems::PipelineExecutor, std::shared_ptr<dm::engine::systems::PipelineExecutor>>(m, "PipelineExecutor")
        .def(py::init<>())
        .def("set_context_var", &dm::engine::systems::PipelineExecutor::set_context_var)
        .def("get_context_var_int", [](const dm::engine::systems::PipelineExecutor& p, const std::string& key) {
            auto v = p.get_context_var(key);
            if (std::holds_alternative<int>(v)) return std::get<int>(v);
            return 0;
        })
        .def("get_execution_history", [](const dm::engine::systems::PipelineExecutor& p) {
            nlohmann::json j = p.get_execution_history();
            return py::module::import("json").attr("loads")(j.dump());
        })
        .def("dump_context", [](const dm::engine::systems::PipelineExecutor& p) {
            nlohmann::json j = p.dump_context();
            return py::module::import("json").attr("loads")(j.dump());
        })
        .def("dump_call_stack", [](const dm::engine::systems::PipelineExecutor& p) {
            nlohmann::json j = p.dump_call_stack();
            return py::module::import("json").attr("loads")(j.dump());
        })
        .def_readonly("execution_paused", &dm::engine::systems::PipelineExecutor::execution_paused)
        .def_readonly("waiting_for_key", &dm::engine::systems::PipelineExecutor::waiting_for_key)
        .def("resume", &dm::engine::systems::PipelineExecutor::resume)
        .def("execute", static_cast<void (dm::engine::systems::PipelineExecutor::*)(const std::vector<dm::core::Instruction>&, core::GameState&, const std::map<core::CardID, core::CardDefinition>&)>(&dm::engine::systems::PipelineExecutor::execute))
        // execute_command wrapper: accept a dict or bound command objects and dispatch
        .def("execute_command", [&m](dm::engine::systems::PipelineExecutor& exec, py::object obj, core::GameState& state) {
            try {
                std::unique_ptr<dm::engine::game_command::GameCommand> cmd;
                if (py::isinstance<py::dict>(obj)) {
                    py::dict d = obj.cast<py::dict>();
                    std::string t = py::str(d["type"]);
                    if (t == "PLAY_CARD") {
                        int iid = d["instance_id"].cast<int>();
                        auto p = std::make_unique<dm::engine::game_command::PlayCardCommand>(iid);
                        if (d.contains("target_slot_index")) p->target_slot_index = d["target_slot_index"].cast<int>();
                        if (d.contains("is_spell_side")) p->is_spell_side = d["is_spell_side"].cast<bool>();
                        cmd = std::move(p);
                    } else if (t == "MANA_CHARGE") {
                        int iid = d["instance_id"].cast<int>();
                        cmd = std::make_unique<dm::engine::game_command::ManaChargeCommand>(iid);
                    } else if (t == "PASS") {
                        cmd = std::make_unique<dm::engine::game_command::PassCommand>();
                    } else if (t == "ATTACK") {
                        int src = d["source_id"].cast<int>();
                        int tgt = d["target_id"].cast<int>();
                        int pid = 0;
                        if (d.contains("target_player_id")) pid = d["target_player_id"].cast<int>();
                        cmd = std::make_unique<dm::engine::game_command::AttackCommand>(src, tgt, (dm::core::PlayerID)pid);
                    }
                } else {
                    // If it's one of the bound command types, attempt to cast and copy fields
                    try {
                        if (py::isinstance(obj, m.attr("PlayCardCommand"))) {
                            auto pc = obj.cast<std::shared_ptr<dm::engine::game_command::PlayCardCommand>>();
                            auto p = std::make_unique<dm::engine::game_command::PlayCardCommand>(pc->card_instance_id);
                            p->target_slot_index = pc->target_slot_index;
                            p->is_spell_side = pc->is_spell_side;
                            p->spawn_source = pc->spawn_source;
                            cmd = std::move(p);
                        }
                    } catch(...) {}
                }

                if (cmd) exec.execute_command(std::move(cmd), state);
            } catch (const py::error_already_set& e) {
                throw;
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("Error in PipelineExecutor.execute_command wrapper: ") + e.what());
            } catch (...) {
                throw std::runtime_error("Unknown error in PipelineExecutor.execute_command wrapper");
            }
        });

    py::class_<dm::engine::systems::TriggerManager>(m, "TriggerManager")
        .def(py::init<>())
        .def("check_triggers", &dm::engine::systems::TriggerManager::check_triggers)
        .def("check_reactions", &dm::engine::systems::TriggerManager::check_reactions)
        .def("dispatch", &dm::engine::systems::TriggerManager::dispatch)
        .def("clear", &dm::engine::systems::TriggerManager::clear);

    // Systems
    py::class_<IntentGenerator>(m, "IntentGenerator")
        .def_static("generate_legal_actions", [](const GameState& gs, const std::map<CardID, CardDefinition>& db){
            try {
                std::filesystem::create_directories("logs");
                std::ofstream diag("logs/crash_diag.txt", std::ios::app);
                if (diag) {
                    diag << "BINDING generate_legal_actions entry player=" << static_cast<int>(gs.active_player_id)
                         << " phase=" << static_cast<int>(gs.current_phase) << "\n";
                    diag.close();
                }
            } catch(...) {}
            return IntentGenerator::generate_legal_actions(gs, db);
        });

    // Alias for backward compatibility
    m.attr("ActionGenerator") = m.attr("IntentGenerator");

    auto effect_resolver = py::class_<dm::engine::systems::GameLogicSystem>(m, "EffectResolver");
    effect_resolver
        .def_static("get_breaker_count", [](GameState& state, const CardInstance& card, const std::map<CardID, CardDefinition>& db) {
            return dm::engine::systems::GameLogicSystem::get_breaker_count(state, card, db);
        })
        .def_static("resume", [](GameState& state, const std::map<CardID, CardDefinition>& db, py::object input_val) {
             try {
                 if (!state.active_pipeline) return;

                 dm::engine::systems::ContextValue val;
                 if (py::isinstance<py::int_>(input_val)) {
                     val = input_val.cast<int>();
                 } else if (py::isinstance<py::str>(input_val)) {
                     val = input_val.cast<std::string>();
                 } else if (py::isinstance<py::list>(input_val)) {
                     std::vector<int> vec;
                     for (auto item : input_val.cast<py::list>()) {
                         vec.push_back(item.cast<int>());
                     }
                     val = vec;
                 }

                 auto pipeline = get_active_pipeline(state);
                 if (pipeline) {
                     pipeline->resume(state, db, val);
                     if (pipeline->call_stack.empty()) {
                         state.active_pipeline.reset();
                     }
                 }
             } catch (const py::error_already_set& e) {
                throw;
             } catch (const std::exception& e) {
                throw std::runtime_error("Error in EffectResolver.resume: " + std::string(e.what()));
             } catch (...) {
                throw std::runtime_error("Unknown error in EffectResolver.resume");
             }
        });


    py::class_<JsonLoader>(m, "JsonLoader")
        .def(py::init<>())
        .def_static("load_cards", [](const std::string& filepath) {
            auto map_val = JsonLoader::load_cards(filepath);
            // Move into shared pointer to prevent copy when returning to Python
            return std::make_shared<CardDatabase>(std::move(map_val));
        });

    py::class_<DevTools>(m, "DevTools")
        .def_static("move_cards", &DevTools::move_cards)
        .def_static("trigger_loop_detection", &DevTools::trigger_loop_detection);

    m.def("register_card_data", [](const CardData& data) {
         try {
             nlohmann::json j;
             dm::core::to_json(j, data);
             std::string json_str = j.dump();
             CardRegistry::load_from_json(json_str);
         } catch (const py::error_already_set& e) {
            throw;
         } catch (const std::exception& e) {
            throw std::runtime_error("Error in register_card_data: " + std::string(e.what()));
         } catch (...) {
            throw std::runtime_error("Unknown error in register_card_data");
         }
    });

    py::class_<GameInstance>(m, "GameInstance")
        .def(py::init([](uint32_t seed, std::shared_ptr<const CardDatabase> db) {
            return std::make_unique<GameInstance>(seed, db);
        }))
        .def(py::init<uint32_t>())
        .def_property_readonly("state", [](GameInstance &g) -> core::GameState& { return g.state; }, py::return_value_policy::reference_internal)
        .def("start_game", &GameInstance::start_game)
        .def("resolve_action", &GameInstance::resolve_action)
        .def("execute_command", [&m](GameInstance& gi, py::object obj) {
            try {
                // Prefer to dispatch via pipeline if available
                std::unique_ptr<dm::engine::game_command::GameCommand> cmd;
                if (py::isinstance<py::dict>(obj)) {
                    py::dict d = obj.cast<py::dict>();
                    std::string t = py::str(d["type"]);
                    if (t == "PLAY_CARD") {
                        int iid = d["instance_id"].cast<int>();
                        auto p = std::make_unique<dm::engine::game_command::PlayCardCommand>(iid);
                        if (d.contains("target_slot_index")) p->target_slot_index = d["target_slot_index"].cast<int>();
                        if (d.contains("is_spell_side")) p->is_spell_side = d["is_spell_side"].cast<bool>();
                        cmd = std::move(p);
                    } else if (t == "MANA_CHARGE") {
                        int iid = d["instance_id"].cast<int>();
                        cmd = std::make_unique<dm::engine::game_command::ManaChargeCommand>(iid);
                    } else if (t == "PASS") {
                        cmd = std::make_unique<dm::engine::game_command::PassCommand>();
                    }
                } else {
                    // Try to cast known bound command types
                    try {
                        if (py::isinstance(obj, m.attr("PlayCardCommand"))) {
                            auto pc = obj.cast<std::shared_ptr<dm::engine::game_command::PlayCardCommand>>();
                            auto p = std::make_unique<dm::engine::game_command::PlayCardCommand>(pc->card_instance_id);
                            p->target_slot_index = pc->target_slot_index;
                            p->is_spell_side = pc->is_spell_side;
                            p->spawn_source = pc->spawn_source;
                            cmd = std::move(p);
                        }
                    } catch(...) {}
                }

                if (cmd) {
                    // dispatch to state's execute_command
                    gi.state.execute_command(std::move(cmd));
                }
            } catch (const py::error_already_set& e) {
                throw;
            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("Error in GameInstance.execute_command wrapper: ") + e.what());
            } catch (...) {
                throw std::runtime_error("Unknown error in GameInstance.execute_command wrapper");
            }
        })
        .def("undo", &GameInstance::undo)
        .def("initialize_card_stats", &GameInstance::initialize_card_stats)
        .def("reset_with_scenario", &GameInstance::reset_with_scenario);

    py::class_<CardRegistry>(m, "CardRegistry")
        .def_static("register_card_data", [](const CardData& data) {
             try {
                 nlohmann::json j;
                 dm::core::to_json(j, data);
                 std::string json_str = j.dump();
                 CardRegistry::load_from_json(json_str);
             } catch (const py::error_already_set& e) {
                throw;
             } catch (const std::exception& e) {
                throw std::runtime_error("Error in CardRegistry.register_card_data: " + std::string(e.what()));
             } catch (...) {
                throw std::runtime_error("Unknown error in CardRegistry.register_card_data");
             }
        })
        .def_static("load_from_json", &CardRegistry::load_from_json)
        .def_static("get_all_cards", []() {
             // Return a copy of the static map as CardDatabase (shared ptr probably better but copied for safety/simplicity)
             // But CardDatabase is std::map<CardID, CardDefinition>.
             // CardRegistry has get_all_definitions()
             return CardRegistry::get_all_definitions();
        })
        .def_static("clear", &CardRegistry::clear);

    py::class_<PhaseManager>(m, "PhaseManager")
        .def_static("start_game", &PhaseManager::start_game)
        .def_static("setup_scenario", &PhaseManager::setup_scenario)
        .def_static("start_turn", &PhaseManager::start_turn)
        .def_static("next_phase", &PhaseManager::next_phase)
        .def_static("fast_forward", &PhaseManager::fast_forward)
        .def_static("check_game_over", &PhaseManager::check_game_over);
}
