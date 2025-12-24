#include "bindings/bindings.hpp"
#include "bindings/bindings_helper.hpp"
#include "engine/game_instance.hpp"
#include "engine/actions/intent_generator.hpp"
#include "engine/systems/card/card_registry.hpp"
#include "engine/systems/game_logic_system.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/systems/pipeline_executor.hpp"
#include "engine/systems/card/json_loader.hpp"
#include "engine/systems/flow/phase_manager.hpp"
#include "engine/systems/trigger_system/trigger_manager.hpp"
#include "engine/game_command/commands.hpp"
#include <pybind11/stl.h>

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
        .def("get_type", &dm::engine::game_command::GameCommand::get_type);

    py::enum_<dm::engine::game_command::CommandType>(m, "CommandType")
        .value("TRANSITION", dm::engine::game_command::CommandType::TRANSITION)
        .value("MUTATE", dm::engine::game_command::CommandType::MUTATE)
        .value("ATTACH", dm::engine::game_command::CommandType::ATTACH)
        .value("FLOW", dm::engine::game_command::CommandType::FLOW)
        .value("QUERY", dm::engine::game_command::CommandType::QUERY)
        .value("DECIDE", dm::engine::game_command::CommandType::DECIDE)
        .value("DECLARE_REACTION", dm::engine::game_command::CommandType::DECLARE_REACTION)
        .value("STAT", dm::engine::game_command::CommandType::STAT)
        .value("GAME_RESULT", dm::engine::game_command::CommandType::GAME_RESULT)
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

    py::class_<dm::engine::systems::PipelineExecutor, std::shared_ptr<dm::engine::systems::PipelineExecutor>>(m, "PipelineExecutor")
        .def(py::init<>())
        .def("set_context_var", &dm::engine::systems::PipelineExecutor::set_context_var)
        .def("get_context_var_int", [](const dm::engine::systems::PipelineExecutor& p, const std::string& key) {
            auto v = p.get_context_var(key);
            if (std::holds_alternative<int>(v)) return std::get<int>(v);
            return 0;
        })
        .def_readonly("execution_paused", &dm::engine::systems::PipelineExecutor::execution_paused)
        .def_readonly("waiting_for_key", &dm::engine::systems::PipelineExecutor::waiting_for_key)
        .def("resume", &dm::engine::systems::PipelineExecutor::resume)
        .def("execute", static_cast<void (dm::engine::systems::PipelineExecutor::*)(const std::vector<dm::core::Instruction>&, core::GameState&, const std::map<core::CardID, core::CardDefinition>&)>(&dm::engine::systems::PipelineExecutor::execute));

    py::class_<dm::engine::systems::TriggerManager>(m, "TriggerManager")
        .def(py::init<>())
        .def("check_triggers", &dm::engine::systems::TriggerManager::check_triggers)
        .def("check_reactions", &dm::engine::systems::TriggerManager::check_reactions)
        .def("dispatch", &dm::engine::systems::TriggerManager::dispatch)
        .def("clear", &dm::engine::systems::TriggerManager::clear);

    // Systems
    py::class_<IntentGenerator>(m, "IntentGenerator")
        .def_static("generate_legal_actions", &IntentGenerator::generate_legal_actions);

    // Alias for backward compatibility
    m.attr("ActionGenerator") = m.attr("IntentGenerator");

    py::class_<dm::engine::EffectSystem, std::unique_ptr<dm::engine::EffectSystem, py::nodelete>>(m, "EffectSystem")
        .def_static("instance", [](){ return &dm::engine::EffectSystem::instance(); }, py::return_value_policy::reference)
        .def_static("compile_action", [](GameState& state, const ActionDef& action, int source_id, std::map<CardID, CardDefinition>& db, py::object py_ctx) {
            try {
                std::vector<Instruction> instructions;
                std::map<std::string, int> execution_context;
                // Note: py_ctx extraction logic deferred/omitted as in original bindings.cpp
                dm::engine::EffectSystem::instance().compile_action(state, action, source_id, execution_context, db, instructions);
                return instructions;
            } catch (const py::error_already_set& e) {
                throw;
            } catch (const std::exception& e) {
                throw std::runtime_error("Error in EffectSystem.compile_action: " + std::string(e.what()));
            } catch (...) {
                throw std::runtime_error("Unknown error in EffectSystem.compile_action");
            }
        });

    auto effect_resolver = py::class_<dm::engine::systems::GameLogicSystem>(m, "EffectResolver");
    effect_resolver
        .def_static("resolve_action", [](GameState& state, const Action& action, const std::map<CardID, CardDefinition>& db){
            try {
                dm::engine::systems::GameLogicSystem::resolve_action(state, action, db);
            } catch (const py::error_already_set& e) {
                throw;
            } catch (const std::exception& e) {
                throw std::runtime_error("Error in EffectResolver.resolve_action: " + std::string(e.what()));
            } catch (...) {
                throw std::runtime_error("Unknown error in EffectResolver.resolve_action");
            }
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

    struct GenericCardSystemWrapper {};
    py::class_<GenericCardSystemWrapper>(m, "GenericCardSystem")
        .def_static("resolve_action", [](GameState& state, const ActionDef& action, int source_id) {
            try {
                auto db = CardRegistry::get_all_definitions();
                std::vector<Instruction> instructions;
                std::map<std::string, int> ctx;
                dm::engine::EffectSystem::instance().compile_action(state, action, source_id, ctx, db, instructions);
                if (!instructions.empty()) {
                    dm::engine::systems::PipelineExecutor pipeline;
                    pipeline.set_context_var("$source", source_id);
                    int controller = 0;
                    if ((size_t)source_id < state.card_owner_map.size()) controller = state.card_owner_map[source_id];
                    pipeline.set_context_var("$controller", controller);
                    pipeline.execute(instructions, state, db);
                }
            } catch (const py::error_already_set& e) {
                throw;
            } catch (const std::exception& e) {
                throw std::runtime_error("Error in GenericCardSystem.resolve_action: " + std::string(e.what()));
            } catch (...) {
                throw std::runtime_error("Unknown error in GenericCardSystem.resolve_action");
            }
        })
        .def_static("resolve_effect", [](GameState& state, const EffectDef& eff, int source_id) {
             try {
                 auto db = CardRegistry::get_all_definitions();
                 dm::engine::EffectSystem::instance().resolve_effect(state, eff, source_id, db);
             } catch (const py::error_already_set& e) {
                throw;
             } catch (const std::exception& e) {
                throw std::runtime_error("Error in GenericCardSystem.resolve_effect: " + std::string(e.what()));
             } catch (...) {
                throw std::runtime_error("Unknown error in GenericCardSystem.resolve_effect");
             }
        })
        .def_static("resolve_effect_with_db", [](GameState& state, const EffectDef& eff, int source_id, const std::map<CardID, CardDefinition>& db) {
             try {
                 dm::engine::EffectSystem::instance().resolve_effect(state, eff, source_id, db);
             } catch (const py::error_already_set& e) {
                throw;
             } catch (const std::exception& e) {
                throw std::runtime_error("Error in GenericCardSystem.resolve_effect_with_db: " + std::string(e.what()));
             } catch (...) {
                throw std::runtime_error("Unknown error in GenericCardSystem.resolve_effect_with_db");
             }
        })
        .def_static("resolve_effect_with_targets", [](GameState& state, const EffectDef& eff, const std::vector<int>& targets, int source_id, const std::map<CardID, CardDefinition>& db, std::map<std::string, int> ctx) {
             try {
                 dm::engine::EffectSystem::instance().resolve_effect_with_targets(state, eff, targets, source_id, db, ctx);
                 return ctx;
             } catch (const py::error_already_set& e) {
                throw;
             } catch (const std::exception& e) {
                throw std::runtime_error("Error in GenericCardSystem.resolve_effect_with_targets: " + std::string(e.what()));
             } catch (...) {
                throw std::runtime_error("Unknown error in GenericCardSystem.resolve_effect_with_targets");
             }
        })
        .def_static("resolve_action_with_context", [](GameState& state, int source_id, const ActionDef& action, const std::map<CardID, CardDefinition>& db, std::map<std::string, int> ctx) {
            try {
                std::vector<Instruction> instructions;
                dm::engine::EffectSystem::instance().compile_action(state, action, source_id, ctx, db, instructions);
                if (!instructions.empty()) {
                    dm::engine::systems::PipelineExecutor pipeline;
                    // Load context
                    for (const auto& [k, v] : ctx) {
                        pipeline.set_context_var("$" + k, v);
                    }

                    // Add default source/controller if missing
                     pipeline.set_context_var("$source", source_id);
                     int controller = 0;
                     if ((size_t)source_id < state.card_owner_map.size()) controller = state.card_owner_map[source_id];
                     pipeline.set_context_var("$controller", controller);

                    pipeline.execute(instructions, state, db);

                    // Extract context back
                    for (const auto& [k, v] : pipeline.context) {
                        if (std::holds_alternative<int>(v)) {
                             std::string clean_key = k;
                             if (clean_key.rfind("$", 0) == 0) clean_key = clean_key.substr(1);
                             ctx[clean_key] = std::get<int>(v);
                        }
                    }
                }
                return ctx;
            } catch (const py::error_already_set& e) {
                throw;
            } catch (const std::exception& e) {
                throw std::runtime_error("Error in GenericCardSystem.resolve_action_with_context: " + std::string(e.what()));
            } catch (...) {
                throw std::runtime_error("Unknown error in GenericCardSystem.resolve_action_with_context");
            }
        });

    py::class_<JsonLoader>(m, "JsonLoader")
        .def_static("load_cards", &JsonLoader::load_cards);

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
        .def(py::init<uint32_t, const std::map<core::CardID, core::CardDefinition>&>())
        .def(py::init<uint32_t>())
        .def_readonly("state", &GameInstance::state)
        .def("start_game", &GameInstance::start_game)
        .def("resolve_action", &GameInstance::resolve_action)
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
        .def_static("clear", &CardRegistry::clear);

    py::class_<PhaseManager>(m, "PhaseManager")
        .def_static("start_game", &PhaseManager::start_game)
        .def_static("setup_scenario", &PhaseManager::setup_scenario)
        .def_static("next_phase", &PhaseManager::next_phase)
        .def_static("check_game_over", &PhaseManager::check_game_over);
}
