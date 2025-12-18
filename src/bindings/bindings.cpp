#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "engine/game_instance.hpp"
#include "engine/action_gen/action_generator.hpp"
#include "engine/game/effect_resolver.hpp"
#include "engine/systems/card/card_registry.hpp"
#include "engine/systems/game_logic_system.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/systems/pipeline_executor.hpp"
#include "ai/mcts/mcts.hpp"
#include "ai/evaluators/heuristic_evaluator.hpp"
#include "ai/utils/determinizer.hpp"
#include "core/card_json_types.hpp"
#include "engine/systems/card/json_loader.hpp"
#include "ai/mcts/parallel_runner.hpp"
#include "engine/game/phase_manager.hpp"
#include "core/card_stats.hpp"
#include "ai/solver/lethal_solver.hpp"
#include "engine/systems/card/condition_system.hpp"
#include "engine/systems/trigger_system/trigger_manager.hpp" // Added to restore missing bindings
#include "engine/systems/mana/mana_system.hpp" // Added to restore missing bindings
#include "engine/systems/cost_payment_system.hpp" // Added to restore missing bindings

namespace py = pybind11;
using namespace dm;
using namespace dm::core;
using namespace dm::engine;
using namespace dm::ai;

// Helper to access pipeline
std::shared_ptr<dm::engine::systems::PipelineExecutor> get_active_pipeline(GameState& state) {
    return std::static_pointer_cast<dm::engine::systems::PipelineExecutor>(state.active_pipeline);
}

PYBIND11_MODULE(dm_ai_module, m) {
    m.doc() = "Duel Masters AI Module";

    // Enums
    py::enum_<Civilization>(m, "Civilization")
        .value("FIRE", Civilization::FIRE)
        .value("WATER", Civilization::WATER)
        .value("NATURE", Civilization::NATURE)
        .value("LIGHT", Civilization::LIGHT)
        .value("DARKNESS", Civilization::DARKNESS)
        .value("ZERO", Civilization::ZERO)
        .value("NONE", Civilization::NONE)
        .export_values();

    py::enum_<CardType>(m, "CardType")
        .value("CREATURE", CardType::CREATURE)
        .value("SPELL", CardType::SPELL)
        .value("EVOLUTION_CREATURE", CardType::EVOLUTION_CREATURE)
        .value("CROSS_GEAR", CardType::CROSS_GEAR)
        .value("TAMASEED", CardType::TAMASEED)
        .value("PSYCHIC_CREATURE", CardType::PSYCHIC_CREATURE)
        .value("GR_CREATURE", CardType::GR_CREATURE)
        .export_values();

    py::enum_<Zone>(m, "Zone")
        .value("DECK", Zone::DECK)
        .value("HAND", Zone::HAND)
        .value("MANA", Zone::MANA)
        .value("GRAVEYARD", Zone::GRAVEYARD)
        .value("BATTLE", Zone::BATTLE)
        .value("SHIELD", Zone::SHIELD)
        .value("HYPER_SPATIAL", Zone::HYPER_SPATIAL)
        .value("GR_ZONE", Zone::GR_ZONE)
        .value("BUFFER", Zone::BUFFER)
        .value("STACK", Zone::STACK)
        .export_values();

    py::enum_<Phase>(m, "Phase")
        .value("START_OF_TURN", Phase::START_OF_TURN)
        .value("DRAW", Phase::DRAW)
        .value("MANA", Phase::MANA)
        .value("MAIN", Phase::MAIN)
        .value("ATTACK", Phase::ATTACK)
        .value("BLOCK", Phase::BLOCK)
        .value("END_OF_TURN", Phase::END_OF_TURN)
        .value("GAME_OVER", Phase::GAME_OVER)
        .export_values();

    py::enum_<ActionType>(m, "ActionType")
        .value("PASS", ActionType::PASS)
        .value("PLAY_CARD", ActionType::PLAY_CARD)
        .value("MANA_CHARGE", ActionType::MANA_CHARGE)
        .value("ATTACK_CREATURE", ActionType::ATTACK_CREATURE)
        .value("ATTACK_PLAYER", ActionType::ATTACK_PLAYER)
        .value("BLOCK", ActionType::BLOCK)
        .value("USE_SHIELD_TRIGGER", ActionType::USE_SHIELD_TRIGGER)
        .value("RESOLVE_EFFECT", ActionType::RESOLVE_EFFECT)
        .value("SELECT_TARGET", ActionType::SELECT_TARGET)
        .value("USE_ABILITY", ActionType::USE_ABILITY)
        .value("DECLARE_REACTION", ActionType::DECLARE_REACTION)
        .value("PLAY_CARD_INTERNAL", ActionType::PLAY_CARD_INTERNAL)
        .value("RESOLVE_BATTLE", ActionType::RESOLVE_BATTLE)
        .value("BREAK_SHIELD", ActionType::BREAK_SHIELD)
        .export_values();

    py::enum_<GameResult>(m, "GameResult")
        .value("NONE", GameResult::NONE)
        .value("P1_WIN", GameResult::P1_WIN)
        .value("P2_WIN", GameResult::P2_WIN)
        .value("DRAW", GameResult::DRAW)
        .export_values();

    py::enum_<TargetScope>(m, "TargetScope")
        .value("TARGET_SELECT", TargetScope::TARGET_SELECT)
        .export_values();

    py::enum_<EffectActionType>(m, "EffectActionType")
        .value("DRAW_CARD", EffectActionType::DRAW_CARD)
        .export_values();

    // Card Data Structures
    py::class_<CardKeywords>(m, "CardKeywords")
        .def(py::init<>())
        .def_readwrite("g_zero", &CardKeywords::g_zero)
        .def_readwrite("revolution_change", &CardKeywords::revolution_change)
        .def_readwrite("mach_fighter", &CardKeywords::mach_fighter)
        .def_readwrite("speed_attacker", &CardKeywords::speed_attacker)
        .def_readwrite("blocker", &CardKeywords::blocker)
        .def_readwrite("slayer", &CardKeywords::slayer)
        .def_readwrite("double_breaker", &CardKeywords::double_breaker)
        .def_readwrite("triple_breaker", &CardKeywords::triple_breaker)
        .def_readwrite("shield_trigger", &CardKeywords::shield_trigger)
        .def_readwrite("evolution", &CardKeywords::evolution)
        .def_readwrite("cip", &CardKeywords::cip)
        .def_readwrite("at_attack", &CardKeywords::at_attack)
        .def_readwrite("destruction", &CardKeywords::destruction)
        .def_readwrite("just_diver", &CardKeywords::just_diver)
        .def_readwrite("hyper_energy", &CardKeywords::hyper_energy);

    py::class_<FilterDef>(m, "FilterDef")
        .def(py::init<>())
        .def_readwrite("zones", &FilterDef::zones)
        .def_readwrite("civilizations", &FilterDef::civilizations)
        .def_readwrite("races", &FilterDef::races)
        .def_readwrite("min_cost", &FilterDef::min_cost)
        .def_readwrite("max_cost", &FilterDef::max_cost)
        .def_readwrite("min_power", &FilterDef::min_power)
        .def_readwrite("max_power", &FilterDef::max_power)
        .def_readwrite("is_tapped", &FilterDef::is_tapped)
        .def_readwrite("is_blocker", &FilterDef::is_blocker)
        .def_readwrite("is_evolution", &FilterDef::is_evolution)
        .def_readwrite("owner", &FilterDef::owner)
        .def_readwrite("count", &FilterDef::count);

    py::class_<ActionDef>(m, "ActionDef")
        .def(py::init<>())
        .def_readwrite("optional", &ActionDef::optional);

    py::class_<EffectDef>(m, "EffectDef")
        .def(py::init<>())
        .def_readwrite("actions", &EffectDef::actions);

    py::class_<CardDefinition>(m, "CardDefinition")
        .def(py::init<>())
        .def_readwrite("id", &CardDefinition::id)
        .def_readwrite("name", &CardDefinition::name)
        .def_readwrite("cost", &CardDefinition::cost)
        .def_readwrite("power", &CardDefinition::power)
        .def_readwrite("type", &CardDefinition::type)
        .def_readwrite("races", &CardDefinition::races)
        .def_readwrite("keywords", &CardDefinition::keywords)
        .def_readwrite("effects", &CardDefinition::effects)
        .def_readwrite("revolution_change_condition", &CardDefinition::revolution_change_condition)
        .def_readwrite("is_key_card", &CardDefinition::is_key_card)
        .def_readwrite("ai_importance_score", &CardDefinition::ai_importance_score)
        .def_property("civilization",
            [](const CardDefinition& c) { return c.civilizations.empty() ? Civilization::NONE : c.civilizations[0]; },
            [](CardDefinition& c, Civilization civ) { c.civilizations = {civ}; });

    py::class_<CardData>(m, "CardData")
        .def(py::init<CardID, std::string, int, std::string, int, std::string, std::vector<std::string>, std::vector<EffectDef>>());

    py::class_<CardInstance>(m, "CardInstance")
        .def(py::init<>())
        .def_readwrite("instance_id", &CardInstance::instance_id)
        .def_readwrite("card_id", &CardInstance::card_id)
        .def_readwrite("owner", &CardInstance::owner)
        .def_readwrite("is_tapped", &CardInstance::is_tapped)
        .def_readwrite("summoning_sickness", &CardInstance::summoning_sickness)
        .def_readwrite("turn_played", &CardInstance::turn_played)
        .def_readwrite("is_face_down", &CardInstance::is_face_down);

    py::class_<GameState::QueryContext>(m, "QueryContext")
        .def_readwrite("query_id", &GameState::QueryContext::query_id)
        .def_readwrite("query_type", &GameState::QueryContext::query_type)
        .def_readwrite("params", &GameState::QueryContext::params)
        .def_readwrite("valid_targets", &GameState::QueryContext::valid_targets)
        .def_readwrite("options", &GameState::QueryContext::options);

    py::class_<Player>(m, "Player")
        .def_readwrite("hand", &Player::hand)
        .def_readwrite("mana_zone", &Player::mana_zone)
        .def_readwrite("battle_zone", &Player::battle_zone)
        .def_readwrite("shield_zone", &Player::shield_zone)
        .def_readwrite("graveyard", &Player::graveyard)
        .def_readwrite("deck", &Player::deck)
        .def_readwrite("effect_buffer", &Player::effect_buffer);

    py::class_<GameState>(m, "GameState")
        .def(py::init<int>())
        .def("setup_test_duel", &GameState::setup_test_duel)
        .def_readwrite("turn_number", &GameState::turn_number)
        .def_readwrite("active_player_id", &GameState::active_player_id)
        .def_readwrite("current_phase", &GameState::current_phase)
        .def_readwrite("players", &GameState::players)
        .def_readwrite("game_over", &GameState::game_over)
        .def_readwrite("winner", &GameState::winner)
        .def_readwrite("waiting_for_user_input", &GameState::waiting_for_user_input)
        .def_readwrite("pending_query", &GameState::pending_query)
        .def("clone", &GameState::clone)
        .def("get_card_instance", [](GameState& s, int id) { return s.get_card_instance(id); }, py::return_value_policy::reference)
        .def("get_zone", &GameState::get_zone)
        .def("set_deck", [](GameState& s, PlayerID pid, std::vector<int> ids) {
             s.players[pid].deck.clear();
             for (int id : ids) s.players[pid].deck.push_back(CardInstance(id, pid));
        })
        .def("add_test_card_to_battle", [](GameState& s, PlayerID pid, int cid, int iid, bool tapped, bool sick) {
             CardInstance c(cid, pid);
             c.instance_id = iid;
             c.is_tapped = tapped;
             c.summoning_sickness = sick;
             s.players[pid].battle_zone.push_back(c);
             if((size_t)iid >= s.card_owner_map.size()) s.card_owner_map.resize(iid+1);
             s.card_owner_map[iid] = pid;
        })
        .def("add_card_to_hand", [](GameState& s, PlayerID pid, int cid, int iid) {
             CardInstance c(cid, pid);
             c.instance_id = iid;
             s.players[pid].hand.push_back(c);
             if((size_t)iid >= s.card_owner_map.size()) s.card_owner_map.resize(iid+1);
             s.card_owner_map[iid] = pid;
        })
        .def("add_card_to_mana", [](GameState& s, PlayerID pid, int cid, int iid) {
             CardInstance c(cid, pid);
             c.instance_id = iid;
             s.players[pid].mana_zone.push_back(c);
             if((size_t)iid >= s.card_owner_map.size()) s.card_owner_map.resize(iid+1);
             s.card_owner_map[iid] = pid;
        })
        .def("add_card_to_deck", [](GameState& s, PlayerID pid, int cid, int iid) {
             CardInstance c(cid, pid);
             c.instance_id = iid;
             s.players[pid].deck.push_back(c);
             if((size_t)iid >= s.card_owner_map.size()) s.card_owner_map.resize(iid+1);
             s.card_owner_map[iid] = pid;
        });

    py::class_<Action>(m, "Action")
        .def(py::init<>())
        .def_readwrite("type", &Action::type)
        .def_readwrite("source_instance_id", &Action::source_instance_id)
        .def_readwrite("target_instance_id", &Action::target_instance_id)
        .def_readwrite("slot_index", &Action::slot_index)
        .def("to_string", &Action::to_string);

    // Systems
    py::class_<ActionGenerator>(m, "ActionGenerator")
        .def_static("generate_legal_actions", &ActionGenerator::generate_legal_actions);

    py::class_<dm::engine::systems::EffectSystem>(m, "EffectSystem")
        .def_static("instance", &dm::engine::systems::EffectSystem::instance, py::return_value_policy::reference);

    py::class_<EffectResolver>(m, "EffectResolver")
        .def_static("resolve_action", [](GameState& state, const Action& action, const std::map<CardID, CardDefinition>& db){
            EffectResolver::resolve_action(state, action, db);
        })
        .def_static("resume", [](GameState& state, const std::map<CardID, CardDefinition>& db, py::object input_val) {
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
        });

    py::class_<JsonLoader>(m, "JsonLoader")
        .def_static("load_cards", &JsonLoader::load_cards);

    py::class_<PhaseManager>(m, "PhaseManager")
        .def_static("start_game", &PhaseManager::start_game)
        .def_static("next_phase", &PhaseManager::next_phase)
        .def_static("check_game_over", &PhaseManager::check_game_over);

    // AI Components
    py::class_<MCTSNode, std::shared_ptr<MCTSNode>>(m, "MCTSNode")
        .def_readonly("visit_count", &MCTSNode::visit_count)
        .def_readonly("value", &MCTSNode::value_sum)
        .def_readonly("action", &MCTSNode::action)
        .def_readonly("children", &MCTSNode::children);

    py::class_<MCTS>(m, "MCTS")
        .def(py::init<const std::map<CardID, CardDefinition>&, float, float, float, int>())
        .def("search", &MCTS::search)
        .def("get_last_root", &MCTS::get_last_root);

    py::class_<HeuristicEvaluator>(m, "HeuristicEvaluator")
        .def(py::init<const std::map<CardID, CardDefinition>&>())
        .def("evaluate", &HeuristicEvaluator::evaluate);

    py::class_<Determinizer>(m, "Determinizer")
        .def_static("determinize", &Determinizer::determinize);

    py::class_<LethalSolver>(m, "LethalSolver")
        .def_static("is_lethal", &LethalSolver::is_lethal);

    py::class_<ParallelRunner>(m, "ParallelRunner")
        .def(py::init<const std::map<CardID, CardDefinition>&>())
        .def("play_games", &ParallelRunner::play_games)
        .def("play_scenario_match", &ParallelRunner::play_scenario_match)
        .def("play_deck_matchup", &ParallelRunner::play_deck_matchup);

    // Scenarios
    py::class_<ScenarioConfig>(m, "ScenarioConfig")
        .def(py::init<>())
        .def_readwrite("my_mana", &ScenarioConfig::my_mana)
        .def_readwrite("my_hand_cards", &ScenarioConfig::my_hand_cards)
        .def_readwrite("my_battle_zone", &ScenarioConfig::my_battle_zone)
        .def_readwrite("my_mana_zone", &ScenarioConfig::my_mana_zone)
        .def_readwrite("my_grave_yard", &ScenarioConfig::my_grave_yard)
        .def_readwrite("my_shields", &ScenarioConfig::my_shields)
        .def_readwrite("enemy_shield_count", &ScenarioConfig::enemy_shield_count)
        .def_readwrite("enemy_battle_zone", &ScenarioConfig::enemy_battle_zone)
        .def_readwrite("enemy_can_use_trigger", &ScenarioConfig::enemy_can_use_trigger)
        .def_readwrite("loop_proof_mode", &ScenarioConfig::loop_proof_mode);

    py::class_<ScenarioExecutor>(m, "ScenarioExecutor")
        .def(py::init<const std::map<CardID, CardDefinition>&>())
        .def("run_scenario", &ScenarioExecutor::run_scenario);
}
