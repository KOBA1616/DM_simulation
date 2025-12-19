#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "engine/game_instance.hpp"
#include "engine/actions/action_generator.hpp"
// #include "engine/game/effect_resolver.hpp" // Removed
#include "engine/systems/card/card_registry.hpp"
#include "engine/systems/game_logic_system.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/systems/pipeline_executor.hpp"
#include "ai/mcts/mcts.hpp"
#include "ai/evaluator/heuristic_evaluator.hpp" // Typo fix evaluator/evaluators
#include "engine/utils/determinizer.hpp" // Typo fix ai/utils -> engine/utils
#include "core/card_json_types.hpp"
#include "engine/systems/card/json_loader.hpp"
#include "ai/self_play/parallel_runner.hpp" // Typo fix mcts/parallel_runner
#include "engine/systems/flow/phase_manager.hpp" // Typo fix game/phase_manager
#include "core/card_stats.hpp"
#include "ai/solver/lethal_solver.hpp"
#include "engine/systems/card/condition_system.hpp"
#include "engine/systems/trigger_system/trigger_manager.hpp"
#include "engine/systems/mana/mana_system.hpp"
#include "engine/cost_payment_system.hpp" // Typo fix systems/cost... -> engine/cost...
#include "ai/self_play/self_play.hpp" // Added to include GameResultInfo definition
#include "ai/scenario/scenario_executor.hpp" // Missing include

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
        .value("GR_ZONE", Zone::GR_DECK)
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
        .def_property("g_zero", [](const CardKeywords& k) { return k.g_zero; }, [](CardKeywords& k, bool v) { k.g_zero = v; })
        .def_property("revolution_change", [](const CardKeywords& k) { return k.revolution_change; }, [](CardKeywords& k, bool v) { k.revolution_change = v; })
        .def_property("mach_fighter", [](const CardKeywords& k) { return k.mach_fighter; }, [](CardKeywords& k, bool v) { k.mach_fighter = v; })
        .def_property("speed_attacker", [](const CardKeywords& k) { return k.speed_attacker; }, [](CardKeywords& k, bool v) { k.speed_attacker = v; })
        .def_property("blocker", [](const CardKeywords& k) { return k.blocker; }, [](CardKeywords& k, bool v) { k.blocker = v; })
        .def_property("slayer", [](const CardKeywords& k) { return k.slayer; }, [](CardKeywords& k, bool v) { k.slayer = v; })
        .def_property("double_breaker", [](const CardKeywords& k) { return k.double_breaker; }, [](CardKeywords& k, bool v) { k.double_breaker = v; })
        .def_property("triple_breaker", [](const CardKeywords& k) { return k.triple_breaker; }, [](CardKeywords& k, bool v) { k.triple_breaker = v; })
        .def_property("shield_trigger", [](const CardKeywords& k) { return k.shield_trigger; }, [](CardKeywords& k, bool v) { k.shield_trigger = v; })
        .def_property("evolution", [](const CardKeywords& k) { return k.evolution; }, [](CardKeywords& k, bool v) { k.evolution = v; })
        .def_property("cip", [](const CardKeywords& k) { return k.cip; }, [](CardKeywords& k, bool v) { k.cip = v; })
        .def_property("at_attack", [](const CardKeywords& k) { return k.at_attack; }, [](CardKeywords& k, bool v) { k.at_attack = v; })
        .def_property("destruction", [](const CardKeywords& k) { return k.destruction; }, [](CardKeywords& k, bool v) { k.destruction = v; })
        .def_property("just_diver", [](const CardKeywords& k) { return k.just_diver; }, [](CardKeywords& k, bool v) { k.just_diver = v; })
        .def_property("hyper_energy", [](const CardKeywords& k) { return k.hyper_energy; }, [](CardKeywords& k, bool v) { k.hyper_energy = v; });

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
        .def(py::init([](CardID id, std::string name, int cost, std::string civilization, int power, std::string type, std::vector<std::string> races, std::vector<EffectDef> effects) {
            CardData c;
            c.id = id;
            c.name = name;
            c.cost = cost;
            if (civilization == "FIRE") c.civilizations.push_back(Civilization::FIRE);
            else if (civilization == "WATER") c.civilizations.push_back(Civilization::WATER);
            else if (civilization == "NATURE") c.civilizations.push_back(Civilization::NATURE);
            else if (civilization == "LIGHT") c.civilizations.push_back(Civilization::LIGHT);
            else if (civilization == "DARKNESS") c.civilizations.push_back(Civilization::DARKNESS);
            c.power = power;
            c.type = type;
            c.races = races;
            c.effects = effects;
            return c;
        }));

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

    py::class_<dm::engine::EffectSystem, std::unique_ptr<dm::engine::EffectSystem, py::nodelete>>(m, "EffectSystem")
        .def_static("instance", [](){ return &dm::engine::EffectSystem::instance(); }, py::return_value_policy::reference);

    // Bind GameLogicSystem instead of EffectResolver
    py::class_<dm::engine::systems::GameLogicSystem>(m, "EffectResolver") // Keep name for compatibility
        .def_static("resolve_action", [](GameState& state, const Action& action, const std::map<CardID, CardDefinition>& db){
            dm::engine::systems::GameLogicSystem::resolve_action(state, action, db);
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
    // Register shared_ptr holder for GameState if not already implicit (usually good practice for pybind11)
    // But py::class_<GameState> is already defined.
    // If we want to return vector<shared_ptr<GameState>>, we rely on pybind11 smart pointer handling.

    py::class_<MCTSNode, std::shared_ptr<MCTSNode>>(m, "MCTSNode")
        .def_readonly("visit_count", &MCTSNode::visit_count)
        .def_readonly("value", &MCTSNode::value_sum)
        .def_readonly("action", &MCTSNode::action_from_parent)
        .def_property_readonly("children", [](const MCTSNode& node) {
            std::vector<MCTSNode*> children_ptrs;
            for (const auto& child : node.children) {
                children_ptrs.push_back(child.get());
            }
            return children_ptrs;
        }, py::return_value_policy::reference);

    py::class_<MCTS>(m, "MCTS")
        .def(py::init<const std::map<CardID, CardDefinition>&, float, float, float, int, float>())
        .def("search", &MCTS::search)
        .def("get_last_root", &MCTS::get_last_root);

    py::class_<HeuristicEvaluator>(m, "HeuristicEvaluator")
        .def(py::init<const std::map<CardID, CardDefinition>&>())
        .def("evaluate", &HeuristicEvaluator::evaluate);

    py::class_<Determinizer>(m, "Determinizer")
        .def_static("determinize", &Determinizer::determinize);

    py::class_<LethalSolver>(m, "LethalSolver")
        .def_static("is_lethal", &LethalSolver::is_lethal);

    py::class_<GameResultInfo>(m, "GameResultInfo")
        .def_readwrite("result", &GameResultInfo::result)
        .def_readwrite("turn_count", &GameResultInfo::turn_count)
        .def_readwrite("states", &GameResultInfo::states)
        .def_readwrite("policies", &GameResultInfo::policies)
        .def_readwrite("active_players", &GameResultInfo::active_players);

    py::class_<ParallelRunner>(m, "ParallelRunner")
        .def(py::init<const std::map<CardID, CardDefinition>&, int, int>())
        .def("play_games", &ParallelRunner::play_games, py::return_value_policy::move) // Use move policy
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
