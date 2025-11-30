#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "../core/game_state.hpp"
#include "../core/card_def.hpp"
#include "../engine/action_gen/action_generator.hpp"
#include "../engine/effects/effect_resolver.hpp"
#include "../engine/flow/phase_manager.hpp"
#include "../ai/encoders/tensor_converter.hpp"
#include "../ai/encoders/action_encoder.hpp"
#include "../ai/mcts/mcts.hpp"
#include "../ai/evaluator/heuristic_evaluator.hpp"
#include "../ai/self_play/self_play.hpp"
#include "../ai/self_play/parallel_runner.hpp"
#include "../engine/utils/determinizer.hpp"
#include "../utils/csv_loader.hpp"

#include "../ai/self_play/parallel_runner.hpp"
#include "../engine/utils/dev_tools.hpp"

namespace py = pybind11;
using namespace dm::core;
using namespace dm::engine;
using namespace dm::ai;
using namespace dm::utils;

PYBIND11_MODULE(dm_ai_module, m) {
    m.doc() = "Duel Masters AI Simulator Core Module";

    // Enums
    py::enum_<Zone>(m, "Zone")
        .value("DECK", Zone::DECK)
        .value("HAND", Zone::HAND)
        .value("MANA", Zone::MANA)
        .value("BATTLE", Zone::BATTLE)
        .value("GRAVEYARD", Zone::GRAVEYARD)
        .value("SHIELD", Zone::SHIELD)
        .value("HYPER_SPATIAL", Zone::HYPER_SPATIAL)
        .value("GR_DECK", Zone::GR_DECK)
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
        .value("MANA_CHARGE", ActionType::MANA_CHARGE)
        .value("PLAY_CARD", ActionType::PLAY_CARD)
        .value("ATTACK_PLAYER", ActionType::ATTACK_PLAYER)
        .value("ATTACK_CREATURE", ActionType::ATTACK_CREATURE)
        .value("RESOLVE_EFFECT", ActionType::RESOLVE_EFFECT)
        .export_values();

    py::enum_<GameResult>(m, "GameResult")
        .value("NONE", GameResult::NONE)
        .value("P1_WIN", GameResult::P1_WIN)
        .value("P2_WIN", GameResult::P2_WIN)
        .value("DRAW", GameResult::DRAW)
        .export_values();

    py::enum_<Civilization>(m, "Civilization")
        .value("NONE", Civilization::NONE)
        .value("LIGHT", Civilization::LIGHT)
        .value("WATER", Civilization::WATER)
        .value("DARKNESS", Civilization::DARKNESS)
        .value("FIRE", Civilization::FIRE)
        .value("NATURE", Civilization::NATURE)
        .value("ZERO", Civilization::ZERO)
        .export_values();

    py::enum_<CardType>(m, "CardType")
        .value("CREATURE", CardType::CREATURE)
        .value("SPELL", CardType::SPELL)
        .value("EVOLUTION_CREATURE", CardType::EVOLUTION_CREATURE)
        .value("CROSS_GEAR", CardType::CROSS_GEAR)
        .value("CASTLE", CardType::CASTLE)
        .value("PSYCHIC_CREATURE", CardType::PSYCHIC_CREATURE)
        .value("GR_CREATURE", CardType::GR_CREATURE)
        .export_values();

    // Core Structures
    py::class_<CardKeywords>(m, "CardKeywords")
        .def_property_readonly("g_zero", [](const CardKeywords& k) { return k.g_zero; })
        .def_property_readonly("revolution_change", [](const CardKeywords& k) { return k.revolution_change; })
        .def_property_readonly("mach_fighter", [](const CardKeywords& k) { return k.mach_fighter; })
        .def_property_readonly("g_strike", [](const CardKeywords& k) { return k.g_strike; })
        .def_property_readonly("speed_attacker", [](const CardKeywords& k) { return k.speed_attacker; })
        .def_property_readonly("blocker", [](const CardKeywords& k) { return k.blocker; })
        .def_property_readonly("slayer", [](const CardKeywords& k) { return k.slayer; })
        .def_property_readonly("double_breaker", [](const CardKeywords& k) { return k.double_breaker; })
        .def_property_readonly("triple_breaker", [](const CardKeywords& k) { return k.triple_breaker; })
        .def_property_readonly("power_attacker", [](const CardKeywords& k) { return k.power_attacker; })
        .def_property_readonly("shield_trigger", [](const CardKeywords& k) { return k.shield_trigger; })
        .def_property_readonly("evolution", [](const CardKeywords& k) { return k.evolution; })
        .def_property_readonly("cip", [](const CardKeywords& k) { return k.cip; })
        .def_property_readonly("at_attack", [](const CardKeywords& k) { return k.at_attack; })
        .def_property_readonly("at_block", [](const CardKeywords& k) { return k.at_block; })
        .def_property_readonly("at_start_of_turn", [](const CardKeywords& k) { return k.at_start_of_turn; })
        .def_property_readonly("at_end_of_turn", [](const CardKeywords& k) { return k.at_end_of_turn; })
        .def_property_readonly("destruction", [](const CardKeywords& k) { return k.destruction; });

    py::class_<CardDefinition>(m, "CardDefinition")
        .def_readonly("id", &CardDefinition::id)
        .def_readonly("name", &CardDefinition::name)
        .def_readonly("cost", &CardDefinition::cost)
        .def_readonly("power", &CardDefinition::power)
        .def_readonly("power_attacker_bonus", &CardDefinition::power_attacker_bonus)
        .def_readonly("civilization", &CardDefinition::civilization)
        .def_readonly("type", &CardDefinition::type)
        .def_readonly("races", &CardDefinition::races)
        .def_readonly("keywords", &CardDefinition::keywords);

    py::class_<CardInstance>(m, "CardInstance")
        .def(py::init<CardID, int>())
        .def_readonly("card_id", &CardInstance::card_id)
        .def_readonly("instance_id", &CardInstance::instance_id)
        .def_readwrite("is_tapped", &CardInstance::is_tapped)
        .def_readwrite("summoning_sickness", &CardInstance::summoning_sickness);

    py::class_<Player>(m, "Player")
        .def_readonly("id", &Player::id)
        .def_readonly("hand", &Player::hand)
        .def_readonly("deck", &Player::deck)
        .def_readonly("mana_zone", &Player::mana_zone)
        .def_readonly("battle_zone", &Player::battle_zone)
        .def_readonly("shield_zone", &Player::shield_zone)
        .def_readonly("graveyard", &Player::graveyard);

    py::class_<GameState>(m, "GameState")
        .def(py::init<uint32_t>())
        .def("clone", [](const GameState& s) { return GameState(s); })
        .def("setup_test_duel", [](GameState& s) {
            // Add 40 cards to each deck
            for (int i = 0; i < 40; ++i) {
                s.players[0].deck.emplace_back(1, i); // ID 1
                s.players[1].deck.emplace_back(1, i + 100);
            }
        })
        .def("set_deck", [](GameState& s, int player_id, const std::vector<uint16_t>& card_ids) {
            if (player_id < 0 || player_id >= 2) return;
            auto& player = s.players[player_id];
            player.deck.clear();
            int instance_id_counter = player_id * 10000; 
            for (uint16_t cid : card_ids) {
                player.deck.emplace_back(cid, instance_id_counter++);
            }
        })
        .def("add_test_card_to_battle", [](GameState& s, int player_id, int card_id, int instance_id, bool tapped, bool sick) {
             if (player_id < 0 || player_id >= 2) return;
             CardInstance c(card_id, instance_id);
             c.is_tapped = tapped;
             c.summoning_sickness = sick;
             s.players[player_id].battle_zone.push_back(c);
        })
        .def_readwrite("turn_number", &GameState::turn_number)
        .def_readwrite("active_player_id", &GameState::active_player_id)
        .def_readwrite("current_phase", &GameState::current_phase)
        .def_readonly("players", &GameState::players);

    py::class_<Action>(m, "Action")
        .def(py::init<>())
        .def_readwrite("type", &Action::type)
        .def_readwrite("card_id", &Action::card_id)
        .def_readwrite("source_instance_id", &Action::source_instance_id)
        .def_readwrite("target_instance_id", &Action::target_instance_id)
        .def_readwrite("target_player", &Action::target_player)
        .def_readwrite("slot_index", &Action::slot_index)
        .def_readwrite("target_slot_index", &Action::target_slot_index)
        .def("to_string", &Action::to_string);

    // Engine Classes
    py::class_<PhaseManager>(m, "PhaseManager")
        .def_static("start_game", &PhaseManager::start_game)
        .def_static("next_phase", &PhaseManager::next_phase)
        .def_static("fast_forward", &PhaseManager::fast_forward)
        .def_static("check_game_over", [](GameState& gs) {
            GameResult res;
            bool over = PhaseManager::check_game_over(gs, res);
            return std::make_pair(over, (int)res);
        });

    py::class_<ActionGenerator>(m, "ActionGenerator")
        .def_static("generate_legal_actions", &ActionGenerator::generate_legal_actions);

    py::class_<EffectResolver>(m, "EffectResolver")
        .def_static("resolve_action", &EffectResolver::resolve_action);

    // Utils
    py::class_<Determinizer>(m, "Determinizer")
        .def_static("determinize", &Determinizer::determinize);

    py::class_<CsvLoader>(m, "CsvLoader")
        .def_static("load_cards", &CsvLoader::load_cards);

    py::class_<DevTools>(m, "DevTools")
        .def_static("move_cards", &DevTools::move_cards,
            py::arg("state"), py::arg("player_id"), py::arg("source"), py::arg("target"), py::arg("count"), py::arg("card_id_filter") = -1);

    // AI
    py::class_<TensorConverter>(m, "TensorConverter")
        .def_readonly_static("INPUT_SIZE", &TensorConverter::INPUT_SIZE)
        .def_static("convert_to_tensor", &TensorConverter::convert_to_tensor)
        .def_static("convert_batch_flat", &TensorConverter::convert_batch_flat);

    py::class_<ActionEncoder>(m, "ActionEncoder")
        .def_readonly_static("TOTAL_ACTION_SIZE", &ActionEncoder::TOTAL_ACTION_SIZE)
        .def_static("action_to_index", &ActionEncoder::action_to_index);

    py::class_<HeuristicEvaluator>(m, "HeuristicEvaluator")
        .def(py::init<const std::map<CardID, CardDefinition>&>())
        .def("evaluate", &HeuristicEvaluator::evaluate);

    py::class_<MCTSNode>(m, "MCTSNode")
        .def_readonly("visit_count", &MCTSNode::visit_count)
        .def_readonly("value_sum", &MCTSNode::value_sum)
        .def_readonly("prior", &MCTSNode::prior)
        .def_property_readonly("value", &MCTSNode::value)
        .def_property_readonly("action", [](const MCTSNode& n) { return n.action_from_parent; })
        .def_property_readonly("children", [](const MCTSNode& n) {
            std::vector<const MCTSNode*> result;
            for (const auto& child : n.children) {
                result.push_back(child.get());
            }
            return result;
        }, py::return_value_policy::reference);

    py::class_<MCTS>(m, "MCTS")
        .def(py::init<const std::map<CardID, CardDefinition>&, float, float, float, int>(),
             py::arg("card_db"), py::arg("c_puct") = 1.0f, py::arg("dirichlet_alpha") = 0.3f, py::arg("dirichlet_epsilon") = 0.25f, py::arg("batch_size") = 1)
        .def("search", &MCTS::search, py::call_guard<py::gil_scoped_release>(),
             py::arg("root_state"), py::arg("simulations"), py::arg("evaluator"), py::arg("add_noise") = false, py::arg("temperature") = 1.0f)
        .def("search_with_heuristic", [](MCTS& self, const GameState& root, int sims, HeuristicEvaluator& evaluator, bool noise, float temp) {
            return self.search(root, sims, [&](const std::vector<GameState>& states){
                return evaluator.evaluate(states);
            }, noise, temp);
        }, py::arg("root_state"), py::arg("simulations"), py::arg("evaluator"), py::arg("add_noise") = false, py::arg("temperature") = 1.0f)
        .def("get_last_root", &MCTS::get_last_root, py::return_value_policy::reference);

    py::class_<GameResultInfo>(m, "GameResultInfo")
        .def_readonly("result", &GameResultInfo::result)
        .def_readonly("turn_count", &GameResultInfo::turn_count)
        .def_readonly("states", &GameResultInfo::states)
        .def_readonly("policies", &GameResultInfo::policies)
        .def_readonly("active_players", &GameResultInfo::active_players);

    py::class_<SelfPlay>(m, "SelfPlay")
        .def(py::init<const std::map<CardID, CardDefinition>&, int, int>(),
             py::arg("card_db"), py::arg("mcts_simulations") = 50, py::arg("batch_size") = 1)
        .def("play_game", [](SelfPlay& self, GameState initial_state, std::function<std::pair<std::vector<std::vector<float>>, std::vector<float>>(const std::vector<GameState>&)> evaluator, float temp, bool noise) {
            return self.play_game(initial_state, evaluator, temp, noise);
        }, py::call_guard<py::gil_scoped_release>(),
           py::arg("initial_state"), py::arg("evaluator"), py::arg("temperature") = 1.0f, py::arg("add_noise") = true);

    py::class_<ParallelRunner>(m, "ParallelRunner")
        .def(py::init<const std::map<CardID, CardDefinition>&, int, int>(),
             py::arg("card_db"), py::arg("mcts_simulations") = 50, py::arg("batch_size") = 1)
        .def("play_games", [](ParallelRunner& self, const std::vector<GameState>& initial_states, std::function<std::pair<std::vector<std::vector<float>>, std::vector<float>>(const std::vector<GameState>&)> evaluator, float temp, bool noise, int num_threads) {
            return self.play_games(initial_states, evaluator, temp, noise, num_threads);
        }, py::call_guard<py::gil_scoped_release>(),
           py::arg("initial_states"), py::arg("evaluator"), py::arg("temperature") = 1.0f, py::arg("add_noise") = true, py::arg("num_threads") = 4);
}
