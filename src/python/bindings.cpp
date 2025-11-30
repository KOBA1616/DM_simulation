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
#include "../engine/utils/determinizer.hpp"
#include "../utils/csv_loader.hpp"

namespace py = pybind11;
using namespace dm::core;
using namespace dm::engine;
using namespace dm::ai;
using namespace dm::utils;

PYBIND11_MODULE(dm_ai_module, m) {
    m.doc() = "Duel Masters AI Simulator Core Module";

    // Enums
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

    py::enum_<Civilization>(m, "Civilization")
        .value("NONE", Civilization::NONE)
        .value("LIGHT", Civilization::LIGHT)
        .value("WATER", Civilization::WATER)
        .value("DARKNESS", Civilization::DARKNESS)
        .value("FIRE", Civilization::FIRE)
        .value("NATURE", Civilization::NATURE)
        .value("ZERO", Civilization::ZERO)
        .export_values();

    // Core Structures
    py::class_<CardDefinition>(m, "CardDefinition")
        .def_readonly("id", &CardDefinition::id)
        .def_readonly("name", &CardDefinition::name)
        .def_readonly("cost", &CardDefinition::cost)
        .def_readonly("power", &CardDefinition::power)
        .def_readonly("civilization", &CardDefinition::civilization);

    py::class_<CardInstance>(m, "CardInstance")
        .def_readonly("card_id", &CardInstance::card_id)
        .def_readonly("instance_id", &CardInstance::instance_id)
        .def_readonly("is_tapped", &CardInstance::is_tapped);

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
        .def_readwrite("turn_number", &GameState::turn_number)
        .def_readwrite("active_player_id", &GameState::active_player_id)
        .def_readwrite("current_phase", &GameState::current_phase)
        .def_readonly("players", &GameState::players);

    py::class_<Action>(m, "Action")
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

    // AI
    py::class_<TensorConverter>(m, "TensorConverter")
        .def_readonly_static("INPUT_SIZE", &TensorConverter::INPUT_SIZE)
        .def_static("convert_to_tensor", &TensorConverter::convert_to_tensor);

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
        }, py::call_guard<py::gil_scoped_release>(),
           py::arg("root_state"), py::arg("simulations"), py::arg("evaluator"), py::arg("add_noise") = false, py::arg("temperature") = 1.0f)
        .def("get_last_root", &MCTS::get_last_root, py::return_value_policy::reference);
}
