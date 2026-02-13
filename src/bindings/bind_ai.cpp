#include "bindings/bindings.hpp"
#include "bindings/types.hpp"
#include "ai/mcts/mcts_decision_maker.hpp"
#include <pybind11/stl_bind.h>
#include "ai/mcts/mcts.hpp"
#include "ai/evaluator/heuristic_evaluator.hpp"
#include "ai/evaluator/beam_search_evaluator.hpp"
#include "ai/agents/heuristic_agent.hpp"
#if defined(USE_LIBTORCH) || defined(USE_ONNXRUNTIME)
#include "ai/evaluator/neural_evaluator.hpp"
#endif
#include "engine/utils/determinizer.hpp"
#include "ai/self_play/parallel_runner.hpp"
#include "ai/solver/lethal_solver.hpp"
#include "ai/scenario/scenario_executor.hpp"
#include "ai/encoders/token_converter.hpp"
#include "ai/encoders/tensor_converter.hpp"
#include "ai/encoders/action_encoder.hpp"
#include "ai/inference/deck_inference.hpp"
#include "ai/inference/pimc_generator.hpp"
#include "ai/pomdp/pomdp.hpp"
#include "ai/pomdp/parametric_belief.hpp"
#include "ai/data_collection/data_collector.hpp"
#include "bindings/python_batch_inference.hpp"
#include "ai/evolution/deck_evolution.hpp"
#include "ai/evolution/meta_environment.hpp"
#include "ai/neural_net/self_attention.hpp"
#include <pybind11/stl.h>
#include <pybind11/functional.h>

using namespace dm;
using namespace dm::core;
using namespace dm::ai;
using namespace dm::engine;

void bind_ai(py::module& m) {
    // Phase 8: TokenConverter
    py::class_<dm::ai::encoders::TokenConverter>(m, "TokenConverter")
        .def_static("encode_state", &dm::ai::encoders::TokenConverter::encode_state,
            py::arg("state"), py::arg("perspective") = 0, py::arg("max_len") = 0)
        .def_static("get_vocab_size", &dm::ai::encoders::TokenConverter::get_vocab_size);

    py::class_<dm::ai::TensorConverter>(m, "TensorConverter")
        .def_readonly_static("INPUT_SIZE", &dm::ai::TensorConverter::INPUT_SIZE)
        .def_readonly_static("VOCAB_SIZE", &dm::ai::TensorConverter::VOCAB_SIZE)
        .def_readonly_static("MAX_SEQ_LEN", &dm::ai::TensorConverter::MAX_SEQ_LEN)
        .def_static("convert_to_tensor", &dm::ai::TensorConverter::convert_to_tensor,
             py::arg("game_state"), py::arg("player_view"), py::arg("card_db"), py::arg("mask_opponent_hand") = true)
        .def_static("convert_batch_flat", static_cast<std::vector<float> (*)(const std::vector<std::shared_ptr<dm::core::GameState>>&, const std::map<dm::core::CardID, dm::core::CardDefinition>&, bool)>(&dm::ai::TensorConverter::convert_batch_flat),
             py::arg("states"), py::arg("card_db"), py::arg("mask_opponent_hand") = true)
        .def_static("convert_to_sequence", &dm::ai::TensorConverter::convert_to_sequence,
             py::arg("game_state"), py::arg("player_view"), py::arg("card_db"), py::arg("mask_opponent_hand") = true)
        // Helper wrapper to accept vector of pointers for batch sequence
        // Needed because GameState is not copyable, preventing direct std::vector<GameState> binding
        .def_static("convert_batch_sequence", [](const std::vector<std::shared_ptr<dm::core::GameState>>& states, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, bool mask_opponent_hand) {
            std::vector<long> batch_seq;
            batch_seq.reserve(states.size() * dm::ai::TensorConverter::MAX_SEQ_LEN);
            for (const auto& state_ptr : states) {
                if (state_ptr) {
                    std::vector<long> s = dm::ai::TensorConverter::convert_to_sequence(*state_ptr, state_ptr->active_player_id, card_db, mask_opponent_hand);
                    batch_seq.insert(batch_seq.end(), s.begin(), s.end());
                } else {
                    // Pad if null to maintain batch alignment
                    for(int i=0; i<dm::ai::TensorConverter::MAX_SEQ_LEN; ++i) batch_seq.push_back(0);
                }
            }
            return batch_seq;
        }, py::arg("states"), py::arg("card_db"), py::arg("mask_opponent_hand") = true);

    py::class_<ActionEncoder>(m, "ActionEncoder")
        .def_readonly_static("TOTAL_ACTION_SIZE", &ActionEncoder::TOTAL_ACTION_SIZE)
        .def_static("action_to_index", &ActionEncoder::action_to_index);

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

    py::class_<dm::ai::mcts::MCTSDecisionMaker, dm::engine::systems::DecisionMaker>(m, "MCTSDecisionMaker")
        .def(py::init<int>(), py::arg("simulations") = 50);

    py::class_<MCTS>(m, "MCTS")
        .def(py::init<const std::map<CardID, CardDefinition>&, float, float, float, int, float>())
        .def("set_pimc_generator", &MCTS::set_pimc_generator)
        .def("search", &MCTS::search)
        .def("get_last_root", &MCTS::get_last_root);

    py::class_<HeuristicEvaluator>(m, "HeuristicEvaluator")
        .def(py::init<const std::map<CardID, CardDefinition>&>())
        .def("evaluate", &HeuristicEvaluator::evaluate);

    py::class_<HeuristicAgent>(m, "HeuristicAgent")
        .def(py::init<int, const std::map<CardID, CardDefinition>&>())
        .def("get_action", &HeuristicAgent::get_action);

    py::class_<BeamSearchEvaluator>(m, "BeamSearchEvaluator")
        // Primary efficient constructor using CardRegistry
        .def(py::init<int, int>(), py::arg("beam_width")=7, py::arg("max_depth")=3)
        // Legacy/Copy constructor support
        .def(py::init([](const std::map<CardID, CardDefinition>& card_db, int beam_width, int max_depth) {
            // Helper to copy the map into a shared_ptr for the C++ class
            auto shared_db = std::make_shared<std::map<CardID, CardDefinition>>(card_db);
            return std::make_unique<BeamSearchEvaluator>(shared_db, beam_width, max_depth);
        }), py::arg("card_db"), py::arg("beam_width")=7, py::arg("max_depth")=3)
        .def("evaluate", &BeamSearchEvaluator::evaluate);

#if defined(USE_LIBTORCH) || defined(USE_ONNXRUNTIME)
    py::enum_<ModelType>(m, "ModelType")
        .value("RESNET", ModelType::RESNET)
        .value("TRANSFORMER", ModelType::TRANSFORMER)
        .export_values();

    py::class_<NeuralEvaluator>(m, "NeuralEvaluator")
        .def(py::init<const std::map<CardID, CardDefinition>&>())
        .def("load_model", &NeuralEvaluator::load_model)
        .def("set_model_type", &NeuralEvaluator::set_model_type)
        .def("evaluate", &NeuralEvaluator::evaluate);
#endif

    // Batch Inference Callbacks
    m.def("set_batch_callback", &dm::python::set_batch_callback);
    m.def("has_batch_callback", &dm::python::has_batch_callback);
    m.def("clear_batch_callback", &dm::python::clear_batch_callback);

    m.def("set_flat_batch_callback", &dm::python::set_flat_batch_callback);
    m.def("has_flat_batch_callback", &dm::python::has_flat_batch_callback);
    m.def("clear_flat_batch_callback", &dm::python::clear_flat_batch_callback);

    m.def("set_sequence_batch_callback", &dm::python::set_sequence_batch_callback);
    m.def("has_sequence_batch_callback", &dm::python::has_sequence_batch_callback);
    m.def("clear_sequence_batch_callback", &dm::python::clear_sequence_batch_callback);

    py::class_<Determinizer>(m, "Determinizer")
        .def_static("determinize", &Determinizer::determinize);

    py::class_<LethalSolver>(m, "LethalSolver")
        .def_static("is_lethal", &LethalSolver::is_lethal);

    py::class_<DeckEvolutionConfig>(m, "DeckEvolutionConfig")
        .def(py::init<>())
        .def_readwrite("target_deck_size", &DeckEvolutionConfig::target_deck_size)
        .def_readwrite("mutation_rate", &DeckEvolutionConfig::mutation_rate)
        .def_readwrite("crossover_rate", &DeckEvolutionConfig::crossover_rate)
        .def_readwrite("synergy_weight", &DeckEvolutionConfig::synergy_weight)
        .def_readwrite("curve_weight", &DeckEvolutionConfig::curve_weight);

    py::class_<DeckEvolution>(m, "DeckEvolution")
        .def(py::init<const std::map<dm::core::CardID, dm::core::CardDefinition>&>())
        .def("evolve_deck", &DeckEvolution::evolve_deck)
        .def("crossover_decks", &DeckEvolution::crossover_decks)
        .def("calculate_interaction_score", &DeckEvolution::calculate_interaction_score)
        .def("get_candidates_by_civ", &DeckEvolution::get_candidates_by_civ);

    py::class_<DeckAgent>(m, "DeckAgent")
        .def_readwrite("id", &DeckAgent::id)
        .def_readwrite("deck", &DeckAgent::deck)
        .def_readwrite("elo_rating", &DeckAgent::elo_rating)
        .def_readwrite("matches_played", &DeckAgent::matches_played)
        .def_readwrite("wins", &DeckAgent::wins)
        .def_readwrite("generation", &DeckAgent::generation)
        .def_readwrite("archetype", &DeckAgent::archetype);

    py::class_<MetaEnvironment>(m, "MetaEnvironment")
        .def(py::init<const std::map<dm::core::CardID, dm::core::CardDefinition>&>())
        .def("initialize_population", &MetaEnvironment::initialize_population)
        .def("record_match", &MetaEnvironment::record_match)
        .def("step_generation", &MetaEnvironment::step_generation)
        .def("get_population", &MetaEnvironment::get_population)
        .def("get_agent", &MetaEnvironment::get_agent);

    py::class_<dm::ai::inference::DeckInference, std::shared_ptr<dm::ai::inference::DeckInference>>(m, "DeckInference")
        .def(py::init<>())
        .def("load_decks", &dm::ai::inference::DeckInference::load_decks)
        .def("infer_probabilities", &dm::ai::inference::DeckInference::infer_probabilities)
        .def("sample_hidden_cards", &dm::ai::inference::DeckInference::sample_hidden_cards);

    py::class_<dm::ai::inference::PimcGenerator, std::shared_ptr<dm::ai::inference::PimcGenerator>>(m, "PimcGenerator")
        .def(py::init([](const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
            return std::make_unique<dm::ai::inference::PimcGenerator>(std::make_shared<std::map<dm::core::CardID, dm::core::CardDefinition>>(card_db));
        }))
        .def("set_inference_model", &dm::ai::inference::PimcGenerator::set_inference_model)
        .def("generate_determinized_state", static_cast<dm::core::GameState (dm::ai::inference::PimcGenerator::*)(const dm::core::GameState&, dm::core::PlayerID, uint32_t)>(&dm::ai::inference::PimcGenerator::generate_determinized_state))
        .def_static("generate_determinized_state_static", static_cast<dm::core::GameState (*)(const dm::core::GameState&, const std::map<dm::core::CardID, dm::core::CardDefinition>&, dm::core::PlayerID, const std::vector<dm::core::CardID>&, uint32_t)>(&dm::ai::inference::PimcGenerator::generate_determinized_state));

    py::class_<dm::ai::POMDPInference>(m, "POMDPInference")
        .def(py::init<>())
        .def("initialize", &dm::ai::POMDPInference::initialize)
        .def("update_belief", &dm::ai::POMDPInference::update_belief)
        .def("sample_state", &dm::ai::POMDPInference::sample_state)
        .def("get_deck_probabilities", &dm::ai::POMDPInference::get_deck_probabilities);

    py::class_<GameResultInfo>(m, "GameResultInfo")
        .def_readwrite("result", &GameResultInfo::result)
        .def_readwrite("turn_count", &GameResultInfo::turn_count)
        .def_readwrite("states", &GameResultInfo::states)
        .def_readwrite("policies", &GameResultInfo::policies)
        .def_readwrite("active_players", &GameResultInfo::active_players);

    py::class_<ParallelRunner>(m, "ParallelRunner")
        // Constructor accepting shared_ptr to ensure memory safety
        .def(py::init([](std::shared_ptr<const std::map<CardID, CardDefinition>> card_db, int mcts_simulations, int batch_size) {
             return std::make_unique<ParallelRunner>(card_db, mcts_simulations, batch_size);
        }), py::arg("card_db"), py::arg("mcts_simulations"), py::arg("batch_size"))
        // Legacy/Copy constructor support (less safe but existing code might rely on it implicitly via conversions)
        .def(py::init<const std::map<CardID, CardDefinition>&, int, int>())
        .def(py::init<int, int>())
        .def("enable_pimc", &ParallelRunner::enable_pimc)
        .def("load_meta_decks", &ParallelRunner::load_meta_decks)
        .def("play_games", &ParallelRunner::play_games, py::return_value_policy::move)
        
#if defined(USE_LIBTORCH) || defined(USE_ONNXRUNTIME)
        .def("play_games", [](ParallelRunner& self,
                              const std::vector<std::shared_ptr<dm::core::GameState>>& initial_states,
                              NeuralEvaluator& evaluator,
                              float temperature,
                              bool add_noise,
                              int num_threads,
                              float alpha,
                              bool collect_data) {
            return self.play_games(initial_states,
                [&evaluator](const std::vector<std::shared_ptr<dm::core::GameState>>& states) {
                    return evaluator.evaluate(states);
                },
                temperature, add_noise, num_threads, alpha, collect_data);
        }, py::arg("initial_states"), py::arg("evaluator"), py::arg("temperature")=1.0f, py::arg("add_noise")=true, py::arg("num_threads")=4, py::arg("alpha")=0.0f, py::arg("collect_data")=true, py::return_value_policy::move)
#endif
        .def("play_scenario_match", &ParallelRunner::play_scenario_match)
        .def("play_deck_matchup", &ParallelRunner::play_deck_matchup)
        .def("play_deck_matchup_with_stats", &ParallelRunner::play_deck_matchup_with_stats);

    py::class_<ScenarioConfig>(m, "ScenarioConfig")
        .def(py::init<>())
        .def_readwrite("my_mana", &ScenarioConfig::my_mana)
        .def_readwrite("my_hand_cards", &ScenarioConfig::my_hand_cards)
        .def_readwrite("my_battle_zone", &ScenarioConfig::my_battle_zone)
        .def_readwrite("my_mana_zone", &ScenarioConfig::my_mana_zone)
        .def_readwrite("my_grave_yard", &ScenarioConfig::my_grave_yard)
        .def_readwrite("my_shields", &ScenarioConfig::my_shields)
        .def_readwrite("my_deck", &ScenarioConfig::my_deck)
        .def_readwrite("enemy_hand_cards", &ScenarioConfig::enemy_hand_cards) // Added
        .def_readwrite("enemy_shield_count", &ScenarioConfig::enemy_shield_count)
        .def_readwrite("enemy_deck", &ScenarioConfig::enemy_deck)
        .def_readwrite("enemy_battle_zone", &ScenarioConfig::enemy_battle_zone)
        .def_readwrite("enemy_can_use_trigger", &ScenarioConfig::enemy_can_use_trigger)
        .def_readwrite("loop_proof_mode", &ScenarioConfig::loop_proof_mode);

    py::class_<ScenarioExecutor>(m, "ScenarioExecutor")
        .def(py::init<std::shared_ptr<const CardDatabase>>())
        .def(py::init<>())
        .def("run_scenario", &ScenarioExecutor::run_scenario);

    py::class_<ParametricBelief>(m, "ParametricBelief")
        .def(py::init<>())
        .def("set_weights", &ParametricBelief::set_weights)
        .def("initialize", &ParametricBelief::initialize)
        .def("update", &ParametricBelief::update)
        .def("get_vector", &ParametricBelief::get_vector);

    py::class_<CollectedBatch>(m, "CollectedBatch")
        .def(py::init<>())
        .def_readwrite("token_states", &CollectedBatch::token_states)
        .def_readwrite("tensor_states", &CollectedBatch::tensor_states)
        .def_readwrite("policies", &CollectedBatch::policies)
        .def_readwrite("masks", &CollectedBatch::masks)
        .def_readwrite("values", &CollectedBatch::values);

    py::class_<DataCollector>(m, "DataCollector")
        .def(py::init([](std::shared_ptr<const CardDatabase> db) {
            return std::make_unique<DataCollector>(db);
        }))
        .def(py::init<>())
        .def("collect_data_batch_heuristic", &DataCollector::collect_data_batch_heuristic,
             py::arg("episodes"), py::arg("collect_tokens") = false, py::arg("collect_tensors") = true);

    py::class_<dm::ai::neural_net::Tensor2D>(m, "Tensor2D")
        .def(py::init<int, int>())
        .def_readwrite("data", &dm::ai::neural_net::Tensor2D::data)
        .def_readwrite("rows", &dm::ai::neural_net::Tensor2D::rows)
        .def_readwrite("cols", &dm::ai::neural_net::Tensor2D::cols);

    py::class_<dm::ai::neural_net::SelfAttention>(m, "SelfAttention")
        .def(py::init<int, int>())
        .def("forward", &dm::ai::neural_net::SelfAttention::forward)
        .def("initialize_weights", &dm::ai::neural_net::SelfAttention::initialize_weights);
}
