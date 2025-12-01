#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include "../core/game_state.hpp"
#include "../core/card_def.hpp"
#include "../engine/action_gen/action_generator.hpp"
#include "../engine/effects/effect_resolver.hpp"
#include "../engine/flow/phase_manager.hpp"
#include "../engine/card_system/card_registry.hpp"
#include "../ai/encoders/tensor_converter.hpp"
#include "../ai/encoders/action_encoder.hpp"
#include "../ai/mcts/mcts.hpp"
#include "../ai/evaluator/heuristic_evaluator.hpp"
#include "../ai/self_play/self_play.hpp"
#include "../ai/self_play/parallel_runner.hpp"
#include "../engine/utils/determinizer.hpp"
#include "../utils/csv_loader.hpp"

#include "../ai/pomdp/pomdp.hpp"
#include "../ai/pomdp/parametric_belief.hpp"

#include "../ai/self_play/parallel_runner.hpp"
#include "../engine/utils/dev_tools.hpp"
#include "../python/python_batch_inference.hpp"
#include "../ai/evaluator/neural_evaluator.hpp"

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
        .value("BLOCK", ActionType::BLOCK)
        .value("USE_SHIELD_TRIGGER", ActionType::USE_SHIELD_TRIGGER)
        .value("SELECT_TARGET", ActionType::SELECT_TARGET)
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

    // Expose stats/POMDP helpers as module-level helpers (wrappers)
    m.def("initialize_card_stats", [](GameState &s, const std::map<CardID, CardDefinition> &card_db, int deck_size) {
        for (const auto &p : card_db) {
            CardID cid = p.first;
            if (s.global_card_stats.find(cid) == s.global_card_stats.end()) {
                s.global_card_stats.emplace(cid, CardStats{});
            }
        }
        s.initial_deck_count = deck_size;
        s.visible_card_count = 0;
        s.visible_stats_sum = CardStats{};
        s.initial_deck_stats_sum = CardStats{};
    }, py::arg("state"), py::arg("card_db"), py::arg("deck_size") = 40);

    m.def("vectorize_card_stats", [](const GameState &s, CardID cid){
        auto it = s.global_card_stats.find(cid);
        if (it != s.global_card_stats.end()) return it->second.to_vector();
        return std::vector<float>(16, 0.0f);
    }, py::arg("state"), py::arg("card_id"));

    m.def("get_library_potential", [](const GameState &s){
        int remaining = s.initial_deck_count - s.visible_card_count;
        if (remaining <= 0) return std::vector<float>(16, 0.0f);
        std::vector<float> potential(16, 0.0f);
        potential[0] = static_cast<float>((s.initial_deck_stats_sum.sum_early_usage - s.visible_stats_sum.sum_early_usage) / remaining);
        potential[1] = static_cast<float>((s.initial_deck_stats_sum.sum_late_usage - s.visible_stats_sum.sum_late_usage) / remaining);
        potential[2] = static_cast<float>((s.initial_deck_stats_sum.sum_trigger_rate - s.visible_stats_sum.sum_trigger_rate) / remaining);
        potential[3] = static_cast<float>((s.initial_deck_stats_sum.sum_cost_discount - s.visible_stats_sum.sum_cost_discount) / remaining);

        potential[4] = static_cast<float>((s.initial_deck_stats_sum.sum_hand_adv - s.visible_stats_sum.sum_hand_adv) / remaining);
        potential[5] = static_cast<float>((s.initial_deck_stats_sum.sum_board_adv - s.visible_stats_sum.sum_board_adv) / remaining);
        potential[6] = static_cast<float>((s.initial_deck_stats_sum.sum_mana_adv - s.visible_stats_sum.sum_mana_adv) / remaining);
        potential[7] = static_cast<float>((s.initial_deck_stats_sum.sum_shield_dmg - s.visible_stats_sum.sum_shield_dmg) / remaining);

        potential[8] = static_cast<float>((s.initial_deck_stats_sum.sum_hand_var - s.visible_stats_sum.sum_hand_var) / remaining);
        potential[9] = static_cast<float>((s.initial_deck_stats_sum.sum_board_var - s.visible_stats_sum.sum_board_var) / remaining);
        potential[10] = static_cast<float>((s.initial_deck_stats_sum.sum_survival_rate - s.visible_stats_sum.sum_survival_rate) / remaining);
        potential[11] = static_cast<float>((s.initial_deck_stats_sum.sum_effect_death - s.visible_stats_sum.sum_effect_death) / remaining);

        potential[12] = static_cast<float>((s.initial_deck_stats_sum.sum_win_contribution - s.visible_stats_sum.sum_win_contribution) / remaining);
        potential[13] = static_cast<float>((s.initial_deck_stats_sum.sum_comeback_win - s.visible_stats_sum.sum_comeback_win) / remaining);
        potential[14] = static_cast<float>((s.initial_deck_stats_sum.sum_finish_blow - s.visible_stats_sum.sum_finish_blow) / remaining);
        potential[15] = static_cast<float>((s.initial_deck_stats_sum.sum_deck_consumption - s.visible_stats_sum.sum_deck_consumption) / remaining);

        return potential;
    }, py::arg("state"));

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

    // Debug helper to inspect pending effects (type, source_instance_id, controller)
    m.def("get_pending_effects_info", [](const GameState& s) {
        std::vector<std::tuple<int,int,int>> out;
        for (const auto &pe : s.pending_effects) {
            out.emplace_back((int)pe.type, pe.source_instance_id, (int)pe.controller);
        }
        return out;
    });

    // Verbose debug helper: include num_targets_needed, selected targets count, has_effect_def
    m.def("get_pending_effects_verbose", [](const GameState& s) {
        std::vector<std::tuple<int,int,int,int,int,bool>> out;
        for (const auto &pe : s.pending_effects) {
            out.emplace_back((int)pe.type, pe.source_instance_id, (int)pe.controller, pe.num_targets_needed, (int)pe.target_instance_ids.size(), (bool)pe.effect_def.has_value());
        }
        return out;
    });

    // Utils
    py::class_<Determinizer>(m, "Determinizer")
        .def_static("determinize", &Determinizer::determinize);

    py::class_<CsvLoader>(m, "CsvLoader")
        .def_static("load_cards", &CsvLoader::load_cards);

    py::class_<DevTools>(m, "DevTools")
        .def_static("move_cards", &DevTools::move_cards,
            py::arg("state"), py::arg("player_id"), py::arg("source"), py::arg("target"), py::arg("count"), py::arg("card_id_filter") = -1);

    // CardRegistry JSON loader (for GenericCardSystem)
    m.def("card_registry_load_from_json", &dm::engine::CardRegistry::load_from_json, "Load card definitions from a JSON string into the CardRegistry");

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

        // Batch inference registration: allow Python to register a batched model callback
        m.def("register_batch_inference", [](py::function func) {
            dm::python::BatchCallback cb = [func](const dm::python::BatchInput& in) -> dm::python::BatchOutput {
                py::gil_scoped_acquire acquire;
                py::list py_in;
                for (size_t i = 0; i < in.size(); ++i) {
                    py::list row;
                    for (size_t j = 0; j < in[i].size(); ++j) row.append(in[i][j]);
                    py_in.append(row);
                }

                py::object result = func(py_in);
                // Expect (policies, values)
                py::tuple tup = result.cast<py::tuple>();
                py::list py_policies = tup[0].cast<py::list>();
                py::list py_values = tup[1].cast<py::list>();

                dm::python::BatchOutput out;
                out.first.reserve(py_policies.size());
                out.second.reserve(py_values.size());

                for (auto item : py_policies) {
                    py::list plist = item.cast<py::list>();
                    std::vector<float> pv;
                    pv.reserve(plist.size());
                    for (auto v : plist) pv.push_back(v.cast<float>());
                    out.first.push_back(std::move(pv));
                }

                for (auto v : py_values) out.second.push_back(v.cast<float>());
                return out;
            };

            dm::python::set_batch_callback(cb);
        }, "Register a Python function for batch inference. The function should accept a list of feature-lists and return (policies_list, values_list)");

        // Register a Python function that accepts a NumPy ndarray of shape (batch, stride).
        // This tries to minimize per-row allocations by passing a single contiguous buffer.
        m.def("register_batch_inference_numpy", [](py::function func) {
            dm::python::FlatBatchCallback cb = [func](const std::vector<float>& flat, size_t n, size_t stride) -> dm::python::BatchOutput {
                py::gil_scoped_acquire acquire;

                // Make a shared copy of flat to ensure lifetime when NumPy views it.
                auto data_ptr = std::make_shared<std::vector<float>>(flat);
                // Allocate a heap-owned shared_ptr wrapper for the capsule
                auto capsule_owner = new std::shared_ptr<std::vector<float>>(data_ptr);
                py::capsule base_capsule(capsule_owner, [](void *v){
                    auto p = reinterpret_cast<std::shared_ptr<std::vector<float>>*>(v);
                    delete p;
                });

                // Create numpy array that references the shared vector, using capsule as base/owner
                py::array_t<float> arr({(py::ssize_t)n, (py::ssize_t)stride}, {(py::ssize_t)(stride * sizeof(float)), (py::ssize_t)sizeof(float)}, data_ptr->data(), base_capsule);

#ifdef AI_DEBUG
                fprintf(stderr, "bindings: about to call Python func with ndarray shape=(%zu,%zu)\n", n, stride);
#endif
                py::object result;
                try {
                    result = func(arr);
                } catch (py::error_already_set &e) {
                    fprintf(stderr, "bindings: Python callback raised: %s\n", e.what());
                    throw;
                }
                // Debug: print result type and repr to stderr
                try {
                    std::string rtype = std::string(py::str(result.get_type()));
                    std::string rrepr = std::string(py::str(result));
#ifdef AI_DEBUG
                    fprintf(stderr, "bindings: result type=%s repr=%s\n", rtype.c_str(), rrepr.c_str());
#endif
                } catch (...) {}

                // Expect (policies, values)
                dm::python::BatchOutput out;

                if (py::isinstance<py::tuple>(result)) {
                    py::tuple tup = result.cast<py::tuple>();
                    py::object py_policies = tup[0];
                    py::object py_values = tup[1];

                    // Policies: accept ndarray or list
                    if (py::isinstance<py::array>(py_policies)) {
                        py::array p_arr = py_policies.cast<py::array>();
                        py::buffer_info info = p_arr.request();
                        if (info.ndim == 2) {
                            py::ssize_t rows = info.shape[0];
                            py::ssize_t cols = info.shape[1];
                            out.first.resize((size_t)rows);

                            // Try fast paths based on dtype
                            std::string fmt = info.format;
                            if (fmt == py::format_descriptor<float>::format()) {
                                float* base = static_cast<float*>(info.ptr);
                                for (py::ssize_t i = 0; i < rows; ++i) {
                                    out.first[(size_t)i].assign(base + i*cols, base + i*cols + cols);
                                }
                            } else if (fmt == py::format_descriptor<double>::format()) {
                                double* base = static_cast<double*>(info.ptr);
                                for (py::ssize_t i = 0; i < rows; ++i) {
                                    auto &dst = out.first[(size_t)i];
                                    dst.resize((size_t)cols);
                                    for (py::ssize_t j = 0; j < cols; ++j) dst[(size_t)j] = static_cast<float>(base[i*cols + j]);
                                }
                            } else {
                                // Fallback: element-wise conversion (handles non-contiguous or other dtypes)
                                for (py::ssize_t i = 0; i < rows; ++i) {
                                    py::object row_obj = p_arr[py::int_(i)];
                                    py::iterable row_iter = row_obj.cast<py::iterable>();
                                    auto &dst = out.first[(size_t)i];
                                    for (auto item : row_iter) dst.push_back(item.cast<float>());
                                }
                            }
                        }
                    } else if (py::isinstance<py::list>(py_policies)) {
                        py::list plist = py_policies.cast<py::list>();
                        out.first.reserve(plist.size());
                        for (auto item : plist) {
                            py::list row = item.cast<py::list>();
                            std::vector<float> pv;
                            pv.reserve(row.size());
                            for (auto v : row) pv.push_back(v.cast<float>());
                            out.first.push_back(std::move(pv));
                        }
                    }

                    // Values: accept ndarray or list
                    if (py::isinstance<py::array>(py_values)) {
                        py::array v_arr = py_values.cast<py::array>();
                        py::buffer_info info = v_arr.request();
                        if (info.ndim == 1) {
                            py::ssize_t len = info.shape[0];
                            std::string fmt = info.format;
                            if (fmt == py::format_descriptor<float>::format()) {
                                float* base = static_cast<float*>(info.ptr);
                                out.second.assign(base, base + len);
                            } else if (fmt == py::format_descriptor<double>::format()) {
                                double* base = static_cast<double*>(info.ptr);
                                out.second.resize((size_t)len);
                                for (py::ssize_t i = 0; i < len; ++i) out.second[(size_t)i] = static_cast<float>(base[i]);
                            } else {
                                // fallback
                                out.second.reserve((size_t)len);
                                for (py::ssize_t i = 0; i < len; ++i) out.second.push_back(v_arr[py::int_(i)].cast<float>());
                            }
                        }
                    } else if (py::isinstance<py::list>(py_values)) {
                        py::list vlist = py_values.cast<py::list>();
                        out.second.reserve(vlist.size());
                        for (auto vv : vlist) out.second.push_back(vv.cast<float>());
                    }
                }

                // If conversion produced empty policies (unexpected), try a fallback: call Python func with list-of-lists
                bool need_fallback = false;
                if (out.first.empty()) need_fallback = true;
                else if (!out.first.empty() && out.first[0].empty()) need_fallback = true;
                if (need_fallback) {
                    fprintf(stderr, "bindings: fallback - calling Python func with list-of-lists\n");
                    py::list py_in;
                    for (size_t i = 0; i < n; ++i) {
                        py::list row;
                        for (size_t j = 0; j < stride; ++j) row.append((*data_ptr)[i*stride + j]);
                        py_in.append(row);
                    }
                    py::object result2 = func(py_in);
                    if (py::isinstance<py::tuple>(result2)) {
                        py::tuple tup2 = result2.cast<py::tuple>();
                        py::object py_policies2 = tup2[0];
                        py::object py_values2 = tup2[1];
                        // Reuse existing conversion logic by temporarily assigning
                        // Handle policies
                        out.first.clear();
                        if (py::isinstance<py::array>(py_policies2)) {
                            py::array p_arr = py_policies2.cast<py::array>();
                            py::buffer_info info = p_arr.request();
                            if (info.ndim == 2) {
                                py::ssize_t rows = info.shape[0];
                                py::ssize_t cols = info.shape[1];
                                out.first.resize((size_t)rows);
                                std::string fmt = info.format;
                                if (fmt == py::format_descriptor<float>::format()) {
                                    float* base = static_cast<float*>(info.ptr);
                                    for (py::ssize_t i = 0; i < rows; ++i) out.first[(size_t)i].assign(base + i*cols, base + i*cols + cols);
                                } else if (fmt == py::format_descriptor<double>::format()) {
                                    double* base = static_cast<double*>(info.ptr);
                                    for (py::ssize_t i = 0; i < rows; ++i) {
                                        auto &dst = out.first[(size_t)i]; dst.resize((size_t)cols);
                                        for (py::ssize_t j = 0; j < cols; ++j) dst[(size_t)j] = static_cast<float>(base[i*cols + j]);
                                    }
                                } else {
                                    for (py::ssize_t i = 0; i < rows; ++i) {
                                        py::object row_obj = p_arr[py::int_(i)];
                                        py::iterable row_iter = row_obj.cast<py::iterable>();
                                        auto &dst = out.first[(size_t)i];
                                        for (auto item : row_iter) dst.push_back(item.cast<float>());
                                    }
                                }
                            }
                        } else if (py::isinstance<py::list>(py_policies2)) {
                            py::list plist = py_policies2.cast<py::list>();
                            out.first.reserve(plist.size());
                            for (auto item : plist) {
                                py::list row = item.cast<py::list>();
                                std::vector<float> pv;
                                pv.reserve(row.size());
                                for (auto v : row) pv.push_back(v.cast<float>());
                                out.first.push_back(std::move(pv));
                            }
                        }

                        // Values
                        out.second.clear();
                        if (py::isinstance<py::array>(py_values2)) {
                            py::array v_arr = py_values2.cast<py::array>();
                            py::buffer_info info = v_arr.request();
                            if (info.ndim == 1) {
                                py::ssize_t len = info.shape[0];
                                std::string fmt = info.format;
                                if (fmt == py::format_descriptor<float>::format()) {
                                    float* base = static_cast<float*>(info.ptr);
                                    out.second.assign(base, base + len);
                                } else if (fmt == py::format_descriptor<double>::format()) {
                                    double* base = static_cast<double*>(info.ptr);
                                    out.second.resize((size_t)len);
                                    for (py::ssize_t i = 0; i < len; ++i) out.second[(size_t)i] = static_cast<float>(base[i]);
                                } else {
                                    for (py::ssize_t i = 0; i < info.shape[0]; ++i) out.second.push_back(v_arr[py::int_(i)].cast<float>());
                                }
                            }
                        } else if (py::isinstance<py::list>(py_values2)) {
                            py::list vlist = py_values2.cast<py::list>();
                            out.second.reserve(vlist.size());
                            for (auto vv : vlist) out.second.push_back(vv.cast<float>());
                        }
                    }
                }

                return out;
            };

            dm::python::set_flat_batch_callback(cb);
        }, "Register a Python function that accepts a NumPy ndarray of shape (batch, stride) and returns (policies, values)");

        m.def("has_batch_inference_registered", []() {
            return dm::python::has_batch_callback();
        });

        m.def("clear_batch_inference", []() {
            dm::python::clear_batch_callback();
        });

        m.def("clear_batch_inference_numpy", []() {
            dm::python::clear_flat_batch_callback();
        });

        m.def("has_flat_batch_inference_registered", []() {
            return dm::python::has_flat_batch_callback();
        });

        py::class_<NeuralEvaluator>(m, "NeuralEvaluator")
            .def(py::init<const std::map<CardID, CardDefinition>&>())
            .def("evaluate", &NeuralEvaluator::evaluate);

        // POMDP Inference (Requirement 11) - lightweight stub exposed to Python
        py::class_<dm::ai::POMDPInference>(m, "POMDPInference")
            .def(py::init<>())
            .def("initialize", &dm::ai::POMDPInference::initialize, py::arg("card_db"))
            .def("update_belief", &dm::ai::POMDPInference::update_belief, py::arg("state"))
            .def("infer_action", &dm::ai::POMDPInference::infer_action, py::arg("state"))
            .def("get_belief_vector", &dm::ai::POMDPInference::get_belief_vector);

        // Parametric belief model (simple per-card probabilities)
        py::class_<dm::ai::ParametricBelief>(m, "ParametricBelief")
            .def(py::init<>())
            .def("initialize", &dm::ai::ParametricBelief::initialize, py::arg("card_db"))
            .def("initialize_ids", &dm::ai::ParametricBelief::initialize_ids, py::arg("ids"))
            .def("update", &dm::ai::ParametricBelief::update, py::arg("state"))
            .def("get_vector", &dm::ai::ParametricBelief::get_vector)
            .def("set_weights", [](dm::ai::ParametricBelief &b, float strong_w, float deck_w){ b.set_weights(strong_w, deck_w); }, py::arg("strong_weight")=1.0f, py::arg("deck_weight")=0.25f)
            .def("get_weights", [](const dm::ai::ParametricBelief &b){ auto p = b.get_weights(); return py::make_tuple(p.first, p.second); });
}
