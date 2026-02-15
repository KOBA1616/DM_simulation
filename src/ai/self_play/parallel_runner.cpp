#include "parallel_runner.hpp"
#include <iostream>
#include <chrono>
#include "ai/agents/heuristic_agent.hpp"
#include "engine/systems/flow/phase_manager.hpp"
#include "engine/systems/card/card_registry.hpp"
#include "engine/actions/intent_generator.hpp"
#include "engine/systems/director/game_logic_system.hpp"
#include "engine/game_instance.hpp"
#include "ai/mcts/mcts.hpp"
#include "ai/inference/pimc_generator.hpp"
#include <omp.h>
#include <random>
#include <algorithm>
#include <fstream>

namespace dm::ai {

    using namespace dm::engine::systems;

    ParallelRunner::ParallelRunner(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db,
                                   int mcts_simulations,
                                   int batch_size)
        : card_db_(std::make_shared<std::map<dm::core::CardID, dm::core::CardDefinition>>(card_db)),
          mcts_simulations_(mcts_simulations), batch_size_(batch_size) {}

    ParallelRunner::ParallelRunner(std::shared_ptr<const std::map<dm::core::CardID, dm::core::CardDefinition>> card_db,
                                   int mcts_simulations,
                                   int batch_size)
        : card_db_(card_db), mcts_simulations_(mcts_simulations), batch_size_(batch_size) {}

    ParallelRunner::ParallelRunner(int mcts_simulations, int batch_size)
        : card_db_(dm::engine::CardRegistry::get_all_definitions_ptr()), mcts_simulations_(mcts_simulations), batch_size_(batch_size) {}

    ParallelRunner::~ParallelRunner() {
        {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            stop_pool_ = true;
        }
        pool_cv_.notify_all();
        for (auto& t : pool_) {
            if (t.joinable()) t.join();
        }
    }

    void ParallelRunner::enable_pimc(bool enabled) {
        pimc_enabled_ = enabled;
        if (pimc_enabled_) {
            if (!pimc_generator_) {
                pimc_generator_ = std::make_shared<dm::ai::inference::PimcGenerator>(card_db_);
            }
            if (deck_inference_) {
                pimc_generator_->set_inference_model(deck_inference_);
            }
        }
    }

    void ParallelRunner::load_meta_decks(const std::string& json_path) {
        if (!deck_inference_) {
            deck_inference_ = std::make_shared<dm::ai::inference::DeckInference>();
        }
        deck_inference_->load_decks(json_path);
        if (pimc_generator_) {
            pimc_generator_->set_inference_model(deck_inference_);
        }
    }

    void ParallelRunner::ensure_thread_pool(int num_threads) {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        if ((int)pool_.size() >= num_threads && !pool_.empty()) return;

        int current_size = pool_.size();
        for (int i = current_size; i < num_threads; ++i) {
            pool_.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lk(pool_mutex_);
                        pool_cv_.wait(lk, [this] {
                            return stop_pool_ || !tasks_.empty();
                        });

                        if (stop_pool_ && tasks_.empty()) return;

                        if (!tasks_.empty()) {
                            task = std::move(tasks_.front());
                            tasks_.pop();
                        }
                    }
                    if (task) {
                        try {
                            task();
                        } catch (...) {
                        }
                    }
                }
            });
        }
    }

    void ParallelRunner::submit_task(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(pool_mutex_);
            tasks_.push(std::move(task));
        }
        pool_cv_.notify_one();
    }

    std::vector<GameResultInfo> ParallelRunner::play_games(
        const std::vector<std::shared_ptr<dm::core::GameState>>& initial_states,
        BatchEvaluatorCallback evaluator,
        float temperature,
        bool add_noise,
        int num_threads,
        float alpha,
        bool collect_data
    ) {
        std::vector<GameResultInfo> results(initial_states.size());
        InferenceQueue inf_queue;
        std::atomic<int> games_completed = 0;
        int total_games = initial_states.size();

        // Prepare PIMC Generator (shared)
        if (pimc_enabled_ && !pimc_generator_) {
            pimc_generator_ = std::make_shared<dm::ai::inference::PimcGenerator>(card_db_);
            if (deck_inference_) {
                pimc_generator_->set_inference_model(deck_inference_);
            }
        }

        auto worker_func = [&](int game_idx) {
            try {
                SelfPlay sp(card_db_, mcts_simulations_, batch_size_);

                if (pimc_enabled_) {
                    sp.set_pimc_generator(pimc_generator_);
                }

                BatchEvaluatorCallback worker_cb = [&](const std::vector<std::shared_ptr<dm::core::GameState>>& states) {
                    auto req = std::make_shared<InferenceRequest>();
                    req->states.reserve(states.size());
                    for (const auto& s : states) {
                        req->states.push_back(s);
                    }

                    auto fut = req->promise.get_future();

                    {
                        std::lock_guard<std::mutex> lock(inf_queue.mutex);
                        inf_queue.queue.push(req);
                    }
                    inf_queue.cv.notify_one();
                    return fut.get();
                };

                results[game_idx] = sp.play_game(*initial_states[game_idx], worker_cb, temperature, add_noise, alpha, collect_data);
            } catch (...) {
            }
            games_completed++;
            inf_queue.cv.notify_one();
        };

        ensure_thread_pool(num_threads);

        std::atomic<int> next_game_idx = 0;
        std::atomic<bool> stop_loop = false;

        std::vector<std::future<void>> futures;
        for (int i = 0; i < num_threads; ++i) {
            auto task = std::make_shared<std::packaged_task<void()>>([&]() {
                while (!stop_loop) {
                    int idx = next_game_idx.fetch_add(1);
                    if (idx >= total_games) break;
                    worker_func(idx);
                }
            });

            futures.push_back(task->get_future());

            submit_task([task]() {
                (*task)();
            });
        }

        try {
            while (games_completed < total_games) {
                std::vector<std::shared_ptr<InferenceRequest>> batch;
                {
                    std::unique_lock<std::mutex> lock(inf_queue.mutex);
                    inf_queue.cv.wait_for(lock, std::chrono::milliseconds(1), [&] {
                        return !inf_queue.queue.empty() || games_completed == total_games;
                    });

                    if (games_completed == total_games && inf_queue.queue.empty()) break;

                    while (!inf_queue.queue.empty()) {
                        batch.push_back(inf_queue.queue.front());
                        inf_queue.queue.pop();
                        if (batch.size() >= 32) break;
                    }
                }

                if (batch.empty()) continue;

                std::vector<std::shared_ptr<dm::core::GameState>> all_states;
                std::vector<int> split_indices;
                for (auto& req : batch) {
                    for (auto& s : req->states) {
                        all_states.push_back(s);
                    }
                    split_indices.push_back(req->states.size());
                }

                auto result_pair = evaluator(all_states);
                const auto& all_policies = result_pair.first;
                const auto& all_values = result_pair.second;

                int offset = 0;
                for (size_t i = 0; i < batch.size(); ++i) {
                    int count = split_indices[i];
                    std::vector<std::vector<float>> policies;
                    std::vector<float> values;
                    for (int j = 0; j < count; ++j) {
                        policies.push_back(all_policies[offset + j]);
                        values.push_back(all_values[offset + j]);
                    }
                    offset += count;
                    batch[i]->promise.set_value({policies, values});
                }
            }
        } catch (...) {
            stop_loop = true;
            inf_queue.cv.notify_all();
        }

        stop_loop = true;
        inf_queue.cv.notify_all();

        for (auto& fut : futures) {
            fut.wait();
        }

        return results;
    }

    std::vector<float> ParallelRunner::run_pimc_search(
            const dm::core::GameState& observation,
            dm::core::PlayerID observer_id,
            const std::vector<dm::core::CardID>& opponent_deck_candidates,
            BatchEvaluatorCallback evaluator,
            int num_threads,
            float temperature
    ) {
        int num_worlds = num_threads;
        std::vector<std::vector<float>> policies(num_worlds);

        InferenceQueue inf_queue;
        std::atomic<int> completed_worlds = 0;
        std::atomic<bool> stop_threads = false;

        auto worker_func = [&](int world_idx) {
            if (stop_threads) return;
            try {
                uint32_t seed = std::random_device{}();
                dm::core::GameState determinized_state = dm::ai::inference::PimcGenerator::generate_determinized_state(
                    observation, *card_db_, observer_id, opponent_deck_candidates, seed
                );

                MCTS mcts(card_db_, 1.0f, 0.3f, 0.25f, batch_size_, 0.0f);

                BatchEvaluatorCallback worker_cb = [&](const std::vector<std::shared_ptr<dm::core::GameState>>& states) {
                    auto req = std::make_shared<InferenceRequest>();
                    req->states.reserve(states.size());
                    for (const auto& s : states) {
                        req->states.push_back(s);
                    }
                    auto fut = req->promise.get_future();

                    {
                        std::lock_guard<std::mutex> lock(inf_queue.mutex);
                        inf_queue.queue.push(req);
                    }
                    inf_queue.cv.notify_one();
                    return fut.get();
                };

                policies[world_idx] = mcts.search(determinized_state, mcts_simulations_, worker_cb, true, temperature);
            } catch (...) { }

            completed_worlds++;
            inf_queue.cv.notify_one();
        };

        ensure_thread_pool(num_threads);

        std::vector<std::future<void>> futures;
        for (int i = 0; i < num_worlds; ++i) {
            auto task = std::make_shared<std::packaged_task<void()>>([&, i]() {
                worker_func(i);
            });
            futures.push_back(task->get_future());
            submit_task([task]() { (*task)(); });
        }

        try {
            while (completed_worlds < num_worlds) {
                std::vector<std::shared_ptr<InferenceRequest>> batch;
                {
                    std::unique_lock<std::mutex> lock(inf_queue.mutex);
                    inf_queue.cv.wait_for(lock, std::chrono::milliseconds(1), [&] {
                        return !inf_queue.queue.empty() || completed_worlds == num_worlds;
                    });

                    if (completed_worlds == num_worlds && inf_queue.queue.empty()) break;

                    while (!inf_queue.queue.empty()) {
                        batch.push_back(inf_queue.queue.front());
                        inf_queue.queue.pop();
                        if (batch.size() >= 32) break;
                    }
                }

                if (batch.empty()) continue;

                std::vector<std::shared_ptr<dm::core::GameState>> all_states;
                std::vector<int> split_indices;
                for (auto& req : batch) {
                    for (auto& s : req->states) {
                        all_states.push_back(s);
                    }
                    split_indices.push_back(req->states.size());
                }

                auto result_pair = evaluator(all_states);
                const auto& all_policies = result_pair.first;
                const auto& all_values = result_pair.second;

                int offset = 0;
                for (size_t i = 0; i < batch.size(); ++i) {
                    int count = split_indices[i];
                    std::vector<std::vector<float>> batch_policies;
                    std::vector<float> batch_values;
                    for (int j = 0; j < count; ++j) {
                        batch_policies.push_back(all_policies[offset + j]);
                        batch_values.push_back(all_values[offset + j]);
                    }
                    offset += count;
                    batch[i]->promise.set_value({batch_policies, batch_values});
                }
            }
        } catch (...) {
            stop_threads = true;
            inf_queue.cv.notify_all();
        }

        stop_threads = true;
        inf_queue.cv.notify_all();

        for (auto& fut : futures) {
            fut.wait();
        }

        if (policies.empty()) return {};

        std::vector<float> aggregated_policy(policies[0].size(), 0.0f);
        for (const auto& p : policies) {
            if (p.empty()) continue;
            for (size_t i = 0; i < p.size(); ++i) {
                if (i < aggregated_policy.size()) aggregated_policy[i] += p[i];
            }
        }

        float sum = 0.0f;
        for (float v : aggregated_policy) sum += v;
        if (sum > 1e-6f) {
            for (auto& v : aggregated_policy) v /= sum;
        }

        return aggregated_policy;
    }

    std::vector<int> ParallelRunner::play_scenario_match(
        const dm::core::ScenarioConfig& config,
        int num_games,
        int num_threads
    ) {
        (void)num_threads;
        std::vector<int> results(num_games);

        #pragma omp parallel for
        for (int i = 0; i < num_games; ++i) {
            std::random_device rd;
            uint32_t seed = rd() + i;

            dm::engine::GameInstance instance(seed, card_db_);
            instance.reset_with_scenario(config);

            HeuristicAgent agent0(0, *card_db_);
            HeuristicAgent agent1(1, *card_db_);

            int steps = 0;
            int max_steps = 10000;
            dm::core::GameResult final_res = dm::core::GameResult::NONE;

            while (steps < max_steps) {
                 dm::core::GameResult res;
                 if(dm::engine::PhaseManager::check_game_over(instance.state, res)) {
                    final_res = res;
                    break;
                 }

                 auto legal_actions = dm::engine::IntentGenerator::generate_legal_commands(instance.state, *card_db_);
                 if (legal_actions.empty()) {
                     dm::engine::PhaseManager::next_phase(instance.state, *card_db_);
                     if(dm::engine::PhaseManager::check_game_over(instance.state, res)) {
                         final_res = res;
                         break;
                     }
                     continue;
                 }

                 dm::core::CommandDef action;
                 if (instance.state.active_player_id == 0) {
                     action = agent0.get_action(instance.state, legal_actions);
                 } else {
                     action = agent1.get_action(instance.state, legal_actions);
                 }

                 GameLogicSystem::resolve_command_oneshot(instance.state, action, *card_db_);
                 steps++;
            }

            // Ensure finalization to capture winner if engine has pending resolution
            if (final_res == dm::core::GameResult::NONE) {
                dm::core::GameResult tmp = final_res;
                if (dm::engine::PhaseManager::check_game_over(instance.state, tmp)) {
                    final_res = tmp;
                }
                // Try finalization hook if available on state
                try {
                    instance.state.on_game_finished(final_res);
                } catch(...) {
                    // Ignore if not present
                }
                // Update from state if set
                if (instance.state.winner != dm::core::GameResult::NONE) {
                    final_res = instance.state.winner;
                }
            }

            // Per-game JSONL logging for diagnostics (append)
            try {
                std::ofstream ofs("logs/runner_debug.jsonl", std::ios::app);
                if (ofs) {
                    ofs << "{\"game\":" << i << ",\"seed\":" << seed << ",\"winner\":";
                    if (final_res == dm::core::GameResult::P1_WIN) ofs << 1;
                    else if (final_res == dm::core::GameResult::P2_WIN) ofs << 2;
                    else ofs << 0;
                    ofs << ",\"turn\":" << instance.state.turn_number;
                    ofs << ",\"p1_hand\":" << instance.state.players[0].hand.size();
                    ofs << ",\"p2_hand\":" << instance.state.players[1].hand.size();
                    ofs << "}\n";
                    ofs.close();
                }
            } catch(...) {
                // ignore logging failures
            }

            if (final_res == dm::core::GameResult::P1_WIN) results[i] = 1;
            else if (final_res == dm::core::GameResult::P2_WIN) results[i] = 2;
            else results[i] = 0;
        }

        return results;
    }

    std::vector<int> ParallelRunner::play_deck_matchup(
        const std::vector<dm::core::CardID>& deck1,
        const std::vector<dm::core::CardID>& deck2,
        int num_games,
        int num_threads
    ) {
        (void)num_threads;
        std::vector<int> results(num_games);

        #pragma omp parallel for
        for (int i = 0; i < num_games; ++i) {
            std::random_device rd;
            uint32_t seed = rd() + i;

            dm::engine::GameInstance instance(seed, card_db_);

            int instance_counter = 0;
            auto setup_deck = [&](dm::core::Player& p, const std::vector<dm::core::CardID>& deck_list) {
                p.deck.clear();
                for(auto cid : deck_list) {
                    p.deck.emplace_back(cid, instance_counter++, p.id);
                }
                std::mt19937 rng(seed + p.id);
                std::shuffle(p.deck.begin(), p.deck.end(), rng);
            };

            setup_deck(instance.state.players[0], deck1);
            setup_deck(instance.state.players[1], deck2);

            // Initialize card_owner_map logic
            instance.state.card_owner_map.assign(instance_counter, 0); // Resize and init
            for (const auto& card : instance.state.players[0].deck) {
                if (card.instance_id >= 0 && card.instance_id < instance_counter)
                    instance.state.card_owner_map[card.instance_id] = 0;
            }
            for (const auto& card : instance.state.players[1].deck) {
                if (card.instance_id >= 0 && card.instance_id < instance_counter)
                    instance.state.card_owner_map[card.instance_id] = 1;
            }

            dm::engine::PhaseManager::start_game(instance.state, *card_db_);

            HeuristicAgent agent0(0, *card_db_);
            HeuristicAgent agent1(1, *card_db_);

            int steps = 0;
            int max_steps = 10000;
            dm::core::GameResult final_res = dm::core::GameResult::NONE;

            while (steps < max_steps) {
                 if(instance.state.winner != dm::core::GameResult::NONE) {
                     final_res = instance.state.winner;
                     break;
                 }

                 dm::core::GameResult res;
                 if(dm::engine::PhaseManager::check_game_over(instance.state, res)) {
                    final_res = res;
                    break;
                 }

                 auto legal_actions = dm::engine::IntentGenerator::generate_legal_commands(instance.state, *card_db_);
                 if (legal_actions.empty()) {
                     dm::engine::PhaseManager::next_phase(instance.state, *card_db_);
                     continue;
                 }

                 dm::core::CommandDef action;
                 if (instance.state.active_player_id == 0) {
                     action = agent0.get_action(instance.state, legal_actions);
                 } else {
                     action = agent1.get_action(instance.state, legal_actions);
                 }

                 GameLogicSystem::resolve_command_oneshot(instance.state, action, *card_db_);
                 steps++;
            }

            if (final_res == dm::core::GameResult::NONE && instance.state.winner != dm::core::GameResult::NONE) {
                final_res = instance.state.winner;
            }

                // Try additional finalization if still none
                if (final_res == dm::core::GameResult::NONE) {
                    dm::core::GameResult tmp = final_res;
                    if (dm::engine::PhaseManager::check_game_over(instance.state, tmp)) {
                        final_res = tmp;
                    }
                    try {
                        instance.state.on_game_finished(final_res);
                    } catch(...) { }
                    if (instance.state.winner != dm::core::GameResult::NONE) final_res = instance.state.winner;
                }

                // Per-game JSONL logging for diagnostics (append)
                try {
                    std::ofstream ofs("logs/runner_debug.jsonl", std::ios::app);
                    if (ofs) {
                        ofs << "{\"game\":" << i << ",\"seed\":" << seed << ",\"winner\":";
                        if (final_res == dm::core::GameResult::P1_WIN) ofs << 1;
                        else if (final_res == dm::core::GameResult::P2_WIN) ofs << 2;
                        else ofs << 0;
                        ofs << ",\"turn\":" << instance.state.turn_number;
                        ofs << ",\"p1_hand\":" << instance.state.players[0].hand.size();
                        ofs << ",\"p2_hand\":" << instance.state.players[1].hand.size();
                        ofs << "}\n";
                        ofs.close();
                    }
                } catch(...) { }

                if (final_res == dm::core::GameResult::P1_WIN) results[i] = 1;
                else if (final_res == dm::core::GameResult::P2_WIN) results[i] = 2;
                else results[i] = 0;
        }

        return results;
    }

    std::map<dm::core::CardID, dm::core::CardStats> ParallelRunner::play_deck_matchup_with_stats(
            const std::vector<dm::core::CardID>& deck1,
            const std::vector<dm::core::CardID>& deck2,
            int num_games,
            int num_threads
    ) {
        (void)num_threads;
        std::map<dm::core::CardID, dm::core::CardStats> aggregated_stats;
        std::mutex stats_mutex;

        #pragma omp parallel for
        for (int i = 0; i < num_games; ++i) {
            std::random_device rd;
            uint32_t seed = rd() + i;

            dm::engine::GameInstance instance(seed, card_db_);

            int instance_counter = 0;
            auto setup_deck = [&](dm::core::Player& p, const std::vector<dm::core::CardID>& deck_list) {
                p.deck.clear();
                for(auto cid : deck_list) {
                    p.deck.emplace_back(cid, instance_counter++, p.id);
                }
                std::mt19937 rng(seed + p.id);
                std::shuffle(p.deck.begin(), p.deck.end(), rng);
            };

            setup_deck(instance.state.players[0], deck1);
            setup_deck(instance.state.players[1], deck2);

            // Initialize card_owner_map logic
            instance.state.card_owner_map.assign(instance_counter, 0); // Resize and init
            for (const auto& card : instance.state.players[0].deck) {
                if (card.instance_id >= 0 && card.instance_id < instance_counter)
                    instance.state.card_owner_map[card.instance_id] = 0;
            }
            for (const auto& card : instance.state.players[1].deck) {
                if (card.instance_id >= 0 && card.instance_id < instance_counter)
                    instance.state.card_owner_map[card.instance_id] = 1;
            }

            dm::engine::PhaseManager::start_game(instance.state, *card_db_);

            HeuristicAgent agent0(0, *card_db_);
            HeuristicAgent agent1(1, *card_db_);

            int steps = 0;
            int max_steps = 1000;
            dm::core::GameResult final_res = dm::core::GameResult::NONE;

            while (steps < max_steps) {
                 if(instance.state.winner != dm::core::GameResult::NONE) break;

                 dm::core::GameResult res;
                 if(dm::engine::PhaseManager::check_game_over(instance.state, res)) break;

                 auto legal_actions = dm::engine::IntentGenerator::generate_legal_commands(instance.state, *card_db_);
                 if (legal_actions.empty()) {
                     dm::engine::PhaseManager::next_phase(instance.state, *card_db_);
                     continue;
                 }

                 dm::core::CommandDef action;
                 if (instance.state.active_player_id == 0) {
                     action = agent0.get_action(instance.state, legal_actions);
                 } else {
                     action = agent1.get_action(instance.state, legal_actions);
                 }

                 GameLogicSystem::resolve_command_oneshot(instance.state, action, *card_db_);
                 steps++;
            }

            // Trigger stats finalization (e.g. tracking mana usage)
            dm::core::GameResult final_res_val = instance.state.winner;
            if (final_res_val == dm::core::GameResult::NONE) {
                 dm::engine::PhaseManager::check_game_over(instance.state, final_res_val);
            }
            // Explicitly call on_game_finished to ensure mana usage and win stats are recorded
            instance.state.on_game_finished(final_res_val);

            // Merge stats into aggregated_stats
            {
                std::lock_guard<std::mutex> lock(stats_mutex);
                for (const auto& [cid, stats] : instance.state.global_card_stats) {
                    auto& agg = aggregated_stats[cid];
                    agg.play_count += stats.play_count;
                    agg.win_count += stats.win_count;
                    agg.sum_cost_discount += stats.sum_cost_discount;
                    agg.sum_early_usage += stats.sum_early_usage;
                    agg.sum_late_usage += stats.sum_late_usage;
                    agg.mana_usage_count += stats.mana_usage_count;
                    agg.sum_win_contribution += stats.sum_win_contribution;

                    // Added new fields merging
                    agg.shield_trigger_count += stats.shield_trigger_count;
                    agg.hand_play_count += stats.hand_play_count;
                }
            }
        }

        return aggregated_stats;
    }

}
