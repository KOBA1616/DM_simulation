#include "parallel_runner.hpp"
#include <iostream>
#include <chrono>
#include "../agents/heuristic_agent.hpp"
#include "../../engine/systems/flow/phase_manager.hpp"
#include "../../engine/actions/action_generator.hpp"
#include "../../engine/effects/effect_resolver.hpp"
#include "../../engine/game_instance.hpp" // Added include
#include <omp.h>
#include <random>
#include <algorithm>

namespace dm::ai {

    ParallelRunner::ParallelRunner(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db,
                                   int mcts_simulations,
                                   int batch_size)
        : card_db_(card_db), mcts_simulations_(mcts_simulations), batch_size_(batch_size) {}

    std::vector<GameResultInfo> ParallelRunner::play_games(
        const std::vector<dm::core::GameState>& initial_states,
        BatchEvaluatorCallback evaluator,
        float temperature,
        bool add_noise,
        int num_threads,
        float alpha
    ) {
        std::vector<GameResultInfo> results(initial_states.size());
        InferenceQueue inf_queue;
        std::atomic<int> games_completed = 0;
        int total_games = initial_states.size();

        // Worker Lambda
        auto worker_func = [&](int game_idx) {
            SelfPlay sp(card_db_, mcts_simulations_, batch_size_);
            
            BatchEvaluatorCallback worker_cb = [&](const std::vector<dm::core::GameState>& states) {
                InferenceRequest req;
                req.states = states;
                auto fut = req.promise.get_future();

                {
                    std::lock_guard<std::mutex> lock(inf_queue.mutex);
                    inf_queue.queue.push(&req);
                }
                inf_queue.cv.notify_one();
                return fut.get();
            };

            results[game_idx] = sp.play_game(initial_states[game_idx], worker_cb, temperature, add_noise, alpha);
            games_completed++;
            inf_queue.cv.notify_one();
        };

        std::vector<std::thread> threads;
        std::atomic<int> next_game_idx = 0;

        for (int i = 0; i < num_threads; ++i) {
            threads.emplace_back([&]() {
                while (true) {
                    int idx = next_game_idx.fetch_add(1);
                    if (idx >= total_games) break;
                    worker_func(idx);
                }
            });
        }

        while (games_completed < total_games) {
            std::vector<InferenceRequest*> batch;
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

            std::vector<dm::core::GameState> all_states;
            std::vector<int> split_indices;
            for (auto* req : batch) {
                all_states.insert(all_states.end(), req->states.begin(), req->states.end());
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

        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }

        return results;
    }

    std::vector<int> ParallelRunner::play_scenario_match(
        const dm::core::ScenarioConfig& config,
        int num_games,
        int /*num_threads*/
    ) {
        std::vector<int> results(num_games);

        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < num_games; ++i) {
            std::random_device rd;
            uint32_t seed = rd() + i;

            dm::engine::GameInstance instance(seed, card_db_);
            instance.reset_with_scenario(config);

            HeuristicAgent agent0(0, card_db_);
            HeuristicAgent agent1(1, card_db_);

            int steps = 0;
            int max_steps = 1000;
            dm::core::GameResult final_res = dm::core::GameResult::NONE;

            while (steps < max_steps) {
                 dm::core::GameResult res;
                 if(dm::engine::PhaseManager::check_game_over(instance.state, res)) {
                    final_res = res;
                    break;
                 }

                 auto legal_actions = dm::engine::ActionGenerator::generate_legal_actions(instance.state, card_db_);
                 if (legal_actions.empty()) {
                     dm::engine::PhaseManager::next_phase(instance.state, card_db_);
                     if(dm::engine::PhaseManager::check_game_over(instance.state, res)) {
                         final_res = res;
                         break;
                     }
                     continue;
                 }

                 dm::core::Action action;
                 if (instance.state.active_player_id == 0) {
                     action = agent0.get_action(instance.state, legal_actions);
                 } else {
                     action = agent1.get_action(instance.state, legal_actions);
                 }

                 dm::engine::EffectResolver::resolve_action(instance.state, action, card_db_);
                 steps++;
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
        int /*num_threads*/
    ) {
        std::vector<int> results(num_games);

        #pragma omp parallel for num_threads(num_threads)
        for (int i = 0; i < num_games; ++i) {
            std::random_device rd;
            uint32_t seed = rd() + i;

            dm::engine::GameInstance instance(seed, card_db_);

            int instance_counter = 0;
            auto setup_deck = [&](dm::core::Player& p, const std::vector<dm::core::CardID>& deck_list) {
                p.deck.clear();
                for(auto cid : deck_list) {
                    p.deck.emplace_back(cid, instance_counter++);
                }
                std::mt19937 rng(seed + p.id);
                std::shuffle(p.deck.begin(), p.deck.end(), rng);
            };

            setup_deck(instance.state.players[0], deck1);
            setup_deck(instance.state.players[1], deck2);

            dm::engine::PhaseManager::start_game(instance.state, card_db_);

            HeuristicAgent agent0(0, card_db_);
            HeuristicAgent agent1(1, card_db_);

            int steps = 0;
            int max_steps = 1000;
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

                 auto legal_actions = dm::engine::ActionGenerator::generate_legal_actions(instance.state, card_db_);
                 if (legal_actions.empty()) {
                     dm::engine::PhaseManager::next_phase(instance.state, card_db_);
                     continue;
                 }

                 dm::core::Action action;
                 if (instance.state.active_player_id == 0) {
                     action = agent0.get_action(instance.state, legal_actions);
                 } else {
                     action = agent1.get_action(instance.state, legal_actions);
                 }

                 dm::engine::EffectResolver::resolve_action(instance.state, action, card_db_);
                 steps++;
            }

            if (final_res == dm::core::GameResult::NONE && instance.state.winner != dm::core::GameResult::NONE) {
                final_res = instance.state.winner;
            }

            if (final_res == dm::core::GameResult::P1_WIN) results[i] = 1;
            else if (final_res == dm::core::GameResult::P2_WIN) results[i] = 2;
            else results[i] = 0;
        }

        return results;
    }

}
