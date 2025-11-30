#include "parallel_runner.hpp"
#include <iostream>
#include <chrono>

// We need to include pybind11 to handle GIL if we are doing it manually?
// No, we should abstract it. The evaluator callback passed from binding will handle GIL if we use pybind11 correctly?
// Actually, if we use py::gil_scoped_release in the binding, the main thread doesn't have GIL.
// When we call the std::function `evaluator`, if it wraps a Python function, pybind11 *should* acquire GIL automatically?
// Let's verify. Pybind11's `std::function` wrapper usually acquires GIL.
// However, for performance, we want to batch.

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
        int num_threads
    ) {
        std::vector<GameResultInfo> results(initial_states.size());
        InferenceQueue inf_queue;
        std::atomic<int> games_completed = 0;
        int total_games = initial_states.size();

        // Worker Lambda
        auto worker_func = [&](int game_idx) {
            SelfPlay sp(card_db_, mcts_simulations_, batch_size_);
            
            // Custom callback that routes to the queue
            BatchEvaluatorCallback worker_cb = [&](const std::vector<dm::core::GameState>& states) {
                InferenceRequest req;
                req.states = states;
                auto fut = req.promise.get_future();

                {
                    std::lock_guard<std::mutex> lock(inf_queue.mutex);
                    inf_queue.queue.push(&req);
                }
                inf_queue.cv.notify_one();

                // Wait for result
                return fut.get();
            };

            results[game_idx] = sp.play_game(initial_states[game_idx], worker_cb, temperature, add_noise);
            games_completed++;
            inf_queue.cv.notify_one(); // Notify main thread to check completion
        };

        // Launch Threads
        // We use a simple approach: launch one thread per game if num_threads >= num_games
        // Or use a pool. For simplicity, let's assume we run batches of games equal to num_threads.
        // But here we just launch threads for each game index, limited by hardware?
        // The user passes `num_threads`. We should run `num_threads` workers that pick up games.
        
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

        // Main Inference Loop
        // This runs on the thread that called play_games (which has released GIL)
        while (games_completed < total_games) {
            std::vector<InferenceRequest*> batch;
            
            {
                std::unique_lock<std::mutex> lock(inf_queue.mutex);
                // Wait for requests or completion
                inf_queue.cv.wait_for(lock, std::chrono::milliseconds(1), [&] {
                    return !inf_queue.queue.empty() || games_completed == total_games;
                });

                if (games_completed == total_games && inf_queue.queue.empty()) break;

                // Collect batch
                // We can limit max batch size here (e.g. 64 * 8 states?)
                // Let's just take everything available to minimize latency
                while (!inf_queue.queue.empty()) {
                    batch.push_back(inf_queue.queue.front());
                    inf_queue.queue.pop();
                    if (batch.size() >= 32) break; // Soft limit to avoid starving? Or hard limit for GPU memory?
                }
            }

            if (batch.empty()) continue;

            // Prepare data for evaluator
            std::vector<dm::core::GameState> all_states;
            std::vector<int> split_indices;
            
            for (auto* req : batch) {
                all_states.insert(all_states.end(), req->states.begin(), req->states.end());
                split_indices.push_back(req->states.size());
            }

            // Call Python Evaluator
            // This will re-acquire GIL inside the wrapper if needed
            auto result_pair = evaluator(all_states);
            
            // Distribute results
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

        // Join threads
        for (auto& t : threads) {
            if (t.joinable()) t.join();
        }

        return results;
    }

}
