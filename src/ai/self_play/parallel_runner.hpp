#pragma once
#include "self_play.hpp"
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <future>

namespace dm::ai {

    class ParallelRunner {
    public:
        ParallelRunner(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db,
                       int mcts_simulations,
                       int batch_size);

        // Run multiple games in parallel
        // Returns a list of GameResultInfo
        std::vector<GameResultInfo> play_games(
            const std::vector<dm::core::GameState>& initial_states,
            BatchEvaluatorCallback evaluator,
            float temperature,
            bool add_noise,
            int num_threads
        );

    private:
        std::map<dm::core::CardID, dm::core::CardDefinition> card_db_;
        int mcts_simulations_;
        int batch_size_;

        struct InferenceRequest {
            std::vector<dm::core::GameState> states;
            std::promise<std::pair<std::vector<std::vector<float>>, std::vector<float>>> promise;
        };

        struct InferenceQueue {
            std::queue<InferenceRequest*> queue;
            std::mutex mutex;
            std::condition_variable cv;
            bool finished = false;
        };
    };

}
