#pragma once

#include "../../core/game_state.hpp"
#include "../../core/card_def.hpp"
#include "../../core/scenario_config.hpp"
#include "self_play.hpp"
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <atomic>
#include <queue>

namespace dm::ai {

    // Structure for batch inference request
    struct InferenceRequest {
        std::vector<dm::core::GameState> states;
        std::promise<std::pair<std::vector<std::vector<float>>, std::vector<float>>> promise;
    };

    // Queue for inference requests
    struct InferenceQueue {
        std::queue<InferenceRequest*> queue;
        std::mutex mutex;
        std::condition_variable cv;
    };

    class ParallelRunner {
    public:
        ParallelRunner(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db,
                       int mcts_simulations,
                       int batch_size);

        // Run self-play games in parallel using MCTS and Python evaluator
        std::vector<GameResultInfo> play_games(
            const std::vector<dm::core::GameState>& initial_states,
            BatchEvaluatorCallback evaluator,
            float temperature = 1.0f,
            bool add_noise = true,
            int num_threads = 4
        );

        // Run Scenario matches in parallel (Heuristic vs Heuristic)
        // Returns vector of winner IDs (0=Draw, 1=P1, 2=P2)
        std::vector<int> play_scenario_match(
            const dm::core::ScenarioConfig& config,
            int num_games,
            int num_threads
        );

        // Run Deck vs Deck matches in parallel (Heuristic vs Heuristic)
        // Returns vector of winner IDs (0=Draw, 1=P1, 2=P2)
        std::vector<int> play_deck_matchup(
            const std::vector<dm::core::CardID>& deck1,
            const std::vector<dm::core::CardID>& deck2,
            int num_games,
            int num_threads
        );

    private:
        std::map<dm::core::CardID, dm::core::CardDefinition> card_db_;
        int mcts_simulations_;
        int batch_size_;
    };

}
