#pragma once
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "ai/mcts/mcts.hpp"
#include <vector>
#include <map>
#include <functional>

namespace dm::ai {

    struct GameResultInfo {
        dm::core::GameResult result;
        int turn_count;
        std::vector<dm::core::GameState> states;
        std::vector<std::vector<float>> policies;
        std::vector<int> active_players; // Who acted in this state
    };

    class SelfPlay {
    public:
        SelfPlay(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db,
                 int mcts_simulations = 50,
                 int batch_size = 1);

        // Play a single game from start to finish
        GameResultInfo play_game(
            dm::core::GameState initial_state,
            BatchEvaluatorCallback evaluator,
            float temperature = 1.0f,
            bool add_noise = true,
            float alpha = 0.0f,
            bool collect_data = true
        );

    private:
        std::map<dm::core::CardID, dm::core::CardDefinition> card_db_;
        int mcts_simulations_;
        int batch_size_;
    };

}
