#pragma once
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "ai/mcts/mcts.hpp"
#include "ai/inference/pimc_generator.hpp"
#include <vector>
#include <map>
#include <memory>
#include <functional>

namespace dm::ai {

    struct GameResultInfo {
        dm::core::GameResult result;
        int turn_count;
        std::vector<std::shared_ptr<dm::core::GameState>> states;
        std::vector<std::vector<float>> policies;
        std::vector<int> active_players; // Who acted in this state
    };

    class SelfPlay {
    public:
        SelfPlay(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db,
                 int mcts_simulations = 50,
                 int batch_size = 1);

        SelfPlay(std::shared_ptr<const std::map<dm::core::CardID, dm::core::CardDefinition>> card_db,
                 int mcts_simulations = 50,
                 int batch_size = 1);

        void set_pimc_generator(std::shared_ptr<dm::ai::inference::PimcGenerator> pimc_generator);

        // Play a single game from start to finish
        // Changed to pass by reference to avoid copy constructor issue
        GameResultInfo play_game(
            const dm::core::GameState& initial_state,
            BatchEvaluatorCallback evaluator,
            float temperature = 1.0f,
            bool add_noise = true,
            float alpha = 0.0f,
            bool collect_data = true
        );

    private:
        std::shared_ptr<const std::map<dm::core::CardID, dm::core::CardDefinition>> card_db_;
        std::shared_ptr<dm::ai::inference::PimcGenerator> pimc_generator_;
        int mcts_simulations_;
        int batch_size_;
    };

}
