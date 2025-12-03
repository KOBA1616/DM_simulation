#pragma once

#include "../../core/game_state.hpp"
#include "../../core/card_def.hpp"
#include "../encoders/tensor_converter.hpp"
#include <vector>
#include <map>

namespace dm::ai {

    // Structure to hold collected data for one step
    struct TrainingSample {
        std::vector<float> state_tensor; // Flat tensor
        std::vector<float> policy_target; // Probabilities for actions (not used by Heuristic but kept for compatibility)
        // Wait, Heuristic Agent outputs a single action, not a probability distribution.
        // For AlphaZero training, we typically want MCTS visits as policy.
        // But if we are doing supervised learning from Heuristic Agent (imitation learning),
        // we can set the chosen action index to 1.0 and others to 0.0.
        float value_target; // 1.0 for win, -1.0 for loss, 0 for draw
        int chosen_action_index; // Index of the action taken (if we want sparse)
    };

    struct CollectedBatch {
        std::vector<std::vector<float>> states;
        std::vector<std::vector<float>> policies;
        std::vector<float> values;
    };

    class DataCollector {
    public:
        DataCollector(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);

        // Run self-play episodes and collect data
        // Uses HeuristicAgent for both players.
        CollectedBatch collect_data_batch(int episodes);

    private:
        // Store by value to ensure ownership and avoid dangling references from Python bindings
        std::map<dm::core::CardID, dm::core::CardDefinition> card_db_;
    };

}
