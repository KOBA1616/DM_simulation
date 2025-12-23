#pragma once

#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "ai/encoders/tensor_converter.hpp"
#include <vector>
#include <map>
#include <memory>

namespace dm::ai {

    // Structure to hold collected data for one step
    struct TrainingSample {
        std::vector<float> state_tensor; // Flat tensor
        std::vector<float> policy_target; // Probabilities for actions
        float value_target; // 1.0 for win, -1.0 for loss, 0 for draw
        int chosen_action_index; // Index of the action taken (if we want sparse)
    };

    struct CollectedBatch {
        std::vector<std::vector<long>> states; // Changed to long for token sequences
        std::vector<std::vector<float>> policies;
        std::vector<std::vector<float>> masks;
        std::vector<float> values;
    };

    class DataCollector {
    public:
        DataCollector(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
        DataCollector(std::shared_ptr<const std::map<dm::core::CardID, dm::core::CardDefinition>> card_db);
        DataCollector(); // Default using Registry

        // Run self-play episodes and collect data (Heuristic vs Heuristic)
        CollectedBatch collect_data_batch(int episodes);
        CollectedBatch collect_data_batch_heuristic(int episodes);

    private:
        std::shared_ptr<const std::map<dm::core::CardID, dm::core::CardDefinition>> card_db_;
    };

}
