#pragma once

#include <vector>
#include <memory>
#include "../../core/game_state.hpp"

namespace dm::ai::mcts {

// Synchronous batch evaluator: collects states, flattens them, calls NativeInferenceManager
// via zero-copy path (infer_flat_ptr). Minimal implementation for initial integration.
class BatchEvaluator {
public:
    BatchEvaluator(int batch_size = 8);

    // Evaluate a batch of GameState pointers. Returns pair(policy_vectors, values)
    std::pair<std::vector<std::vector<float>>, std::vector<float>> evaluate(const std::vector<std::shared_ptr<dm::core::GameState>>& states);

private:
    int batch_size_;

    std::vector<float> encode_state_flat(const dm::core::GameState& st);
};

} // namespace dm::ai::mcts
