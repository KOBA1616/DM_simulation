#pragma once
#include <vector>
#include <utility>
#include "core/game_state.hpp"

namespace dm::ai {

    // Evaluator abstraction used by MCTS: returns (policies, values)
    class IEvaluator {
    public:
        virtual ~IEvaluator() = default;

        // Evaluate a batch of GameState; returns pair of (policies, values)
        // policies: vector of policy vectors, values: scalar per state
        virtual std::pair<std::vector<std::vector<float>>, std::vector<float>>
        evaluate(const std::vector<std::shared_ptr<dm::core::GameState>>& states) = 0;
    };

}
