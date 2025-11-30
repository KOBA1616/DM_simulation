#pragma once
#include "../../core/game_state.hpp"
#include "../../core/card_def.hpp"
#include <vector>
#include <map>

namespace dm::ai {

    class HeuristicEvaluator {
    public:
        HeuristicEvaluator(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);

        // Matches BatchEvaluatorCallback signature
        std::pair<std::vector<std::vector<float>>, std::vector<float>> evaluate(const std::vector<dm::core::GameState>& states);

    private:
        std::map<dm::core::CardID, dm::core::CardDefinition> card_db_;
    };

}
