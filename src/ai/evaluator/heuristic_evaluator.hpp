#pragma once
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "evaluator.hpp"
#include <vector>
#include <map>
#include <memory>

namespace dm::ai {

    class HeuristicEvaluator : public IEvaluator {
    public:
        // Constructor using shared pointer (Recommended)
        HeuristicEvaluator(std::shared_ptr<const std::map<dm::core::CardID, dm::core::CardDefinition>> card_db);

        // Constructor using reference (Legacy, makes a copy into shared_ptr for safety/consistency)
        HeuristicEvaluator(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);

        // Default constructor (Uses CardRegistry)
        HeuristicEvaluator();

        // Implements IEvaluator
        std::pair<std::vector<std::vector<float>>, std::vector<float>>
        evaluate(const std::vector<std::shared_ptr<dm::core::GameState>>& states) override;

    private:
        std::shared_ptr<const std::map<dm::core::CardID, dm::core::CardDefinition>> card_db_;
    };

}
