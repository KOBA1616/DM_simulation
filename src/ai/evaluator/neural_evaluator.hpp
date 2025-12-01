#pragma once
#include "evaluator.hpp"
#include <map>
#include "../../core/card_def.hpp"

namespace dm::ai {

    class NeuralEvaluator : public IEvaluator {
    public:
        NeuralEvaluator(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);

        std::pair<std::vector<std::vector<float>>, std::vector<float>>
        evaluate(const std::vector<dm::core::GameState>& states) override;

    private:
        std::map<dm::core::CardID, dm::core::CardDefinition> card_db_;
    };

}
