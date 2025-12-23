#include "heuristic_evaluator.hpp"
#include "engine/actions/action_generator.hpp"
#include "engine/systems/card/card_registry.hpp"
#include "ai/encoders/action_encoder.hpp"
#include <algorithm>
#include "ai/inference/torch_model.hpp"

namespace dm::ai {

    HeuristicEvaluator::HeuristicEvaluator(std::shared_ptr<const std::map<dm::core::CardID, dm::core::CardDefinition>> card_db)
        : card_db_(card_db) {}

    HeuristicEvaluator::HeuristicEvaluator(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db)
        : card_db_(std::make_shared<std::map<dm::core::CardID, dm::core::CardDefinition>>(card_db)) {}

    HeuristicEvaluator::HeuristicEvaluator()
        : card_db_(dm::engine::CardRegistry::get_all_definitions_ptr()) {}

    std::pair<std::vector<std::vector<float>>, std::vector<float>> HeuristicEvaluator::evaluate(const std::vector<std::shared_ptr<dm::core::GameState>>& states) {
        std::vector<std::vector<float>> policies;
        std::vector<float> values;
        policies.reserve(states.size());
        values.reserve(states.size());

        for (const auto& state_ptr : states) {
            const auto& state = *state_ptr;
            // 1. Calculate Value
            // P0 perspective
            const auto& p0 = state.players[0];
            const auto& p1 = state.players[1];

            float score = 0.0f;

            // Shield Advantage (High weight)
            score += (p0.shield_zone.size() - p1.shield_zone.size()) * 0.2f;

            // Hand Advantage (Medium weight)
            score += (p0.hand.size() - p1.hand.size()) * 0.05f;

            // Battle Zone Advantage (Medium weight)
            // Ideally power sum, but count is a good proxy for now
            score += (p0.battle_zone.size() - p1.battle_zone.size()) * 0.1f;

            // Mana Advantage (Low weight, mostly for early game)
            score += (p0.mana_zone.size() - p1.mana_zone.size()) * 0.02f;

            // Normalize roughly to [-1, 1]
            float val = std::max(-1.0f, std::min(1.0f, score));
            
            if (state.active_player_id == 1) {
                val = -val;
            }
            values.push_back(val);

            // 2. Calculate Policy (Uniform over legal actions)
            std::vector<float> policy(ActionEncoder::TOTAL_ACTION_SIZE, 0.0f);
            auto legal_actions = dm::engine::ActionGenerator::generate_legal_actions(const_cast<dm::core::GameState&>(state), *card_db_);
            
            if (!legal_actions.empty()) {
                float prob = 1.0f / legal_actions.size();
                for (const auto& action : legal_actions) {
                    int idx = ActionEncoder::action_to_index(action);
                    if (idx >= 0 && idx < ActionEncoder::TOTAL_ACTION_SIZE) {
                        policy[idx] += prob; // += in case multiple actions map to same index (shouldn't happen often)
                    }
                }
            }
            policies.push_back(std::move(policy));
        }

        return {policies, values};
    }

}
