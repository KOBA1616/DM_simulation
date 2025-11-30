#include "heuristic_evaluator.hpp"
#include "../../engine/action_gen/action_generator.hpp"
#include "../encoders/action_encoder.hpp"
#include <algorithm>

namespace dm::ai {

    HeuristicEvaluator::HeuristicEvaluator(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db)
        : card_db_(card_db) {}

    std::pair<std::vector<std::vector<float>>, std::vector<float>> HeuristicEvaluator::evaluate(const std::vector<dm::core::GameState>& states) {
        std::vector<std::vector<float>> policies;
        std::vector<float> values;
        policies.reserve(states.size());
        values.reserve(states.size());

        for (const auto& state : states) {
            // 1. Calculate Value (Shield Difference)
            // P0 perspective
            int p0_shields = (int)state.players[0].shield_zone.size();
            int p1_shields = (int)state.players[1].shield_zone.size();
            float diff = (float)(p0_shields - p1_shields);
            
            // Normalize roughly to [-1, 1] (5 shields max usually)
            float val = diff * 0.2f; 
            val = std::max(-1.0f, std::min(1.0f, val));

            // Value is always from the perspective of the current player for MCTS?
            // Usually AlphaZero value head returns value for the current player or always P0?
            // In my MCTS implementation (standard AlphaZero), value is usually for the player who just moved (parent) or current player?
            // Let's check MCTS::backpropagate.
            // If MCTS expects value for the player whose turn it is in the leaf node:
            // If active_player is 0, and val is high (good for 0), return val.
            // If active_player is 1, and val is high (good for 0), return -val.
            
            if (state.active_player_id == 1) {
                val = -val;
            }
            values.push_back(val);

            // 2. Calculate Policy (Uniform over legal actions)
            std::vector<float> policy(ActionEncoder::TOTAL_ACTION_SIZE, 0.0f);
            auto legal_actions = dm::engine::ActionGenerator::generate_legal_actions(const_cast<dm::core::GameState&>(state), card_db_);
            
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
