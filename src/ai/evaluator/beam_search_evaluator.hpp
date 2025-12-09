#pragma once
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "ai/encoders/action_encoder.hpp"
#include <vector>
#include <map>
#include <memory>

namespace dm::ai {

    class BeamSearchEvaluator {
    public:
        BeamSearchEvaluator(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, int beam_width = 7, int max_depth = 3);

        // Single state evaluation
        std::pair<std::vector<float>, float> evaluate(const dm::core::GameState& state);

        // Batch evaluation (for MCTS compatibility)
        std::pair<std::vector<std::vector<float>>, std::vector<float>> evaluate_batch(const std::vector<dm::core::GameState>& states);

    private:
        std::map<dm::core::CardID, dm::core::CardDefinition> card_db_;
        int beam_width_;
        int max_depth_;

        struct BeamNode {
            dm::core::GameState state;
            float score;
            std::vector<dm::core::Action> path; // Path taken to reach this node (not strictly needed for eval but useful for debug)
            dm::core::Action first_action; // The action taken from the root to start this path
            bool is_root = false;
        };

        // Core search logic
        float run_beam_search(const dm::core::GameState& root_state, std::vector<float>& policy_logits);

        // Heuristic Scoring
        float evaluate_state_heuristic(const dm::core::GameState& state, dm::core::PlayerID perspective);

        // Sub-heuristics
        float calculate_opponent_danger(const dm::core::GameState& state, dm::core::PlayerID perspective);
        float calculate_trigger_risk(const dm::core::GameState& state, dm::core::PlayerID perspective);
        float calculate_resource_advantage(const dm::core::GameState& state, dm::core::PlayerID perspective);
    };

}
