#pragma once
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "core/card_json_types.hpp"
#include "ai/encoders/action_encoder.hpp"
#include <vector>
#include <map>
#include <memory>

namespace dm::ai {

    class BeamSearchEvaluator {
    public:
        // Use shared_ptr to share ownership of the potentially large card database
        // and ensure lifetime safety when called from Python
        BeamSearchEvaluator(std::shared_ptr<const std::map<dm::core::CardID, dm::core::CardDefinition>> card_db, int beam_width = 7, int max_depth = 3);

        // Constructor using the singleton CardRegistry (Recommended for memory efficiency)
        BeamSearchEvaluator(int beam_width = 7, int max_depth = 3);

        // Single state evaluation
        std::pair<std::vector<float>, float> evaluate(const dm::core::GameState& state);

        // Batch evaluation (for MCTS compatibility)
        std::pair<std::vector<std::vector<float>>, std::vector<float>> evaluate_batch(const std::vector<dm::core::GameState>& states);

    private:
        std::shared_ptr<const std::map<dm::core::CardID, dm::core::CardDefinition>> card_db_;
        int beam_width_;
        int max_depth_;

        struct BeamNode {
            dm::core::GameState state;
            float score;
            std::vector<dm::core::CommandDef> path;
            dm::core::CommandDef first_action;
            bool is_root = false;

            // Explicit Move Constructor
            BeamNode(BeamNode&& other) noexcept = default;
            // Explicit Move Assignment
            BeamNode& operator=(BeamNode&& other) noexcept = default;

            // Helper constructor from GameState
            BeamNode(dm::core::GameState&& s, float sc) : state(std::move(s)), score(sc) {}

            // Deleted Copy to prevent accidental copying of GameState
            BeamNode(const BeamNode&) = delete;
            BeamNode& operator=(const BeamNode&) = delete;
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
