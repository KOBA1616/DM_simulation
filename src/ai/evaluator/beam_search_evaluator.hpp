#pragma once
#include "evaluator.hpp"
#include "../../core/game_state.hpp"
#include "../../core/card_def.hpp"
#include "../../core/action.hpp"
#include <map>
#include <vector>
#include <unordered_map>
#include <memory>

namespace dm::ai {

    // Config for the Beam Search Evaluator
    struct BeamSearchConfig {
        int beam_width = 7;
        int max_depth = 5; // To prevent infinite loops if game doesn't end

        // Scoring Weights
        float weight_shield_diff = 500.0f;
        float weight_board_power = 0.1f;
        float weight_hand_advantage = 50.0f;
        float weight_mana_advantage = 30.0f;
        float weight_creature_count = 40.0f;
        float weight_trigger_penalty_base = 1000.0f;

        float bonus_hold_hand = 10.0f;
        float bonus_hold_mana = 2.0f;

        float weight_opponent_danger = 100.0f;

        // Key Card Definitions (CardID -> Importance Score)
        std::map<dm::core::CardID, float> key_card_scores;
    };

    class BeamSearchEvaluator : public IEvaluator {
    public:
        BeamSearchEvaluator(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, BeamSearchConfig config = {});

        // IEvaluator Interface
        // Evaluates the state by running a beam search rollout
        std::pair<std::vector<std::vector<float>>, std::vector<float>>
        evaluate(const std::vector<dm::core::GameState>& states) override;

    private:
        std::map<dm::core::CardID, dm::core::CardDefinition> card_db_;
        BeamSearchConfig config_;

        // Transposition Table (Hash -> Score)
        std::unordered_map<size_t, float> transposition_table_;

        // Core logic
        float evaluate_single_state(const dm::core::GameState& root_state);
        float run_beam_search(const dm::core::GameState& start_state);

        // Scoring
        float calculate_score(const dm::core::GameState& state, int perspective_player_id);
        float calculate_trigger_risk(const dm::core::GameState& state, const dm::core::Action& action);
        float calculate_opponent_danger(const dm::core::GameState& state, int opponent_id);
    };

}
