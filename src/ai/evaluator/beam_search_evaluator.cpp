#include "beam_search_evaluator.hpp"
#include "../../engine/action_gen/action_generator.hpp"
#include "../../engine/effects/effect_resolver.hpp"
#include "../../engine/flow/phase_manager.hpp"
#include "../../engine/mana/mana_system.hpp"
#include "../encoders/action_encoder.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace dm::ai {

    using namespace dm::core;
    using namespace dm::engine;

    BeamSearchEvaluator::BeamSearchEvaluator(const std::map<CardID, CardDefinition>& card_db, BeamSearchConfig config)
        : card_db_(card_db), config_(config) {}

    std::pair<std::vector<std::vector<float>>, std::vector<float>>
    BeamSearchEvaluator::evaluate(const std::vector<GameState>& states) {
        std::vector<std::vector<float>> policies;
        std::vector<float> values;

        policies.reserve(states.size());
        values.reserve(states.size());

        for (const auto& state : states) {
            // For policy, we just return uniform or empty since this evaluator focuses on Value.
            // MCTS will use the Value to guide search.
            // If we want MCTS to use this policy, we could populate it based on the first step of beam search.
            // For now, let's return a uniform policy placeholder as the "Rollout" usually just gives Value.
            std::vector<float> policy(ActionEncoder::TOTAL_ACTION_SIZE, 0.0f);
            policies.push_back(policy);

            float val = evaluate_single_state(state);
            values.push_back(val);
        }

        return {policies, values};
    }

    float BeamSearchEvaluator::evaluate_single_state(const GameState& root_state) {
        // Run Beam Search to determine the likely outcome/score of this state
        // We return a value between -1.0 and 1.0 ideally for MCTS compatibility,
        // but since we are using advanced scoring, we might need to normalize or clamp.
        // MCTS expects value for the current player of root_state.

        float raw_score = run_beam_search(root_state);

        // Normalize score to [-1, 1] using tanh or simple clamping
        // Assuming score range might be large (e.g. +/- 1000)
        float normalized_score = std::tanh(raw_score / 1000.0f);

        return normalized_score;
    }

    struct BeamNode {
        GameState state;
        float score;
        // int depth;
        // Action action_taken;

        bool operator>(const BeamNode& other) const {
            return score > other.score;
        }
    };

    float BeamSearchEvaluator::run_beam_search(const GameState& start_state) {
        int root_player = start_state.active_player_id;
        std::vector<BeamNode> current_beam;

        // Initial Evaluation
        float initial_score = calculate_score(start_state, root_player);
        current_beam.push_back({start_state, initial_score});

        // Transposition table for this search (or persistent?)
        // If persistent, we need to clear it or handle collisions carefully.
        // For now, let's keep it per-search or member but beware of state changes.
        // The member `transposition_table_` is available.
        // We should clear it if we want fresh results, or use it as cache.
        // Given complexity, let's just clear for now to be safe.
        transposition_table_.clear();

        for (int d = 0; d < config_.max_depth; ++d) {
            std::vector<BeamNode> next_candidates;
            bool any_non_terminal = false;

            for (const auto& node : current_beam) {
                GameResult result;
                if (PhaseManager::check_game_over(const_cast<GameState&>(node.state), result)) {
                    // Terminal state
                    float terminal_score = 0.0f;
                    if (result == GameResult::P1_WIN) terminal_score = (root_player == 0) ? 10000.0f : -10000.0f;
                    else if (result == GameResult::P2_WIN) terminal_score = (root_player == 1) ? 10000.0f : -10000.0f;
                    else terminal_score = 0.0f; // Draw

                    // Propagate this terminal score (weighted by depth?)
                    next_candidates.push_back({node.state, terminal_score});
                    continue;
                }

                any_non_terminal = true;

                // Expand
                auto legal_actions = ActionGenerator::generate_legal_actions(node.state, card_db_);

                // Heuristic optimization: If only one action (PASS), just do it
                if (legal_actions.empty()) {
                    // Should not happen if not game over, but maybe stuck?
                    continue;
                }

                for (const auto& action : legal_actions) {
                    GameState next_state = node.state;

                    // Apply Action
                    EffectResolver::resolve_action(next_state, action, card_db_);
                     if (action.type == ActionType::PASS || action.type == ActionType::MANA_CHARGE) {
                        PhaseManager::next_phase(next_state, card_db_);
                    }
                    PhaseManager::fast_forward(next_state, card_db_);

                    // Transposition Check
                    size_t h = next_state.calculate_hash();
                    if (transposition_table_.count(h)) {
                        // Already visited this state with potentially different path
                        // We could skip, or if we want to find best score?
                        // For rollout, skip is efficient.
                        continue;
                    }

                    // Calculate Score
                    float risk = calculate_trigger_risk(node.state, action); // Risk of TAKING this action
                    float state_score = calculate_score(next_state, root_player);
                    float total_score = state_score - risk;

                    transposition_table_[h] = total_score;
                    next_candidates.push_back({next_state, total_score});
                }
            }

            if (next_candidates.empty() || !any_non_terminal) {
                break; // All paths ended
            }

            // Select Top K (Beam Width)
            if (next_candidates.size() > (size_t)config_.beam_width) {
                std::partial_sort(next_candidates.begin(),
                                  next_candidates.begin() + config_.beam_width,
                                  next_candidates.end(),
                                  std::greater<BeamNode>()); // Sort descending by score
                next_candidates.resize(config_.beam_width);
            }

            current_beam = std::move(next_candidates);

            // Check for immediate win in beam?
            // If any node in beam has huge score (Winning), we can maybe stop early?
            if (current_beam[0].score > 9000.0f) {
                return current_beam[0].score;
            }
        }

        // Return best score found in the final beam
        if (current_beam.empty()) return 0.0f;

        // Find max score in current beam
        float max_s = -100000.0f;
        for (const auto& n : current_beam) max_s = std::max(max_s, n.score);
        return max_s;
    }

    float BeamSearchEvaluator::calculate_trigger_risk(const GameState& state, const Action& action) {
        if (action.type != ActionType::ATTACK_PLAYER && action.type != ActionType::BREAK_SHIELD) {
            return 0.0f;
        }

        // 1. Identify Opponent
        int opponent_id = 1 - state.active_player_id;
        const Player& opponent = state.players[opponent_id];

        if (opponent.shield_zone.empty()) return 0.0f;

        // 2. Count Triggers in Opponent Shield Zone (God View)
        int trigger_count = 0;
        for (const auto& card : opponent.shield_zone) {
            if (card_db_.count(card.card_id)) {
                const auto& def = card_db_.at(card.card_id);
                if (def.keywords.shield_trigger) {
                    trigger_count++;
                }
            }
        }

        // 3. Calculate Probability
        float prob = (float)trigger_count / (float)opponent.shield_zone.size();

        // 4. Return Risk Score
        return prob * config_.weight_trigger_penalty_base;
    }

    float BeamSearchEvaluator::calculate_score(const GameState& state, int perspective_player_id) {
        // Score from perspective of 'perspective_player_id'

        const Player& me = state.players[perspective_player_id];
        const Player& opp = state.players[1 - perspective_player_id];

        float score = 0.0f;

        // 1. Shield Difference
        int my_shields = me.shield_zone.size();
        int opp_shields = opp.shield_zone.size();
        score += (my_shields - opp_shields) * config_.weight_shield_diff;

        // 2. Board Power & Count
        int my_power = 0;
        for (const auto& c : me.battle_zone) {
             if (card_db_.count(c.card_id)) {
                 my_power += card_db_.at(c.card_id).power + c.power_mod;
             }
        }
        int opp_power = 0;
        for (const auto& c : opp.battle_zone) {
             if (card_db_.count(c.card_id)) {
                 opp_power += card_db_.at(c.card_id).power + c.power_mod;
             }
        }

        score += (my_power - opp_power) * config_.weight_board_power;
        score += (me.battle_zone.size() - opp.battle_zone.size()) * config_.weight_creature_count;

        // 3. Hand Advantage
        score += (me.hand.size() - opp.hand.size()) * config_.weight_hand_advantage;

        // 4. Mana Advantage
        score += (me.mana_zone.size() - opp.mana_zone.size()) * config_.weight_mana_advantage;

        // 5. Opponent Danger
        float danger = calculate_opponent_danger(state, 1 - perspective_player_id);
        score -= danger * config_.weight_opponent_danger;

        // 6. Hold Bonus (If it's MY turn and I still have resources? No, this is static state eval)
        // If the state reflects "End of Turn", we can see how many cards held.
        // Just add static value for current hand size as "Potential"
        score += me.hand.size() * config_.bonus_hold_hand;

        // Unused mana check requires knowing 'tapped' state in mana zone
        int untapped_mana = 0;
        for(const auto& c : me.mana_zone) {
            if (!c.is_tapped) untapped_mana++;
        }
        score += untapped_mana * config_.bonus_hold_mana;

        // 7. Key Card Bonus
        for (const auto& c : me.hand) {
            if (config_.key_card_scores.count(c.card_id)) {
                score += config_.key_card_scores.at(c.card_id);
            }
        }
        for (const auto& c : me.battle_zone) {
            if (config_.key_card_scores.count(c.card_id)) {
                score += config_.key_card_scores.at(c.card_id) * 1.5f; // On board is better?
            }
        }

        return score;
    }

    float BeamSearchEvaluator::calculate_opponent_danger(const GameState& state, int opponent_id) {
        const Player& opp = state.players[opponent_id];
        float danger_score = 0.0f;

        // 1. Check for Key Cards in Opponent Hand/Mana
        for (const auto& c : opp.hand) {
            if (config_.key_card_scores.count(c.card_id)) {
                danger_score += config_.key_card_scores.at(c.card_id);
            }
        }
        for (const auto& c : opp.mana_zone) {
             if (config_.key_card_scores.count(c.card_id)) {
                danger_score += config_.key_card_scores.at(c.card_id) * 0.5f; // In mana is less dangerous? Or setup?
            }
        }

        // 2. Lethal Threat (Simplified)
        // If opponent has enough attackers to kill me?
        // Requires complex logic, but we can approximate:
        // Visible attackers vs My Shields + Blockers
        int potential_attackers = 0;
        for (const auto& c : opp.battle_zone) {
             if (!c.summoning_sickness) potential_attackers++; // Simplified
             // Speed attacker logic requires checking CardDefinition keywords
             else {
                 if (card_db_.count(c.card_id)) {
                     if (card_db_.at(c.card_id).keywords.speed_attacker) potential_attackers++;
                 }
             }
        }

        // Blockers...

        // If attackers > shields + defenders, Danger MAX
        int my_shields = state.players[1 - opponent_id].shield_zone.size();
        if (potential_attackers > my_shields) {
            danger_score += 500.0f; // High danger
        }

        return danger_score;
    }

}
