#include "beam_search_evaluator.hpp"
#include "engine/actions/action_generator.hpp"
#include "engine/systems/game_logic_system.hpp"
#include "engine/systems/flow/phase_manager.hpp"
#include "engine/systems/mana/mana_system.hpp"
#include <cmath>
#include <algorithm>
#include <iostream>

namespace dm::ai {

    using namespace dm::core;
    using namespace dm::engine;
    using namespace dm::engine::systems;

    BeamSearchEvaluator::BeamSearchEvaluator(const std::map<CardID, CardDefinition>& card_db, int beam_width, int max_depth)
        : card_db_(card_db), beam_width_(beam_width), max_depth_(max_depth) {}

    std::pair<std::vector<float>, float> BeamSearchEvaluator::evaluate(const GameState& state) {
        std::vector<float> policy(ActionEncoder::TOTAL_ACTION_SIZE, 0.0f); // Logits (or probs)
        float value = run_beam_search(state, policy);
        return {policy, value};
    }

    std::pair<std::vector<std::vector<float>>, std::vector<float>> BeamSearchEvaluator::evaluate_batch(const std::vector<GameState>& states) {
        std::vector<std::vector<float>> policies;
        std::vector<float> values;
        policies.reserve(states.size());
        values.reserve(states.size());

        for (const auto& state : states) {
            auto [p, v] = evaluate(state);
            policies.push_back(p);
            values.push_back(v);
        }
        return {policies, values};
    }

    float BeamSearchEvaluator::run_beam_search(const GameState& root_state, std::vector<float>& policy_logits) {
        PlayerID root_player = root_state.active_player_id;

        // Initial Nodes (Beam)
        std::vector<BeamNode> current_beam;

        // Generate actions from root to initialize first step policy
        auto root_actions = ActionGenerator::generate_legal_actions(root_state, card_db_);
        if (root_actions.empty()) {
            return -1.0f; // Loss (No moves)
        }

        // Initialize beam with root children
        for (const auto& action : root_actions) {
            GameState next_state = root_state;

            // Use GameLogicSystem instead of EffectResolver
            GameLogicSystem::resolve_action_oneshot(next_state, action, card_db_);

            if (action.type == ActionType::PASS || action.type == ActionType::MANA_CHARGE) {
                PhaseManager::next_phase(next_state, card_db_);
            }
            PhaseManager::fast_forward(next_state, card_db_);

            float score = evaluate_state_heuristic(next_state, root_player);

            BeamNode node;
            node.state = next_state;
            node.score = score;
            node.first_action = action;
            current_beam.push_back(node);
        }

        // Populate policy based on immediate children scores
        // Simple softmax-like distribution based on heuristic scores
        // Or just 1.0 for best, 0.0 for others (Argmax)
        // Let's use scores to set logits. Score range -1 to 1? Or heuristic range.

        // Sort to keep top K
        std::sort(current_beam.begin(), current_beam.end(), [](const BeamNode& a, const BeamNode& b) {
            return a.score > b.score;
        });

        if (current_beam.size() > (size_t)beam_width_) {
            current_beam.resize(beam_width_);
        }

        // Set policy for root
        // We give higher prob to actions that survived the cut
        float min_score = current_beam.back().score;
        for (const auto& node : current_beam) {
            int idx = ActionEncoder::action_to_index(node.first_action);
            if (idx >= 0 && idx < (int)policy_logits.size()) {
                // Heuristic mapping to logit
                policy_logits[idx] = (node.score - min_score) * 10.0f + 1.0f;
            }
        }

        // Deep Search
        for (int d = 1; d < max_depth_; ++d) {
            std::vector<BeamNode> next_candidates;

            for (const auto& node : current_beam) {
                // If game over, keep it
                GameResult res;
                // Copy state for const correctness check_game_over usually shouldn't modify state but signature is non-const
                GameState check_state = node.state;
                if (PhaseManager::check_game_over(check_state, res)) {
                    next_candidates.push_back(node);
                    continue;
                }

                auto actions = ActionGenerator::generate_legal_actions(node.state, card_db_);
                for (const auto& action : actions) {
                    GameState next_state = node.state;

                    // Use GameLogicSystem
                    GameLogicSystem::resolve_action_oneshot(next_state, action, card_db_);

                    if (action.type == ActionType::PASS || action.type == ActionType::MANA_CHARGE) {
                        PhaseManager::next_phase(next_state, card_db_);
                    }
                    PhaseManager::fast_forward(next_state, card_db_);

                    float score = evaluate_state_heuristic(next_state, root_player);

                    BeamNode child;
                    child.state = next_state;
                    child.score = score;
                    child.first_action = node.first_action; // Keep tracking origin
                    next_candidates.push_back(child);
                }
            }

            if (next_candidates.empty()) break;

            // Prune
            std::sort(next_candidates.begin(), next_candidates.end(), [](const BeamNode& a, const BeamNode& b) {
                return a.score > b.score;
            });

            if (next_candidates.size() > (size_t)beam_width_) {
                next_candidates.resize(beam_width_);
            }

            current_beam = next_candidates;
        }

        // Return best score found
        if (current_beam.empty()) return -1.0f;
        return current_beam[0].score;
    }

    float BeamSearchEvaluator::evaluate_state_heuristic(const GameState& state, PlayerID perspective) {
        // 1. Check Win/Loss
        GameResult res;
        GameState check_state = state; // Copy for non-const check
        if (PhaseManager::check_game_over(check_state, res)) {
            if (res == GameResult::DRAW) return 0.0f;
            if (res == GameResult::P1_WIN) return (perspective == 0) ? 1000.0f : -1000.0f;
            if (res == GameResult::P2_WIN) return (perspective == 1) ? 1000.0f : -1000.0f;
        }

        float score = 0.0f;

        // 2. Resource Advantage
        score += calculate_resource_advantage(state, perspective);

        // 3. Opponent Danger (Negative score)
        score -= calculate_opponent_danger(state, perspective);

        // 4. Trigger Risk (Negative score)
        score -= calculate_trigger_risk(state, perspective);

        return score;
    }

    float BeamSearchEvaluator::calculate_resource_advantage(const GameState& state, PlayerID perspective) {
        const Player& me = state.players[perspective];
        const Player& opp = state.players[1 - perspective];

        float score = 0.0f;

        // Mana
        score += (me.mana_zone.size() - opp.mana_zone.size()) * 0.5f;

        // Hand
        score += (me.hand.size() - opp.hand.size()) * 1.0f;

        // Battle Zone (Count + Power)
        score += (me.battle_zone.size() - opp.battle_zone.size()) * 2.0f;

        // Shields (More is better)
        score += (me.shield_zone.size() - opp.shield_zone.size()) * 5.0f;

        return score;
    }

    float BeamSearchEvaluator::calculate_opponent_danger(const GameState& state, PlayerID perspective) {
        PlayerID opp_id = 1 - perspective;
        const Player& opp = state.players[opp_id];

        float danger = 0.0f;

        // Scan Hand and Mana for Key Cards
        auto scan_zone = [&](const std::vector<CardInstance>& zone, float multiplier) {
            for (const auto& card : zone) {
                if (card_db_.find(card.card_id) != card_db_.end()) {
                    const auto& def = card_db_.at(card.card_id);
                    if (def.is_key_card) {
                        float imp = (float)def.ai_importance_score;
                        if (imp == 0) imp = 50.0f; // Default if flagged but score 0
                        danger += imp * multiplier;
                    }
                }
            }
        };

        scan_zone(opp.hand, 1.0f); // In hand = Danger
        scan_zone(opp.mana_zone, 0.5f); // In mana = Potential Danger (resources) or just scary
        scan_zone(opp.battle_zone, 2.0f); // In play = High Danger

        return danger;
    }

    float BeamSearchEvaluator::calculate_trigger_risk(const GameState& state, PlayerID perspective) {
        PlayerID opp_id = 1 - perspective;
        const Player& opp = state.players[opp_id];

        // God View: Scan Opponent Shields
        int trigger_count = 0;
        int total_shields = opp.shield_zone.size();

        if (total_shields == 0) return 0.0f;

        for (const auto& card : opp.shield_zone) {
            if (card_db_.find(card.card_id) != card_db_.end()) {
                const auto& def = card_db_.at(card.card_id);
                if (def.keywords.shield_trigger) {
                    trigger_count++;
                }
            }
        }

        float probability = (float)trigger_count / (float)total_shields;

        // Penalty is proportional to probability * impact
        float impact = 20.0f;

        return probability * impact * trigger_count;
    }

}
