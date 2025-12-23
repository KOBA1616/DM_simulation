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
            GameState next_state = root_state.clone(); // Use clone() instead of copy
            GameLogicSystem::resolve_action(next_state, action, card_db_);
            if (action.type == ActionType::PASS || action.type == ActionType::MANA_CHARGE) {
                PhaseManager::next_phase(next_state, card_db_);
            }
            PhaseManager::fast_forward(next_state, card_db_);

            float score = evaluate_state_heuristic(next_state, root_player);

            current_beam.emplace_back(std::move(next_state), score);
            current_beam.back().first_action = action;
        }

        // Sort to keep top K
        std::sort(current_beam.begin(), current_beam.end(), [](const BeamNode& a, const BeamNode& b) {
            return a.score > b.score;
        });

        if (current_beam.size() > (size_t)beam_width_) {
            // Cannot use resize because it might require default ctor or copy/move assignment that vector likes
            // But we have move assignment.
            // Erase elements from end.
            current_beam.erase(current_beam.begin() + beam_width_, current_beam.end());
        }

        // Set policy for root
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
                // Check win/loss on cloned state (PhaseManager::check_game_over modifies state?)
                // check_game_over signature is non-const but usually logic is const.
                // Assuming it's safe or we accept side effects on this node.
                GameResult res;
                // Note: We can't easily copy `node.state` because copy ctor is deleted.
                // We MUST use clone().
                // And check_game_over probably doesn't need to modify state if implemented well.
                // But PhaseManager::check_game_over(GameState& state) ...

                // Workaround: clone to check game over? Expensive.
                // Or assume node.state is mutable (it is) and we can just pass it?
                // But `node` is const ref in loop `for (const auto& node : current_beam)`.
                // We need a non-const copy or clone.

                // Let's iterate non-const if possible?
                // `current_beam` is local.
                // But resizing and vector reassignment complicates using references.

                // For generation, we need to clone anyway for each child.

                // So:
                GameState base_state = node.state.clone();
                if (PhaseManager::check_game_over(base_state, res)) {
                    // This path ends. Add to next_candidates?
                    // If we add it, we stop expanding.
                    // Score is already calculated in parent step.
                    // We can carry it over.
                    next_candidates.emplace_back(std::move(base_state), node.score);
                    next_candidates.back().first_action = node.first_action;
                    continue;
                }

                auto actions = ActionGenerator::generate_legal_actions(base_state, card_db_);
                for (const auto& action : actions) {
                    GameState next_state = base_state.clone();
                    GameLogicSystem::resolve_action(next_state, action, card_db_);
                    if (action.type == ActionType::PASS || action.type == ActionType::MANA_CHARGE) {
                        PhaseManager::next_phase(next_state, card_db_);
                    }
                    PhaseManager::fast_forward(next_state, card_db_);

                    float score = evaluate_state_heuristic(next_state, root_player);

                    next_candidates.emplace_back(std::move(next_state), score);
                    next_candidates.back().first_action = node.first_action;
                }
            }

            if (next_candidates.empty()) break;

            // Prune
            std::sort(next_candidates.begin(), next_candidates.end(), [](const BeamNode& a, const BeamNode& b) {
                return a.score > b.score;
            });

            if (next_candidates.size() > (size_t)beam_width_) {
                next_candidates.erase(next_candidates.begin() + beam_width_, next_candidates.end());
            }

            // Move next_candidates to current_beam
            // BeamNode has move assignment
            current_beam = std::move(next_candidates);
        }

        // Return best score found
        if (current_beam.empty()) return -1.0f;
        return current_beam[0].score;
    }

    float BeamSearchEvaluator::evaluate_state_heuristic(const GameState& state, PlayerID perspective) {
        // 1. Check Win/Loss
        GameResult res;
        GameState check_state = state.clone(); // Clone for safety
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

        auto scan_zone = [&](const std::vector<CardInstance>& zone, float multiplier) {
            for (const auto& card : zone) {
                if (card_db_.find(card.card_id) != card_db_.end()) {
                    const auto& def = card_db_.at(card.card_id);
                    if (def.is_key_card) {
                        float imp = (float)def.ai_importance_score;
                        if (imp == 0) imp = 50.0f;
                        danger += imp * multiplier;
                    }
                }
            }
        };

        scan_zone(opp.hand, 1.0f);
        scan_zone(opp.mana_zone, 0.5f);
        scan_zone(opp.battle_zone, 2.0f);

        return danger;
    }

    float BeamSearchEvaluator::calculate_trigger_risk(const GameState& state, PlayerID perspective) {
        PlayerID opp_id = 1 - perspective;
        const Player& opp = state.players[opp_id];

        int trigger_count = 0;
        int total_shields = opp.shield_zone.size();

        if (total_shields == 0) return 0.0f;

        for (const auto& card : opp.shield_zone) {
            if (card_db_.find(card.card_id) != card_db_.end()) {
                const auto& def = card_db_.at(card.card_id);
                if (def.keywords.has(Keyword::SHIELD_TRIGGER)) {
                    trigger_count++;
                }
            }
        }

        float probability = (float)trigger_count / (float)total_shields;
        float impact = 20.0f;

        return probability * impact * trigger_count;
    }

}
