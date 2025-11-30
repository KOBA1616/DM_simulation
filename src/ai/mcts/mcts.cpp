#include "mcts.hpp"
#include "../../engine/action_gen/action_generator.hpp"
#include "../../engine/effects/effect_resolver.hpp"
#include "../../engine/flow/phase_manager.hpp"
#include "../encoders/action_encoder.hpp"
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>
#include <iostream>

namespace dm::ai {

    using namespace dm::core;
    using namespace dm::engine;

    MCTS::MCTS(const std::map<CardID, CardDefinition>& card_db, float c_puct, float dirichlet_alpha, float dirichlet_epsilon)
        : card_db_(card_db), c_puct_(c_puct), dirichlet_alpha_(dirichlet_alpha), dirichlet_epsilon_(dirichlet_epsilon) {}

    std::vector<float> MCTS::search(const GameState& root_state, int simulations, EvaluatorCallback evaluator, bool add_noise, float temperature) {
        // 1. Clone and Fast Forward Root
        GameState root_gs = root_state; // Copy
        PhaseManager::fast_forward(root_gs, card_db_);
        
        auto root = std::make_unique<MCTSNode>(root_gs);

        // 2. Expand Root
        expand(root.get(), evaluator);
        
        if (add_noise) {
            add_exploration_noise(root.get());
        }

        // 3. Simulations
        for (int i = 0; i < simulations; ++i) {
            MCTSNode* node = root.get();

            // Selection
            while (node->is_expanded()) {
                MCTSNode* next = select_child(node);
                if (!next) break;
                node = next;
            }

            // Expansion & Evaluation
            float value = 0.0f;
            GameResult result;
            bool is_over = PhaseManager::check_game_over(node->state, result);
            
            if (is_over) {
                // Terminal state value
                // Value is from perspective of the player who just moved (parent's active player)
                // Or simply: 1.0 if P1 wins, -1.0 if P2 wins.
                // Then backprop adjusts based on whose turn it was.
                // Let's standardize: Value is always from perspective of node->state.active_player_id?
                // No, AlphaZero usually returns value [-1, 1] for the current player.
                // If game over, we calculate value for current player.
                
                int current_player = node->state.active_player_id;
                if (result == GameResult::DRAW) value = 0.0f;
                else if (result == GameResult::P1_WIN) value = (current_player == 0) ? 1.0f : -1.0f;
                else if (result == GameResult::P2_WIN) value = (current_player == 1) ? 1.0f : -1.0f;
            } else {
                // Not terminal, expand
                // But if we just selected a node that was already expanded?
                // The while loop goes until a leaf (not expanded).
                // So node is not expanded here.
                // Unless it's terminal, in which case is_expanded is false.
                
                // Check if we can expand (generate actions)
                // expand() calls evaluator which returns value
                // But expand() logic needs to be careful not to double-expand if already visited?
                // In standard MCTS, we expand a leaf once.
                // If visit_count > 0, we might need to expand?
                // AlphaZero: Expand leaf, evaluate, backprop.
                
                // If node has 0 visits, we evaluate it.
                // If node has visits but no children (terminal), we use terminal value.
                
                // Actually, expand() populates children.
                value = expand(node, evaluator);
                // value = node->value(); // The value comes from the evaluator called in expand
                
                // Wait, expand() sets children. The value of the node itself comes from the network.
                // We need to capture that value.
                // Let's refactor expand to return value.
            }

            // Backpropagation
            backpropagate(node, value);
        }

        // 4. Compute Policy
        std::vector<float> policy(ActionEncoder::TOTAL_ACTION_SIZE, 0.0f);
        
        if (root->children.empty()) return policy;

        if (temperature < 1e-3f) {
            // Argmax
            MCTSNode* best_child = nullptr;
            int max_visits = -1;
            for (const auto& child : root->children) {
                if (child->visit_count > max_visits) {
                    max_visits = child->visit_count;
                    best_child = child.get();
                }
            }
            if (best_child) {
                int idx = ActionEncoder::action_to_index(best_child->action_from_parent);
                if (idx >= 0 && idx < (int)policy.size()) {
                    policy[idx] = 1.0f;
                }
            }
        } else {
            // Softmax with temperature
            float sum_visits_pow = 0.0f;
            std::vector<float> visits_pow;
            visits_pow.reserve(root->children.size());
            
            for (const auto& child : root->children) {
                float v = std::pow(static_cast<float>(child->visit_count), 1.0f / temperature);
                visits_pow.push_back(v);
                sum_visits_pow += v;
            }
            
            for (size_t i = 0; i < root->children.size(); ++i) {
                int idx = ActionEncoder::action_to_index(root->children[i]->action_from_parent);
                if (idx >= 0 && idx < (int)policy.size()) {
                    policy[idx] = visits_pow[i] / sum_visits_pow;
                }
            }
        }

        return policy;
    }

    float MCTS::expand(MCTSNode* node, EvaluatorCallback evaluator) {
        // Check Game Over first? Already checked in loop.
        
        // Generate Actions
        auto actions = ActionGenerator::generate_legal_actions(node->state, card_db_);
        
        if (actions.empty()) {
            // No actions? Pass or Terminal?
            // If not terminal but no actions, it's a pass (should be generated) or stuck.
            // Assuming ActionGenerator always returns something if not game over.
            return 0.0f;
        }

        // Evaluate
        auto [policy_logits, value] = evaluator(node->state);
        
        // Create Children
        // We need to normalize policy logits to probabilities?
        // Usually Python side does softmax. Let's assume we get probabilities or logits?
        // The callback signature says vector<float>.
        // Let's assume they are probabilities (softmaxed).
        
        for (const auto& action : actions) {
            int idx = ActionEncoder::action_to_index(action);
            float p = 0.0f;
            if (idx >= 0 && idx < (int)policy_logits.size()) {
                p = policy_logits[idx];
            }
            
            // Create Child
            GameState next_state = node->state; // Copy
            EffectResolver::resolve_action(next_state, action, card_db_);
            if (action.type == ActionType::PASS) {
                PhaseManager::next_phase(next_state, card_db_);
            }
            PhaseManager::fast_forward(next_state, card_db_);
            
            auto child = std::make_unique<MCTSNode>(next_state);
            child->parent = node;
            child->action_from_parent = action;
            child->prior = p;
            
            node->children.push_back(std::move(child));
        }
        
        return value;
    }

    MCTSNode* MCTS::select_child(MCTSNode* node) {
        MCTSNode* best_child = nullptr;
        float best_score = -std::numeric_limits<float>::infinity();

        for (const auto& child : node->children) {
            float q = child->value();
            // UCB
            // c_puct * P(s,a) * sqrt(sum(N)) / (1 + N)
            float u = c_puct_ * child->prior * std::sqrt(static_cast<float>(node->visit_count)) / (1.0f + child->visit_count);
            
            // Q is from perspective of child's active player.
            // We are at 'node' (Parent).
            // If child's active player is same as parent's, Q is good.
            // If child's active player is opponent, Q is bad for us.
            // AlphaZero: v is from perspective of current player.
            // If child state is opponent's turn, the network returns v for opponent.
            // So for us, it is -v.
            // However, child->value() averages the backpropagated values.
            // In backprop, we flip value.
            // So child->value() should already be "Value for the player who made the move (Parent's active player)"?
            // Let's check backprop.
            
            float score = q + u;
            if (score > best_score) {
                best_score = score;
                best_child = child.get();
            }
        }
        return best_child;
    }

    void MCTS::backpropagate(MCTSNode* node, float value) {
        while (node) {
            node->visit_count++;
            node->value_sum += value;
            
            // Flip value for parent
            // If parent's active player is different from current node's active player?
            // Usually in 2-player zero-sum:
            // Value is always relative to the player whose turn it is at that node.
            // So when moving up to parent, we flip.
            value = -value;
            
            node = node->parent;
        }
    }

    void MCTS::add_exploration_noise(MCTSNode* node) {
        if (node->children.empty()) return;
        
        std::mt19937 rng(std::random_device{}());
        std::gamma_distribution<float> dist(dirichlet_alpha_, 1.0f);
        
        std::vector<float> noise;
        float sum = 0.0f;
        for (size_t i = 0; i < node->children.size(); ++i) {
            float n = dist(rng);
            noise.push_back(n);
            sum += n;
        }
        
        for (size_t i = 0; i < node->children.size(); ++i) {
            node->children[i]->prior = node->children[i]->prior * (1 - dirichlet_epsilon_) + (noise[i] / sum) * dirichlet_epsilon_;
        }
    }

}
