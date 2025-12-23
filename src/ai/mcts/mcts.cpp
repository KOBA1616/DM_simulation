#include "mcts.hpp"
#include "engine/actions/action_generator.hpp"
#include "engine/systems/game_logic_system.hpp"
#include "engine/systems/flow/phase_manager.hpp"
#include "engine/systems/card/card_registry.hpp"
#include "ai/encoders/action_encoder.hpp"
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>
#include <iostream>

namespace dm::ai {

    using namespace dm::core;
    using namespace dm::engine;
    using namespace dm::engine::systems;

    MCTSNode::~MCTSNode() {
        // Iterative destruction to avoid stack overflow for deep trees
        std::vector<std::unique_ptr<MCTSNode>> nodes_to_delete;
        // Move children to local vector
        for (auto& child : children) {
            if (child) nodes_to_delete.push_back(std::move(child));
        }
        children.clear();

        while (!nodes_to_delete.empty()) {
            auto node = std::move(nodes_to_delete.back());
            nodes_to_delete.pop_back();

            if (node) {
                for (auto& child : node->children) {
                    if (child) nodes_to_delete.push_back(std::move(child));
                }
                node->children.clear();
            }
        }
    }

    MCTS::MCTS(const std::map<CardID, CardDefinition>& card_db, float c_puct, float dirichlet_alpha, float dirichlet_epsilon, int batch_size, float alpha)
        : card_db_(std::make_shared<std::map<CardID, CardDefinition>>(card_db)), c_puct_(c_puct), dirichlet_alpha_(dirichlet_alpha), dirichlet_epsilon_(dirichlet_epsilon), batch_size_(batch_size), alpha_(alpha) {}

    MCTS::MCTS(std::shared_ptr<const std::map<CardID, CardDefinition>> card_db, float c_puct, float dirichlet_alpha, float dirichlet_epsilon, int batch_size, float alpha)
        : card_db_(card_db), c_puct_(c_puct), dirichlet_alpha_(dirichlet_alpha), dirichlet_epsilon_(dirichlet_epsilon), batch_size_(batch_size), alpha_(alpha) {}

    MCTS::MCTS(float c_puct, float dirichlet_alpha, float dirichlet_epsilon, int batch_size, float alpha)
        : card_db_(dm::engine::CardRegistry::get_all_definitions_ptr()), c_puct_(c_puct), dirichlet_alpha_(dirichlet_alpha), dirichlet_epsilon_(dirichlet_epsilon), batch_size_(batch_size), alpha_(alpha) {}

    std::vector<float> MCTS::search(const GameState& root_state, int simulations, BatchEvaluatorCallback evaluator, bool add_noise, float temperature) {
        // 1. Clone and Fast Forward Root
        GameState root_gs = root_state.clone(); // Copy
        PhaseManager::fast_forward(root_gs, *card_db_);
        
        // Explicitly free the old tree before allocating the new one to reduce peak memory usage
        last_root_.reset();
        last_root_ = std::make_unique<MCTSNode>(std::move(root_gs));
        MCTSNode* root = last_root_.get();

        // Initial expansion of root (needs evaluation)
        // Avoid initializer list copying GameState
        std::vector<std::shared_ptr<GameState>> root_batch;
        root_batch.reserve(1);
        root_batch.push_back(std::make_shared<GameState>(root->state.clone()));

        auto [root_policies, root_values] = evaluator(root_batch);
        expand_node(root, root_policies[0]);
        backpropagate(root, root_values[0]);
        
        if (add_noise) {
            add_exploration_noise(root);
        }

        int simulations_finished = 0;
        
        // 3. Simulations Loop
        while (simulations_finished < simulations) {
            std::vector<MCTSNode*> batch_nodes;
            std::vector<std::shared_ptr<GameState>> batch_states;
            
            int current_batch_limit = std::min(batch_size_, simulations - simulations_finished);
            
            for (int i = 0; i < current_batch_limit; ++i) {
                MCTSNode* leaf = select_leaf(root);
                
                bool already_in_batch = false;
                for (auto* n : batch_nodes) {
                    if (n == leaf) {
                        already_in_batch = true;
                        break;
                    }
                }
                
                if (already_in_batch) {
                    revert_virtual_loss(leaf);
                    break;
                }

                GameResult result;
                bool is_over = PhaseManager::check_game_over(leaf->state, result);
                
                if (is_over) {
                    float value = 0.0f;
                    int current_player = leaf->state.active_player_id;
                    if (result == GameResult::DRAW) value = 0.0f;
                    else if (result == GameResult::P1_WIN) value = (current_player == 0) ? 1.0f : -1.0f;
                    else if (result == GameResult::P2_WIN) value = (current_player == 1) ? 1.0f : -1.0f;
                    
                    backpropagate(leaf, value);
                    revert_virtual_loss(leaf);
                    simulations_finished++;
                } else {
                    batch_nodes.push_back(leaf);
                    // Explicit clone wrapped in shared_ptr
                    batch_states.push_back(std::make_shared<GameState>(leaf->state.clone()));
                }
            }
            
            if (batch_nodes.empty()) {
                if (simulations_finished >= simulations) break;
                continue;
            }
            
            auto [policies, values] = evaluator(batch_states);
            
            for (size_t i = 0; i < batch_nodes.size(); ++i) {
                MCTSNode* node = batch_nodes[i];
                expand_node(node, policies[i]);
                backpropagate(node, values[i]);
                revert_virtual_loss(node);
                simulations_finished++;
            }
        }

        // 4. Compute Policy
        std::vector<float> policy(ActionEncoder::TOTAL_ACTION_SIZE, 0.0f);
        
        if (root->children.empty()) return policy;

        if (temperature < 1e-3f) {
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

    void MCTS::expand_node(MCTSNode* node, const std::vector<float>& policy_logits) {
        auto actions = ActionGenerator::generate_legal_actions(node->state, *card_db_);
        
        if (actions.empty()) return;

        float sum_exp = 0.0f;
        std::vector<float> priors;
        priors.reserve(actions.size());

        for (const auto& action : actions) {
            int idx = ActionEncoder::action_to_index(action);
            float logit = 0.0f;
            if (idx >= 0 && idx < (int)policy_logits.size()) {
                logit = policy_logits[idx];
            }
            float p = std::exp(logit);
            priors.push_back(p);
            sum_exp += p;
        }

        for (size_t i = 0; i < actions.size(); ++i) {
            const auto& action = actions[i];
            float p = priors[i] / sum_exp;
            
            GameState next_state = node->state.clone();
            // Replaced EffectResolver::resolve_action with GameLogicSystem::resolve_action
            GameLogicSystem::resolve_action(next_state, action, *card_db_);

            if (action.type == ActionType::PASS || action.type == ActionType::MANA_CHARGE) {
                PhaseManager::next_phase(next_state, *card_db_);
            }
            PhaseManager::fast_forward(next_state, *card_db_);
            
            auto child = std::make_unique<MCTSNode>(std::move(next_state)); // next_state is moved

            child->parent = node;
            child->action_from_parent = action;
            child->prior = p;
            
            node->children.push_back(std::move(child));
        }
    }

    MCTSNode* MCTS::select_leaf(MCTSNode* node) {
        while (node->is_expanded()) {
            MCTSNode* best_child = nullptr;
            float best_score = -std::numeric_limits<float>::infinity();

            for (const auto& child : node->children) {
                float q = child->value();
                
                // Risk-Aware Scoring: Score = Q - alpha * sigma
                if (alpha_ > 1e-5f && child->visit_count > 1) {
                    float sigma = child->std_dev();
                    q -= alpha_ * sigma;
                }

                float u = c_puct_ * child->prior * std::sqrt(static_cast<float>(node->visit_count + node->virtual_loss)) / (1.0f + child->visit_count + child->virtual_loss);
                
                float score = q + u;
                if (score > best_score) {
                    best_score = score;
                    best_child = child.get();
                }
            }
            
            if (!best_child) break;
            
            node->virtual_loss++;
            node = best_child;
        }
        
        node->virtual_loss++;
        return node;
    }

    void MCTS::revert_virtual_loss(MCTSNode* node) {
        while (node) {
            node->virtual_loss--;
            node = node->parent;
        }
    }

    void MCTS::backpropagate(MCTSNode* node, float value) {
        while (node) {
            node->visit_count++;

            // Variance accumulation: E[X^2]
            node->value_squared_sum += value * value;

            node->value_sum += value;

            // Invert value only if the active player changes
            if (node->parent && node->parent->state.active_player_id != node->state.active_player_id) {
                value = -value;
            }

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
