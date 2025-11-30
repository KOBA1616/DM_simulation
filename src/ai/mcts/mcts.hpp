#pragma once
#include "../../core/game_state.hpp"
#include "../../core/card_def.hpp"
#include "../../core/action.hpp"
#include <vector>
#include <memory>
#include <functional>
#include <map>

namespace dm::ai {

    struct MCTSNode {
        dm::core::GameState state;
        MCTSNode* parent = nullptr;
        dm::core::Action action_from_parent; // The action that led to this state
        std::vector<std::unique_ptr<MCTSNode>> children;
        
        int visit_count = 0;
        float value_sum = 0.0f;
        float prior = 0.0f;

        MCTSNode(const dm::core::GameState& s) : state(s) {}
        
        bool is_expanded() const { return !children.empty(); }
        float value() const { return visit_count == 0 ? 0.0f : value_sum / visit_count; }
    };

    // Callback signature: Takes GameState, returns {PolicyVector, Value}
    using EvaluatorCallback = std::function<std::pair<std::vector<float>, float>(const dm::core::GameState&)>;

    class MCTS {
    public:
        MCTS(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, 
             float c_puct = 1.0f, 
             float dirichlet_alpha = 0.3f, 
             float dirichlet_epsilon = 0.25f);

        // Run MCTS search
        // Returns the policy vector
        std::vector<float> search(const dm::core::GameState& root_state, int simulations, EvaluatorCallback evaluator, bool add_noise = false, float temperature = 1.0f);

        // Helper to get action probabilities from the last search
        // (Usually called after search, but search returns policy vector directly for convenience)
        
    private:
        std::map<dm::core::CardID, dm::core::CardDefinition> card_db_;
        float c_puct_;
        float dirichlet_alpha_;
        float dirichlet_epsilon_;

        // Helpers
        float expand(MCTSNode* node, EvaluatorCallback evaluator);
        MCTSNode* select_child(MCTSNode* node);
        void backpropagate(MCTSNode* node, float value);
        void add_exploration_noise(MCTSNode* node);
    };

}
