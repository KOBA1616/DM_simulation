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
        int virtual_loss = 0; // For batch MCTS

        MCTSNode(const dm::core::GameState& s) : state(s) {}
        MCTSNode(const MCTSNode&) = delete;
        MCTSNode& operator=(const MCTSNode&) = delete;
        
        bool is_expanded() const { return !children.empty(); }
        
        float value() const { 
            int effective_visits = visit_count + virtual_loss;
            if (effective_visits == 0) return 0.0f;
            // Virtual loss acts as a penalty (assuming max value is 1.0)
            return (value_sum - (float)virtual_loss) / effective_visits; 
        }
    };

    // Batch Evaluator Callback: Takes vector of GameStates, returns {BatchPolicy, BatchValue}
    using BatchEvaluatorCallback = std::function<std::pair<std::vector<std::vector<float>>, std::vector<float>>(const std::vector<dm::core::GameState>&)>;

    class MCTS {
    public:
        MCTS(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, 
             float c_puct = 1.0f, 
             float dirichlet_alpha = 0.3f, 
             float dirichlet_epsilon = 0.25f,
             int batch_size = 1);

        // Run MCTS search
        std::vector<float> search(const dm::core::GameState& root_state, int simulations, BatchEvaluatorCallback evaluator, bool add_noise = false, float temperature = 1.0f);

        // Get the root of the last search tree (for visualization)
        const MCTSNode* get_last_root() const { return last_root_.get(); }

    private:
        std::map<dm::core::CardID, dm::core::CardDefinition> card_db_;
        float c_puct_;
        float dirichlet_alpha_;
        float dirichlet_epsilon_;
        int batch_size_;
        
        std::unique_ptr<MCTSNode> last_root_;

        // Helpers
        void expand_node(MCTSNode* node, const std::vector<float>& policy_logits);
        MCTSNode* select_leaf(MCTSNode* node); // Selects a leaf and applies virtual loss
        void backpropagate(MCTSNode* node, float value);
        void add_exploration_noise(MCTSNode* node);
        
        // Undo virtual loss along the path
        void revert_virtual_loss(MCTSNode* node);
    };

}
