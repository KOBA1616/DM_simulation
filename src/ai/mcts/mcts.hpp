#pragma once

#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "ai/inference/pimc_generator.hpp"
#include <vector>
#include <memory>
#include <map>
#include <cmath>
#include <functional>
#include "core/card_json_types.hpp" // Use CommandDef
#include "ai/mcts/tree_manager.hpp"

namespace dm::ai {

    // Define the callback type directly here or include it from a common place.
    // Since we don't have a batch_evaluator.hpp, we define it.
    using BatchEvaluatorCallback = std::function<std::pair<std::vector<std::vector<float>>, std::vector<float>>(const std::vector<std::shared_ptr<dm::core::GameState>>&)>;

    struct MCTSNode {
        dm::core::GameState state;
        MCTSNode* parent = nullptr;
        std::vector<std::unique_ptr<MCTSNode>> children;
        dm::core::CommandDef action_from_parent;
        
        int visit_count = 0;
        float value_sum = 0.0f;
        float value_squared_sum = 0.0f; // For variance calculation
        float prior = 0.0f;
        int virtual_loss = 0;

        MCTSNode(dm::core::GameState&& s) : state(std::move(s)) {}
        ~MCTSNode();

        bool is_expanded() const { return !children.empty(); }
        
        float value() const {
            if (visit_count == 0) return 0.0f;
            return value_sum / visit_count;
        }

        // Variance = E[X^2] - (E[X])^2
        float variance() const {
            if (visit_count == 0) return 0.0f;
            float mean = value();
            // Ensure non-negative due to float precision
            float var = (value_squared_sum / visit_count) - (mean * mean);
            return std::max(0.0f, var);
        }

        float std_dev() const {
            return std::sqrt(variance());
        }
    };

    class MCTS {
    public:
        MCTS(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db, 
             float c_puct = 1.0f, 
             float dirichlet_alpha = 0.3f, 
             float dirichlet_epsilon = 0.25f,
             int batch_size = 1,
             float alpha = 0.0f); // Risk aversion coefficient

        MCTS(std::shared_ptr<const std::map<dm::core::CardID, dm::core::CardDefinition>> card_db,
             float c_puct = 1.0f,
             float dirichlet_alpha = 0.3f,
             float dirichlet_epsilon = 0.25f,
             int batch_size = 1,
             float alpha = 0.0f);

        void set_pimc_generator(std::shared_ptr<dm::ai::inference::PimcGenerator> pimc);

        std::vector<float> search(const dm::core::GameState& root_state,
                                  int simulations,
                                  BatchEvaluatorCallback evaluator,
                                  bool add_noise = true,
                                  float temperature = 1.0f);

        void revert_virtual_loss(MCTSNode* node);

        // Accessor for bindings
        MCTSNode* get_last_root() { return last_root_.get(); }

    private:
        MCTSNode* select_leaf(MCTSNode* node);
        void expand_node(MCTSNode* node, const std::vector<float>& policy_logits);
        void backpropagate(MCTSNode* node, float value);
        void add_exploration_noise(MCTSNode* node);

        std::shared_ptr<const std::map<dm::core::CardID, dm::core::CardDefinition>> card_db_;
        std::shared_ptr<dm::ai::inference::PimcGenerator> pimc_generator_;
        float c_puct_;
        float dirichlet_alpha_;
        float dirichlet_epsilon_;
        int batch_size_;
        float alpha_; // Risk aversion coefficient
        std::unique_ptr<MCTSNode> last_root_;

        // PIMC parameters
        int pimc_samples_ = 4;

        TranspositionTable transposition_table_;
    };

}
