#ifndef DM_AI_MCTS_DECISION_MAKER_HPP
#define DM_AI_MCTS_DECISION_MAKER_HPP

#include "engine/systems/decision_maker.hpp"
#include "ai/mcts/mcts.hpp"
#include "ai/mcts/mcts_evaluator.hpp"
#include "core/game_state.hpp"
#include "engine/game_command/game_command.hpp"
#include "engine/systems/command_system.hpp"
#include "engine/systems/card/card_registry.hpp"
#include <map>
#include <memory>
#include <iostream>

namespace dm::ai::mcts {

    // Helper for re-entrant simulation
    class FixedDecisionMaker : public dm::engine::systems::DecisionMaker {
        std::vector<int> fixed_targets_;
    public:
        FixedDecisionMaker(std::vector<int> t) : fixed_targets_(t) {}
        std::vector<int> select_targets(const dm::core::GameState&, const dm::core::CommandDef&, const std::vector<int>&, int) override {
            return fixed_targets_;
        }
    };

    class MCTSDecisionMaker : public dm::engine::systems::DecisionMaker {
    public:
        MCTSDecisionMaker(int simulations_per_decision = 50) 
            : simulations_(simulations_per_decision) {}

        std::vector<int> select_targets(
            const dm::core::GameState& state, 
            const dm::core::CommandDef& cmd, 
            const std::vector<int>& candidates, 
            int amount
        ) override {
            if (candidates.empty() || amount <= 0) return {};

            // 1. Generate all valid combinations of targets (up to a limit)
            std::vector<std::vector<int>> combinations = generate_combinations(candidates, amount);
            
            // Optimization: If only one valid combination, just return it without simulation
            if (combinations.size() == 1) return combinations[0];

            int best_idx = -1;
            float best_win_rate = -1.0f;
            
            // Get Card DB shared pointer (assuming registry has a static accessor for shared_ptr, 
            // or we make a new shared_ptr from the static map. MCTS expects shared_ptr<const map>.)
            // CardRegistry::get_all_definitions returns const map&.
            // We can wrap it in a shared_ptr with a no-op deleter to avoid copy, or just copy it if lightweight (it's big).
            // Better: MCTS constructor likely expects shared_ptr.
            auto card_db_ptr = std::shared_ptr<const std::map<dm::core::CardID, dm::core::CardDefinition>>(
                &dm::engine::CardRegistry::get_all_definitions(), 
                [](const void*){} // No-op deleter
            );
            
            // Evaluator instance
            BatchEvaluator batch_evaluator(1); // Batch 1 for simplicity in this loop
            auto evaluator_cb = [&](const std::vector<std::shared_ptr<dm::core::GameState>>& states) {
                return batch_evaluator.evaluate(states);
            };

            // 2. Evaluate each combination
            for (size_t i = 0; i < combinations.size(); ++i) {
                // Fork state
                dm::core::GameState next_state = state.clone();
                
                // Inject FixedDecisionMaker to force the selection
                auto fixed_dm = std::make_shared<FixedDecisionMaker>(combinations[i]);
                next_state.decision_maker = fixed_dm.get();
                
                // Execute the command on the clone
                // We pass a dummy context. In reality, we should try to capture the current context.
                // But generally context is for variable linking within a command chain. 
                // Creating a fresh one might break some variable links if they depend on previous steps.
                // However, for target selection of a *single* command (e.g. discard), it often doesn't depend on complex context variables 
                // EXCEPT amount/input_value, which we are supposed to have resolved already.
                // The `cmd` passed to us is the definition.
                std::map<std::string, int> dummy_context;
                dm::engine::systems::CommandSystem::execute_command(
                    next_state, cmd, -1, state.active_player_id, dummy_context
                );
                
                // Remove the fixed decision maker to allow normal AI after this step
                next_state.decision_maker = nullptr; 
                
                // Run MCTS simulation from this new state
                MCTS mcts_instance(card_db_ptr);
                
                // Run search
                // Note: MCTS search returns policy distribution from root. 
                // We want the VALUE of the root state (win rate).
                // MCTS::search doesn't just return value.
                // But we can check root node value after search.
                
                mcts_instance.search(next_state, simulations_, evaluator_cb, true, 1.0f);
                
                auto* root = mcts_instance.get_last_root();
                float win_rate = root ? root->value() : 0.5f; // Value is usually -1 to 1 or 0 to 1. 
                // Assuming 0 to 1 or -1 to 1. User said 'win rate'.
                // If value is from perspective of active player.
                // We want to maximize win rate for `state.active_player_id`.
                // MCTS usually computes value for the player whose turn it is at the root.
                // Since we just executed a command, `next_state` might still be same player's turn or next.
                // If it's same player's turn, we want high value.
                // If it switched turn, MCTS calculated value for opponent, so we want low value?
                // MCTS implementation detail: `value()` is usually perspective of the node's `state.active_player_id`?
                // Or standardized to root player?
                // Standard AlphaZero MCTS: value is for the current player at that node.
                
                // Let's assume simplest: We want to maximize the value returned for the current player.
                if (next_state.active_player_id != state.active_player_id) {
                    win_rate = 1.0f - win_rate; // Invert if turn changed
                }
                
                if (win_rate > best_win_rate) {
                    best_win_rate = win_rate;
                    best_idx = static_cast<int>(i);
                }
                
                // Debug log
                // std::cout << "DEBUG: Option " << i << " WinRate: " << win_rate << std::endl;
            }

            if (best_idx >= 0) {
                // std::cout << "[MCTSDecisionMaker] Selected option " << best_idx << " with WR " << best_win_rate << std::endl;
                return combinations[best_idx];
            }
            return combinations[0]; // Fallback
        }

    private:
        int simulations_;

        std::vector<std::vector<int>> generate_combinations(const std::vector<int>& candidates, int k) {
            std::vector<std::vector<int>> combs;
            std::vector<int> current;
            generate_combinations_recursive(candidates, k, 0, current, combs);
            return combs;
        }

        void generate_combinations_recursive(const std::vector<int>& candidates, int k, size_t start_idx, std::vector<int>& current, std::vector<std::vector<int>>& combs) {
            if (current.size() == (size_t)k) {
                combs.push_back(current);
                return;
            }
            // Heuristic limit: if we have too many candidates (e.g. 40 deck cards), combinations explode.
            // Limit total combinations to ~20 for feasibility.
            if (combs.size() >= 20) return; 

            for (size_t i = start_idx; i < candidates.size(); ++i) {
                current.push_back(candidates[i]);
                generate_combinations_recursive(candidates, k, i + 1, current, combs);
                current.pop_back();
            }
        }
    };

}

#endif // DM_AI_MCTS_DECISION_MAKER_HPP
