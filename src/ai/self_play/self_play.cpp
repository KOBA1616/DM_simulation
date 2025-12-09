#include "self_play.hpp"
#include "../../engine/systems/flow/phase_manager.hpp"
#include "../../engine/effects/effect_resolver.hpp"
#include "../../engine/utils/determinizer.hpp"
#include "../../engine/actions/action_generator.hpp"
#include "../encoders/action_encoder.hpp"
#include <iostream>
#include <random>

namespace dm::ai {

    using namespace dm::core;
    using namespace dm::engine;

    SelfPlay::SelfPlay(const std::map<CardID, CardDefinition>& card_db, int mcts_simulations, int batch_size)
        : card_db_(card_db), mcts_simulations_(mcts_simulations), batch_size_(batch_size) {}

    GameResultInfo SelfPlay::play_game(GameState initial_state, BatchEvaluatorCallback evaluator, float temperature, bool add_noise) {
        GameResultInfo info;
        GameState state = initial_state;
        
        // MCTS instance
        // We create a new MCTS for each game or reuse? 
        // MCTS class resets tree on search usually, but let's create one per game to be safe or per move?
        // MCTS::search resets the tree if we don't pass a root.
        // But here we want to reuse the MCTS object configuration.
        MCTS mcts(card_db_, 1.0f, 0.3f, 0.25f, batch_size_);

        // Game Loop
        while (true) {
            // Check Game Over
            GameResult result;
            if (PhaseManager::check_game_over(state, result)) {
                info.result = result;
                info.turn_count = state.turn_number;
                break;
            }

            // Determinize for the active player
            GameState search_state = state; // Copy
            Determinizer::determinize(search_state, state.active_player_id);

            // Run MCTS
            // Note: MCTS search takes a root state and builds a tree.
            // It returns the policy for the root state.
            std::vector<float> policy = mcts.search(search_state, mcts_simulations_, evaluator, add_noise, temperature);

            // Store data
            info.states.push_back(state); // Store the TRUE state (not determinized) for training? 
            // Actually AlphaZero usually stores the state as seen by the player, but here we have perfect info in replay buffer?
            // No, we store the true state, and masking happens during training data generation (Tensor conversion).
            info.policies.push_back(policy);
            info.active_players.push_back(state.active_player_id);

            // Select Action based on policy
            // Policy is a probability distribution over ALL actions (fixed size).
            // We need to sample from it.
            
            // 1. Filter legal actions to map indices back to Action objects
            auto legal_actions = ActionGenerator::generate_legal_actions(state, card_db_);
            if (legal_actions.empty()) {
                // Auto-Pass if no actions? Or should have been handled by PhaseManager?
                // If no actions, PhaseManager::next_phase should be called.
                // But MCTS search should have handled it?
                // If MCTS returns a policy, it implies there were legal actions.
                // If no legal actions, MCTS might return empty or uniform?
                // Let's assume PhaseManager handles empty actions by auto-passing in the loop?
                // No, PhaseManager::check_game_over doesn't advance phase.
                // We need to check if we need to advance phase.
                
                // If no legal actions, we must pass or next phase.
                // But wait, ActionGenerator usually returns at least PASS or something.
                // If truly empty, we force next phase.
                PhaseManager::next_phase(state, card_db_);
                continue;
            }

            // Sample action index
            int action_idx = -1;
            
            // Argmax or Sample?
            // Training: Sample based on temperature.
            // Eval: Argmax.
            // Here we use the policy returned by MCTS which already incorporates temperature in visit counts?
            // MCTS::search returns visits/sum(visits).
            // If temperature is applied inside MCTS search result (exponentiated), we can sample.
            // Usually MCTS returns raw probabilities (visits).
            // We should sample from this distribution.
            
            std::discrete_distribution<int> dist(policy.begin(), policy.end());
            // We need a random generator.
            // Use state.rng? Or a local one?
            // Better use a local one for sampling actions to not mess with game state determinism if we want to replay?
            // But we want reproducibility. Use state.rng.
            action_idx = dist(state.rng);

            // Find the corresponding Action object
            // This is tricky because ActionEncoder maps Action -> Index (Many-to-One possible? No, usually One-to-One or Many-to-One).
            // But Index -> Action is One-to-Many if we don't have context.
            // We have legal_actions. We can iterate them and find which one maps to action_idx.
            
            Action selected_action;
            bool found = false;
            std::vector<Action> candidates;
            
            for (const auto& act : legal_actions) {
                if (ActionEncoder::action_to_index(act) == action_idx) {
                    candidates.push_back(act);
                }
            }

            if (candidates.empty()) {
                // Should not happen if policy > 0 only for legal actions
                // Fallback: Pick random legal action
                if (!legal_actions.empty()) {
                    std::uniform_int_distribution<int> uni(0, legal_actions.size() - 1);
                    selected_action = legal_actions[uni(state.rng)];
                    found = true;
                }
            } else {
                // If multiple actions map to same index (e.g. "Attack Player" might be same index for different source cards if encoding is simplified? No, encoding usually includes source.)
                // Assuming 1-to-1 for now or just pick first.
                selected_action = candidates[0];
                found = true;
            }

            if (!found) {
                // Force next phase
                PhaseManager::next_phase(state, card_db_);
                continue;
            }

            // Execute Action
            EffectResolver::resolve_action(state, selected_action, card_db_);
            
            // Check if we need to advance phase (Pass/Charge)
            if (selected_action.type == ActionType::PASS || selected_action.type == ActionType::MANA_CHARGE) {
                 // For Mana Charge, we might want to allow multiple charges? 
                 // Rules: 1 charge per turn.
                 // If we charged, we usually pass or move to next step?
                 // Standard: Charge -> Main.
                 // Our ActionGenerator generates Charge actions in MANA phase.
                 // After Charge, we should probably auto-transition or allow undo?
                 // AI doesn't undo.
                 // If we charged, we are done with Mana phase?
                 // Usually yes.
                 if (state.current_phase == Phase::MANA && selected_action.type == ActionType::MANA_CHARGE) {
                     PhaseManager::next_phase(state, card_db_);
                 } else if (selected_action.type == ActionType::PASS) {
                     PhaseManager::next_phase(state, card_db_);
                 }
            }
        }

        return info;
    }

}
