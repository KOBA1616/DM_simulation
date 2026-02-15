#include "self_play.hpp"
#include "engine/systems/flow/phase_system.hpp"
#include "engine/systems/game_logic_system.hpp"
#include "engine/utils/determinizer.hpp"
#include "engine/actions/intent_generator.hpp"
#include "ai/encoders/action_encoder.hpp"
#include <iostream>
#include <random>

namespace dm::ai {

    using namespace dm::core;
    using namespace dm::engine;
    using namespace dm::engine::systems;

    SelfPlay::SelfPlay(const std::map<CardID, CardDefinition>& card_db, int mcts_simulations, int batch_size)
        : card_db_(std::make_shared<std::map<CardID, CardDefinition>>(card_db)),
          mcts_simulations_(mcts_simulations), batch_size_(batch_size) {}

    SelfPlay::SelfPlay(std::shared_ptr<const std::map<CardID, CardDefinition>> card_db, int mcts_simulations, int batch_size)
        : card_db_(card_db), mcts_simulations_(mcts_simulations), batch_size_(batch_size) {}

    void SelfPlay::set_pimc_generator(std::shared_ptr<dm::ai::inference::PimcGenerator> pimc_generator) {
        pimc_generator_ = pimc_generator;
    }

    GameResultInfo SelfPlay::play_game(const GameState& initial_state, BatchEvaluatorCallback evaluator, float temperature, bool add_noise, float alpha, bool collect_data) {
        GameResultInfo info;
        // GameState state = initial_state; // Copy -> Deleted
        GameState state = initial_state.clone();
        
        // MCTS instance with risk aversion coefficient alpha
        // Now MCTS accepts shared_ptr
        MCTS mcts(card_db_, 1.0f, 0.3f, 0.25f, batch_size_, alpha);

        if (pimc_generator_) {
            mcts.set_pimc_generator(pimc_generator_);
        }

        // Game Loop
        while (true) {
            GameResult result;
            // Check game over on current state.
            if (PhaseSystem::check_game_over(state, result)) {
                info.result = result;
                info.turn_count = state.turn_number;
                break;
            }

            // GameState search_state = state.clone();
            GameState search_state = state.clone();

            // If PIMC is NOT enabled, we must determinize the state (hidden information handling)
            // before passing it to MCTS, assuming perfect information search (Open Loop).
            // If PIMC IS enabled, MCTS handles the determinization/sampling internally via PimcGenerator.
            if (!pimc_generator_) {
                 Determinizer::determinize(search_state, state.active_player_id);
            }

            std::vector<float> policy = mcts.search(search_state, mcts_simulations_, evaluator, add_noise, temperature);

            if (collect_data) {
                // info.states.push_back(state); // Copy -> Deleted
                info.states.push_back(std::make_shared<dm::core::GameState>(state.clone()));
                info.policies.push_back(policy);
                info.active_players.push_back(state.active_player_id);
            }

            auto legal_actions = IntentGenerator::generate_legal_actions(state, *card_db_);
            if (legal_actions.empty()) {
                PhaseSystem::next_phase(state, *card_db_);
                continue;
            }

            int action_idx = -1;
            std::discrete_distribution<int> dist(policy.begin(), policy.end());
            action_idx = dist(state.rng);

            Action selected_action;
            bool found = false;
            std::vector<Action> candidates;
            
            for (const auto& act : legal_actions) {
                if (ActionEncoder::action_to_index(act) == action_idx) {
                    candidates.push_back(act);
                }
            }

            if (candidates.empty()) {
                if (!legal_actions.empty()) {
                    std::uniform_int_distribution<int> uni(0, legal_actions.size() - 1);
                    selected_action = legal_actions[uni(state.rng)];
                    found = true;
                }
            } else {
                selected_action = candidates[0];
                found = true;
            }

            if (!found) {
                PhaseSystem::next_phase(state, *card_db_);
                continue;
            }

            GameLogicSystem::resolve_action(state, selected_action, *card_db_);
            
            if (selected_action.type == PlayerIntent::PASS || selected_action.type == PlayerIntent::MANA_CHARGE) {
                if (state.current_phase == Phase::MANA && selected_action.type == PlayerIntent::MANA_CHARGE) {
                     PhaseSystem::next_phase(state, *card_db_);
                 } else if (selected_action.type == PlayerIntent::PASS) {
                     PhaseSystem::next_phase(state, *card_db_);
                 }
            }
        }

        return info;
    }

}
