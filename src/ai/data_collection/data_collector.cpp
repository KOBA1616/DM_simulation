#include "data_collector.hpp"
#include "ai/agents/heuristic_agent.hpp"
#include "engine/game_instance.hpp"
#include "engine/systems/flow/phase_manager.hpp"
#include "engine/systems/card/card_registry.hpp"
#include "engine/actions/intent_generator.hpp"
#include "engine/systems/game_logic_system.hpp"
#include "ai/encoders/action_encoder.hpp"
#include "ai/encoders/token_converter.hpp"
#include <iostream>
#include <chrono>

namespace dm::ai {

    using namespace dm::engine::systems;

    DataCollector::DataCollector(const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db)
        : card_db_(std::make_shared<std::map<dm::core::CardID, dm::core::CardDefinition>>(card_db)) {}

    DataCollector::DataCollector(std::shared_ptr<const std::map<dm::core::CardID, dm::core::CardDefinition>> card_db)
        : card_db_(card_db) {}

    DataCollector::DataCollector()
        : card_db_(dm::engine::CardRegistry::get_all_definitions_ptr()) {}

    CollectedBatch DataCollector::collect_data_batch(int episodes) {
        return collect_data_batch_heuristic(episodes);
    }

    CollectedBatch DataCollector::collect_data_batch_heuristic(int episodes) {
        CollectedBatch batch;

        // Instantiate agents once outside the loop to reuse RNG initialization
        HeuristicAgent agent1(0, *card_db_);
        HeuristicAgent agent2(1, *card_db_);

        // Pre-calculate decks (using a simple strategy if none provided, but here we assume all cards are available)
        // In a real scenario, we might want to randomize decks or pass them in.
        // For now, we construct a simple deck from the DB or default dummy.
        std::vector<dm::core::CardID> deck1, deck2;
        if (card_db_ && !card_db_->empty()) {
            std::vector<dm::core::CardID> available_ids;
            for (const auto& kv : *card_db_) {
                available_ids.push_back(kv.first);
            }
            for (int k = 0; k < 40; ++k) {
                deck1.push_back(available_ids[k % available_ids.size()]);
                deck2.push_back(available_ids[(k + 1) % available_ids.size()]);
            }
        } else {
                deck1.assign(40, 1);
                deck2.assign(40, 1);
        }

        for (int i = 0; i < episodes; ++i) {
            // Setup game
            // Use random seed based on time/iteration
            uint32_t seed = static_cast<uint32_t>(i) + std::chrono::system_clock::now().time_since_epoch().count();
            dm::engine::GameInstance game(seed, card_db_);

            // Manually setup decks
            int instance_counter = 0;
            auto setup_deck = [&](dm::core::Player& p, const std::vector<dm::core::CardID>& deck_list) {
                p.deck.clear();
                for(auto cid : deck_list) {
                    p.deck.emplace_back(cid, instance_counter++, p.id);
                }
            };

            setup_deck(game.state.players[0], deck1);
            setup_deck(game.state.players[1], deck2);

            // Game Loop
            dm::engine::PhaseManager::start_game(game.state, *card_db_);

            std::vector<std::vector<long>> game_states;
            std::vector<std::vector<float>> game_policies;
            std::vector<std::vector<float>> game_masks;
            std::vector<int> game_players;

            int max_steps = 1000;
            int step = 0;

            while (game.state.winner == dm::core::GameResult::NONE && step < max_steps) {
                step++;
                int active_player = game.state.active_player_id;

                dm::engine::IntentGenerator action_gen;
                auto legal_actions = action_gen.generate_legal_actions(game.state, *card_db_);

                // If no actions, transition phase
                if (legal_actions.empty()) {
                     dm::engine::PhaseManager::next_phase(game.state, *card_db_);

                     dm::core::GameResult res;
                     if(dm::engine::PhaseManager::check_game_over(game.state, res)){
                        game.state.winner = res;
                     }
                     continue;
                }

                // Choose action
                dm::core::Action chosen_action;
                if (active_player == 0) {
                    chosen_action = agent1.get_action(game.state, legal_actions);
                } else {
                    chosen_action = agent2.get_action(game.state, legal_actions);
                }

                // Record data
                std::vector<int> tokens_int = encoders::TokenConverter::encode_state(game.state, active_player, 0);
                std::vector<long> state_seq(tokens_int.begin(), tokens_int.end());

                std::vector<float> policy(ActionEncoder::TOTAL_ACTION_SIZE, 0.0f);
                int action_idx = ActionEncoder::action_to_index(chosen_action);
                if (action_idx >= 0 && action_idx < ActionEncoder::TOTAL_ACTION_SIZE) {
                    policy[action_idx] = 1.0f;
                }

                std::vector<float> mask(ActionEncoder::TOTAL_ACTION_SIZE, 0.0f);
                for (const auto& act : legal_actions) {
                    int idx = ActionEncoder::action_to_index(act);
                    if (idx >= 0 && idx < ActionEncoder::TOTAL_ACTION_SIZE) {
                        mask[idx] = 1.0f;
                    }
                }

                game_states.push_back(state_seq);
                game_policies.push_back(policy);
                game_masks.push_back(mask);
                game_players.push_back(active_player);

                // Apply action
                GameLogicSystem::resolve_action(game.state, chosen_action, *card_db_);

                if (chosen_action.type == dm::core::PlayerIntent::PASS) {
                     dm::engine::PhaseManager::next_phase(game.state, *card_db_);
                }

                game.state.update_loop_check();
                if (game.state.loop_proven) {
                     game.state.winner = dm::core::GameResult::DRAW;
                     break;
                }

                dm::core::GameResult res;
                if(dm::engine::PhaseManager::check_game_over(game.state, res)){
                    game.state.winner = res;
                }
            }

            // Game Finished. Assign values.
            float result_p0 = 0.0f;
            float result_p1 = 0.0f;

            if (game.state.winner == dm::core::GameResult::P1_WIN) {
                result_p0 = 1.0f;
                result_p1 = -1.0f;
            } else if (game.state.winner == dm::core::GameResult::P2_WIN) {
                result_p0 = -1.0f;
                result_p1 = 1.0f;
            }

            for (size_t k = 0; k < game_states.size(); ++k) {
                batch.states.push_back(game_states[k]);
                batch.policies.push_back(game_policies[k]);
                batch.masks.push_back(game_masks[k]);

                if (game_players[k] == 0) {
                    batch.values.push_back(result_p0);
                } else {
                    batch.values.push_back(result_p1);
                }
            }
        }

        return batch;
    }

}
