#include "phase_manager.hpp"
#include "engine/systems/mana/mana_system.hpp"
#include "engine/actions/action_generator.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/continuous_effect_system.hpp"
#include "core/constants.hpp"
#include <iostream>
#include <algorithm>

namespace dm::engine {

    using namespace dm::core;

    // Helper to move card via Command
    static void move_card_cmd(GameState& state, std::vector<CardInstance>& from, Zone from_zone, Zone to_zone, PlayerID owner) {
        if (from.empty()) return;
        int iid = from.back().instance_id;

        using namespace dm::engine::game_command;
        auto cmd = std::make_unique<TransitionCommand>(iid, from_zone, to_zone, owner, -1);
        state.execute_command(std::move(cmd));
    }

    // Fallback for direct modification during setup (where history is usually cleared/irrelevant)
    static void move_card_direct(std::vector<CardInstance>& from, std::vector<CardInstance>& to) {
        if (from.empty()) return;
        to.push_back(from.back());
        from.pop_back();
    }

    void PhaseManager::start_game(GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        game_state.turn_number = 1;
        game_state.active_player_id = 0;
        
        // Setup Shields (5 cards)
        for (auto& player : game_state.players) {
            for (int i = 0; i < 5; ++i) {
                if (player.deck.empty()) break;
                move_card_direct(player.deck, player.shield_zone);
            }
            // Draw Hand (5 cards)
            for (int i = 0; i < 5; ++i) {
                if (player.deck.empty()) break;
                move_card_direct(player.deck, player.hand);
            }
        }

        start_turn(game_state, card_db);
        // Initial state loop check
        game_state.update_loop_check();
    }

    void PhaseManager::start_turn(GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        Player& active_player = game_state.players[game_state.active_player_id];
        
        // Reset Turn Stats
        game_state.turn_stats = TurnStats{};

        // Untap (Using Command)
        ManaSystem::untap_all(game_state, active_player);

        // Clear Summoning Sickness
        for (auto& card : active_player.battle_zone) {
            card.summoning_sickness = false;
        }

        // Trigger Reservation: AT_START_OF_TURN
        for (const auto& card : active_player.battle_zone) {
            if (card_db.count(card.card_id)) {
                const auto& def = card_db.at(card.card_id);
                if (def.keywords.at_start_of_turn) {
                    game_state.pending_effects.emplace_back(EffectType::AT_START_OF_TURN, card.instance_id, active_player.id);
                }
            }
        }

        // Recalculate continuous effects at start of turn (refresh state)
        systems::ContinuousEffectSystem::recalculate(game_state, card_db);
        
        // Draw Phase
        bool skip_draw = (game_state.turn_number == 1 && game_state.active_player_id == 0);
        
        if (!skip_draw) {
            draw_card(game_state, active_player);
        }
    }

    void PhaseManager::draw_card(GameState& game_state, Player& player) {
        if (player.deck.empty()) {
            return;
        }
        move_card_cmd(game_state, player.deck, Zone::DECK, Zone::HAND, player.id);
    }

    void PhaseManager::fast_forward(GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        GameResult result;
        while (true) {
            if (check_game_over(game_state, result)) {
                return;
            }
            
            auto actions = ActionGenerator::generate_legal_actions(game_state, card_db);
            if (!actions.empty()) {
                return;
            }
            
            next_phase(game_state, card_db);
        }
    }

    bool PhaseManager::check_game_over(GameState& game_state, GameResult& result) {
        bool is_over = false;

        // Check Winner Flag
        if (game_state.winner != GameResult::NONE) {
            result = game_state.winner;
            is_over = true;
        } else {
            // Check Deck Out
            bool p1_deck_empty = game_state.players[0].deck.empty();
            bool p2_deck_empty = game_state.players[1].deck.empty();

            if (p1_deck_empty && p2_deck_empty) {
                result = GameResult::DRAW;
                game_state.winner = result;
                is_over = true;
            } else if (p1_deck_empty) {
                result = GameResult::P2_WIN;
                game_state.winner = result;
                is_over = true;
            } else if (p2_deck_empty) {
                result = GameResult::P1_WIN;
                game_state.winner = result;
                is_over = true;
            } else if (game_state.turn_number > TURN_LIMIT) {
                // Check Turn Limit
                result = GameResult::DRAW;
                game_state.winner = result;
                is_over = true;
            }
        }

        if (is_over) {
            if (!game_state.stats_recorded) {
                game_state.on_game_finished(result);
                game_state.stats_recorded = true;
            }
            return true;
        }

        result = GameResult::NONE;
        return false;
    }

    void PhaseManager::next_phase(GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        using namespace dm::engine::game_command;

        Phase next_p = game_state.current_phase; // Default same

        // Determine next phase
        switch (game_state.current_phase) {
            case Phase::START_OF_TURN:
                next_p = Phase::DRAW;
                break;
            case Phase::DRAW:
                next_p = Phase::MANA;
                break;
            case Phase::MANA:
                next_p = Phase::MAIN;
                break;
            case Phase::MAIN:
                next_p = Phase::ATTACK;
                break;
            case Phase::ATTACK:
                next_p = Phase::END_OF_TURN;
                break;
            case Phase::BLOCK:
                next_p = Phase::ATTACK;
                break;
            case Phase::END_OF_TURN:
                // Cleanup Step: Remove expired modifiers and passive effects
                {
                     auto& mods = game_state.active_modifiers;
                     mods.erase(std::remove_if(mods.begin(), mods.end(), [](CostModifier& m) {
                         if (m.turns_remaining > 0) m.turns_remaining--;
                         return m.turns_remaining == 0;
                     }), mods.end());

                     auto& passives = game_state.passive_effects;
                     passives.erase(std::remove_if(passives.begin(), passives.end(), [](PassiveEffect& p) {
                         if (p.turns_remaining > 0) p.turns_remaining--;
                         return p.turns_remaining == 0;
                     }), passives.end());
                }

                // Switch turn
                {
                     int next_active = 1 - game_state.active_player_id;
                     int next_turn = game_state.turn_number;
                     if (game_state.active_player_id == 0) {
                          next_turn++;
                     }

                     // FlowCommand for Turn Number
                     if (next_turn != game_state.turn_number) {
                         auto cmd_turn = std::make_unique<FlowCommand>(FlowCommand::FlowType::TURN_CHANGE, next_turn);
                         game_state.execute_command(std::move(cmd_turn));
                     }

                     // Direct modify active player (TODO: Command)
                     game_state.active_player_id = next_active;

                     next_p = Phase::START_OF_TURN;
                }
                break;
        }

        // Execute Phase Change Command
        if (next_p != game_state.current_phase) {
             auto cmd = std::make_unique<FlowCommand>(FlowCommand::FlowType::PHASE_CHANGE, static_cast<int>(next_p));
             game_state.execute_command(std::move(cmd));
        }

        // Post-transition logic (Triggers)
        if (game_state.current_phase == Phase::END_OF_TURN) {
             // 1. Generic Hand Trigger Check
             {
                Player& opponent = game_state.players[1 - game_state.active_player_id];
                for (const auto& card : opponent.hand) {
                    if (card_db.count(card.card_id)) {
                        const auto& def = card_db.at(card.card_id);

                        if (def.keywords.meta_counter_play) {
                            if (game_state.turn_stats.played_without_mana) {
                                game_state.pending_effects.emplace_back(EffectType::META_COUNTER, card.instance_id, opponent.id);
                            }
                        }

                        for (const auto& trigger : def.hand_triggers) {
                            bool condition_met = false;
                            if (trigger.condition.type == "OPPONENT_PLAYED_WITHOUT_MANA") {
                                if (game_state.turn_stats.played_without_mana) condition_met = true;
                            }

                            if (condition_met) {
                                 game_state.pending_effects.emplace_back(EffectType::META_COUNTER, card.instance_id, opponent.id);
                            }
                        }
                    }
                }
             }

             // 2. Trigger Reservation: AT_END_OF_TURN
             {
                Player& active_player = game_state.players[game_state.active_player_id];
                for (const auto& card : active_player.battle_zone) {
                    if (card_db.count(card.card_id)) {
                        const auto& def = card_db.at(card.card_id);
                        if (def.keywords.at_end_of_turn) {
                            game_state.pending_effects.emplace_back(EffectType::AT_END_OF_TURN, card.instance_id, active_player.id);
                        }
                    }
                }
             }
        } else if (game_state.current_phase == Phase::START_OF_TURN) {
             start_turn(game_state, card_db);
        }

        // Update loop check after phase transition
        game_state.update_loop_check();
    }

}
