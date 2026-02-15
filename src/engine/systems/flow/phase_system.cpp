#include "phase_system.hpp"
#include "engine/infrastructure/commands/definitions/commands.hpp"
#include "engine/systems/mechanics/mana_system.hpp"
#include "engine/systems/effects/trigger_system.hpp"
#include "engine/systems/effects/continuous_effect_system.hpp"
#include "core/constants.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>

namespace dm::engine::systems {

    using namespace dm::core;
    using namespace dm::engine::game_command;

    // Helper to move card via Command
    static void move_card_cmd(GameState& state, std::vector<CardInstance>& from, Zone from_zone, Zone to_zone, PlayerID owner) {
        if (from.empty()) return;
        int iid = from.back().instance_id;
        auto cmd = std::make_unique<TransitionCommand>(iid, from_zone, to_zone, owner, -1);
        state.execute_command(std::move(cmd));
    }

    void PhaseSystem::start_game(GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        game_state.turn_number = 1;
        game_state.active_player_id = 0;

        // Setup Shields (5 cards)
        for (auto& player : game_state.players) {
            for (int i = 0; i < 5; ++i) {
                if (player.deck.empty()) break;
                move_card_cmd(game_state, player.deck, Zone::DECK, Zone::SHIELD, player.id);
            }
            // Draw Hand (5 cards)
            for (int i = 0; i < 5; ++i) {
                if (player.deck.empty()) break;
                move_card_cmd(game_state, player.deck, Zone::DECK, Zone::HAND, player.id);
            }
        }

        on_start_turn(game_state, card_db);
        // Initial state loop check
        game_state.update_loop_check();
    }

    void PhaseSystem::handle_pass(GameState& state, const std::map<CardID, CardDefinition>& card_db) {
        next_phase(state, card_db);
    }

    void PhaseSystem::on_start_turn(GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        Player& active_player = game_state.players[game_state.active_player_id];

        // Reset Turn Stats
        auto cmd_stats = std::make_shared<FlowCommand>(FlowCommand::FlowType::RESET_TURN_STATS, 0);
        game_state.execute_command(std::move(cmd_stats));

        // Untap (Using Command)
        ManaSystem::untap_all(game_state, active_player);

        // Clear Summoning Sickness
        for (auto& card : active_player.battle_zone) {
             auto cmd = std::make_shared<MutateCommand>(card.instance_id, MutateCommand::MutationType::SET_SUMMONING_SICKNESS, 0);
             game_state.execute_command(std::move(cmd));
        }

        // Trigger Reservation: AT_START_OF_TURN
        for (const auto& card : active_player.battle_zone) {
            if (card_db.count(card.card_id)) {
                const auto& def = card_db.at(card.card_id);
                if (def.keywords.at_start_of_turn) {
                    PendingEffect pe(EffectType::AT_START_OF_TURN, card.instance_id, active_player.id);
                    TriggerSystem::instance().add_pending_effect(game_state, pe);
                }
            }
        }

        // Recalculate continuous effects at start of turn (refresh state)
        ContinuousEffectSystem::recalculate(game_state, card_db);

        // Draw Phase logic is usually handled by next_phase transition into DRAW
    }

    void PhaseSystem::on_draw_phase(GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        (void)card_db;
        bool skip_draw = (game_state.turn_number == 1 && game_state.active_player_id == 0);

        if (!skip_draw) {
            Player& player = game_state.players[game_state.active_player_id];
            if (player.deck.empty()) {
                // Deck is empty, player loses immediately
                GameResult loss_result = (player.id == 0) ? GameResult::P2_WIN : GameResult::P1_WIN;
                auto cmd = std::make_unique<GameResultCommand>(loss_result);
                game_state.execute_command(std::move(cmd));
                return;
            }
            move_card_cmd(game_state, player.deck, Zone::DECK, Zone::HAND, player.id);
        }
    }

    void PhaseSystem::on_mana_phase(GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        // Nothing special happening automatically on Mana Phase entry
        (void)game_state; (void)card_db;
    }

    void PhaseSystem::on_end_turn(GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
         // Cleanup Step
        {
             auto cmd_cleanup = std::make_shared<FlowCommand>(FlowCommand::FlowType::CLEANUP_STEP, 0);
             game_state.execute_command(std::move(cmd_cleanup));
        }

        // Trigger Reservation: AT_END_OF_TURN
        {
            Player& active_player = game_state.players[game_state.active_player_id];
            for (const auto& card : active_player.battle_zone) {
                if (card_db.count(card.card_id)) {
                    const auto& def = card_db.at(card.card_id);
                    if (def.keywords.at_end_of_turn) {
                        PendingEffect pe(EffectType::AT_END_OF_TURN, card.instance_id, active_player.id);
                        TriggerSystem::instance().add_pending_effect(game_state, pe);
                    }
                }
            }
        }

        // Switch turn
        {
             int next_active = 1 - game_state.active_player_id;
             int next_turn = game_state.turn_number;
             if (game_state.active_player_id == 0) {
                  next_turn++;
             }

             if (next_turn != game_state.turn_number) {
                 auto cmd_turn = std::make_shared<FlowCommand>(FlowCommand::FlowType::TURN_CHANGE, next_turn);
                 game_state.execute_command(std::move(cmd_turn));
             }

             auto cmd_active = std::make_shared<FlowCommand>(FlowCommand::FlowType::SET_ACTIVE_PLAYER, next_active);
             game_state.execute_command(std::move(cmd_active));
        }
    }

    static bool check_game_over_internal(GameState& game_state) {
        bool is_over = false;
        GameResult result = GameResult::NONE;

        if (game_state.winner != GameResult::NONE) {
            result = game_state.winner;
            is_over = true;
        } else {
            bool p1_deck_empty = game_state.players[0].deck.empty();
            bool p2_deck_empty = game_state.players[1].deck.empty();

            if (p1_deck_empty && p2_deck_empty) {
                if (game_state.active_player_id == 0) {
                    result = GameResult::P1_WIN;
                    game_state.winner = result;
                } else {
                    result = GameResult::P2_WIN;
                    game_state.winner = result;
                }
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
        return false;
    }

    void PhaseSystem::next_phase(GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        Phase next_p = game_state.current_phase;

        switch (game_state.current_phase) {
            case Phase::START_OF_TURN: next_p = Phase::DRAW; break;
            case Phase::DRAW: next_p = Phase::MANA; break;
            case Phase::MANA: next_p = Phase::MAIN; break;
            case Phase::MAIN: next_p = Phase::ATTACK; break;
            case Phase::ATTACK: next_p = Phase::END_OF_TURN; break;
            case Phase::BLOCK:
                next_p = Phase::ATTACK;
                // Transitioning from BLOCK to ATTACK means we resolve the battle.
                // NOTE: This logic should ideally be in BattleSystem, but kept here for now to maintain flow.
                if (game_state.current_attack.source_instance_id != -1) {
                    bool blocked = game_state.current_attack.blocked;
                    int target_id = game_state.current_attack.target_instance_id;

                    if (blocked) {
                        PendingEffect pe(EffectType::RESOLVE_BATTLE, game_state.current_attack.source_instance_id, game_state.active_player_id);
                        pe.execution_context["attacker"] = game_state.current_attack.source_instance_id;
                        pe.execution_context["defender"] = game_state.current_attack.blocking_creature_id;
                        TriggerSystem::instance().add_pending_effect(game_state, pe);
                    } else {
                        if (target_id != -1) {
                            PendingEffect pe(EffectType::RESOLVE_BATTLE, game_state.current_attack.source_instance_id, game_state.active_player_id);
                            pe.execution_context["attacker"] = game_state.current_attack.source_instance_id;
                            pe.execution_context["defender"] = target_id;
                            TriggerSystem::instance().add_pending_effect(game_state, pe);
                        } else {
                            const Player& opponent = game_state.players[1 - game_state.active_player_id];
                            if (opponent.shield_zone.empty()) {
                                auto win_cmd = std::make_unique<GameResultCommand>(
                                    game_state.active_player_id == 0 ? GameResult::P1_WIN : GameResult::P2_WIN
                                );
                                game_state.execute_command(std::move(win_cmd));
                            } else {
                                PendingEffect pe(EffectType::BREAK_SHIELD, game_state.current_attack.source_instance_id, game_state.active_player_id);
                                TriggerSystem::instance().add_pending_effect(game_state, pe);
                            }
                        }
                    }
                }
                break;
            case Phase::END_OF_TURN:
                on_end_turn(game_state, card_db);
                next_p = Phase::START_OF_TURN;
                break;
        }

        if (next_p != game_state.current_phase) {
             auto cmd = std::make_shared<FlowCommand>(FlowCommand::FlowType::PHASE_CHANGE, static_cast<int>(next_p));
             game_state.execute_command(std::move(cmd));
        }

        // Auto-actions on entering phase
        if (game_state.current_phase == Phase::START_OF_TURN) {
            on_start_turn(game_state, card_db);
            // Immediately advance to DRAW phase after start_turn processing
            auto cmd_draw = std::make_shared<FlowCommand>(FlowCommand::FlowType::PHASE_CHANGE, static_cast<int>(Phase::DRAW));
            game_state.execute_command(std::move(cmd_draw));
            // And execute Draw Logic
            on_draw_phase(game_state, card_db);
        } else if (game_state.current_phase == Phase::DRAW) {
             on_draw_phase(game_state, card_db);
        }

        check_game_over_internal(game_state);
        game_state.update_loop_check();
    }

}
