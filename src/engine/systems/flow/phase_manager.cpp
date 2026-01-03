#include "phase_manager.hpp"
#include "engine/systems/mana/mana_system.hpp"
#include "engine/actions/intent_generator.hpp"
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
                move_card_cmd(game_state, player.deck, Zone::DECK, Zone::SHIELD, player.id);
            }
            // Draw Hand (5 cards)
            for (int i = 0; i < 5; ++i) {
                if (player.deck.empty()) break;
                move_card_cmd(game_state, player.deck, Zone::DECK, Zone::HAND, player.id);
            }
        }

        start_turn(game_state, card_db);
        // Initial state loop check
        game_state.update_loop_check();
    }

    void PhaseManager::setup_scenario(GameState& state, const ScenarioConfig& config, const std::map<CardID, CardDefinition>& card_db) {
        // 1. Reset Game State
        state.turn_number = 5;
        state.active_player_id = 0;
        state.current_phase = Phase::MAIN;
        state.winner = GameResult::NONE;
        state.pending_effects.clear();
        state.current_attack = AttackState(); // Reset attack context

        // Clear all zones for both players
        for (auto& p : state.players) {
            p.hand.clear();
            p.battle_zone.clear();
            p.mana_zone.clear();
            p.graveyard.clear();
            p.shield_zone.clear();
            p.deck.clear();
        }

        // Instance ID counter
        int instance_id_counter = 0;

        // Fill decks
        // Player 0 (Me)
        if (!config.my_deck.empty()) {
             for (int cid : config.my_deck) {
                  state.players[0].deck.emplace_back((CardID)cid, instance_id_counter++, (PlayerID)0);
             }
        } else {
             // Fallback to dummy deck
             for(int i=0; i<30; ++i) {
                  state.players[0].deck.emplace_back((CardID)1, instance_id_counter++, (PlayerID)0);
             }
        }

        // Player 1 (Enemy)
        if (!config.enemy_deck.empty()) {
             for (int cid : config.enemy_deck) {
                  state.players[1].deck.emplace_back((CardID)cid, instance_id_counter++, (PlayerID)1);
             }
        } else {
             // Fallback to dummy deck
             for(int i=0; i<30; ++i) {
                  state.players[1].deck.emplace_back((CardID)1, instance_id_counter++, (PlayerID)1);
             }
        }

        // 2. Setup My Resources (Player 0)
        Player& me = state.players[0];

        // Hand
        for (int cid : config.my_hand_cards) {
            me.hand.emplace_back((CardID)cid, instance_id_counter++, me.id);
        }

        // Battle Zone
        for (int cid : config.my_battle_zone) {
            CardInstance c((CardID)cid, instance_id_counter++, me.id);
            c.summoning_sickness = false; // Assume creatures on board are ready
            me.battle_zone.push_back(c);
        }

        // Mana Zone
        for (int cid : config.my_mana_zone) {
            CardInstance c((CardID)cid, instance_id_counter++, me.id);
            c.is_tapped = false;
            me.mana_zone.push_back(c);
        }

        if (config.my_mana_zone.empty() && config.my_mana > 0) {
            for (int i = 0; i < config.my_mana; ++i) {
                me.mana_zone.emplace_back(1, instance_id_counter++, me.id);
            }
        }

        // Graveyard
        for (int cid : config.my_grave_yard) {
            me.graveyard.emplace_back((CardID)cid, instance_id_counter++, me.id);
        }

        // My Shields (Player 0)
        for (int cid : config.my_shields) {
             me.shield_zone.emplace_back((CardID)cid, instance_id_counter++, me.id);
        }

        // 3. Setup Enemy Resources (Player 1)
        Player& enemy = state.players[1];

        // Enemy Battle Zone
        for (int cid : config.enemy_battle_zone) {
            CardInstance c((CardID)cid, instance_id_counter++, enemy.id);
            c.summoning_sickness = false;
            enemy.battle_zone.push_back(c);
        }

        // Enemy Shields
        for (int i = 0; i < config.enemy_shield_count; ++i) {
             enemy.shield_zone.emplace_back(1, instance_id_counter++, enemy.id);
        }

        // Initialize Owner Map
        state.card_owner_map.resize(instance_id_counter);

        // Populate owner map
        auto populate_owner = [&](const Player& p) {
            auto register_cards = [&](const std::vector<CardInstance>& cards) {
                for (const auto& c : cards) {
                    if (c.instance_id >= 0 && c.instance_id < (int)state.card_owner_map.size()) {
                        state.card_owner_map[c.instance_id] = p.id;
                    }
                }
            };
            register_cards(p.hand);
            register_cards(p.battle_zone);
            register_cards(p.mana_zone);
            register_cards(p.graveyard);
            register_cards(p.shield_zone);
            register_cards(p.deck);
            register_cards(p.hyper_spatial_zone);
            register_cards(p.gr_deck);
        };

        populate_owner(state.players[0]);
        populate_owner(state.players[1]);

        // Initialize Stats (Optional but good practice)
        state.initialize_card_stats(card_db, 40); // Approx
    }

    void PhaseManager::start_turn(GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        Player& active_player = game_state.players[game_state.active_player_id];
        
        // Reset Turn Stats
        using namespace dm::engine::game_command;
        auto cmd_stats = std::make_unique<FlowCommand>(FlowCommand::FlowType::RESET_TURN_STATS, 0);
        game_state.execute_command(std::move(cmd_stats));

        // Untap (Using Command)
        ManaSystem::untap_all(game_state, active_player);

        // Clear Summoning Sickness
        for (auto& card : active_player.battle_zone) {
             auto cmd = std::make_unique<MutateCommand>(card.instance_id, MutateCommand::MutationType::SET_SUMMONING_SICKNESS, 0);
             game_state.execute_command(std::move(cmd));
        }

        // Trigger Reservation: AT_START_OF_TURN
        for (const auto& card : active_player.battle_zone) {
            if (card_db.count(card.card_id)) {
                const auto& def = card_db.at(card.card_id);
                if (def.keywords.at_start_of_turn) {
                    // Use Command
                    PendingEffect pe(EffectType::AT_START_OF_TURN, card.instance_id, active_player.id);
                    auto cmd = std::make_unique<game_command::MutateCommand>(-1, game_command::MutateCommand::MutationType::ADD_PENDING_EFFECT);
                    cmd->pending_effect = pe;
                    game_state.execute_command(std::move(cmd));
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
            
            auto actions = IntentGenerator::generate_legal_actions(game_state, card_db);
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
            // std::cerr << "Game Over. Result: " << (int)result << ". Turn: " << game_state.turn_number << std::endl;
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

                // Transitioning from BLOCK to ATTACK means we resolve the battle.
                if (game_state.current_attack.source_instance_id != -1) {
                    bool blocked = game_state.current_attack.blocked;
                    int target_id = game_state.current_attack.target_instance_id;

                    if (blocked) {
                        // If blocked, it's a battle between Attacker and Blocker
                        PendingEffect pe(EffectType::RESOLVE_BATTLE, game_state.current_attack.source_instance_id, game_state.active_player_id);
                        pe.execution_context["attacker"] = game_state.current_attack.source_instance_id;
                        pe.execution_context["defender"] = game_state.current_attack.blocking_creature_id;

                        auto cmd = std::make_unique<game_command::MutateCommand>(-1, game_command::MutateCommand::MutationType::ADD_PENDING_EFFECT);
                        cmd->pending_effect = pe;
                        game_state.execute_command(std::move(cmd));

                    } else {
                        // Not blocked
                        if (target_id != -1) {
                            // Creature vs Creature
                            PendingEffect pe(EffectType::RESOLVE_BATTLE, game_state.current_attack.source_instance_id, game_state.active_player_id);
                            pe.execution_context["attacker"] = game_state.current_attack.source_instance_id;
                            pe.execution_context["defender"] = target_id;

                            auto cmd = std::make_unique<game_command::MutateCommand>(-1, game_command::MutateCommand::MutationType::ADD_PENDING_EFFECT);
                            cmd->pending_effect = pe;
                            game_state.execute_command(std::move(cmd));
                        } else {
                            // Creature vs Player
                            const Player& opponent = game_state.players[1 - game_state.active_player_id];
                            if (opponent.shield_zone.empty()) {
                                // Direct Attack -> WIN
                                auto win_cmd = std::make_unique<GameResultCommand>(
                                    game_state.active_player_id == 0 ? GameResult::P1_WIN : GameResult::P2_WIN
                                );
                                game_state.execute_command(std::move(win_cmd));
                            } else {
                                // Shield Break
                                PendingEffect pe(EffectType::BREAK_SHIELD, game_state.current_attack.source_instance_id, game_state.active_player_id);

                                auto cmd = std::make_unique<game_command::MutateCommand>(-1, game_command::MutateCommand::MutationType::ADD_PENDING_EFFECT);
                                cmd->pending_effect = pe;
                                game_state.execute_command(std::move(cmd));
                            }
                        }
                    }
                }
                break;
            case Phase::END_OF_TURN:
                // Cleanup Step
                {
                     auto cmd_cleanup = std::make_unique<FlowCommand>(FlowCommand::FlowType::CLEANUP_STEP, 0);
                     game_state.execute_command(std::move(cmd_cleanup));
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

                     // FlowCommand for Active Player
                     auto cmd_active = std::make_unique<FlowCommand>(FlowCommand::FlowType::SET_ACTIVE_PLAYER, next_active);
                     game_state.execute_command(std::move(cmd_active));

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
                                PendingEffect pe(EffectType::META_COUNTER, card.instance_id, opponent.id);
                                auto cmd = std::make_unique<game_command::MutateCommand>(-1, game_command::MutateCommand::MutationType::ADD_PENDING_EFFECT);
                                cmd->pending_effect = pe;
                                game_state.execute_command(std::move(cmd));
                            }
                        }

                        for (const auto& trigger : def.hand_triggers) {
                            bool condition_met = false;
                            if (trigger.condition.type == "OPPONENT_PLAYED_WITHOUT_MANA") {
                                if (game_state.turn_stats.played_without_mana) condition_met = true;
                            }

                            if (condition_met) {
                                 PendingEffect pe(EffectType::META_COUNTER, card.instance_id, opponent.id);
                                 auto cmd = std::make_unique<game_command::MutateCommand>(-1, game_command::MutateCommand::MutationType::ADD_PENDING_EFFECT);
                                 cmd->pending_effect = pe;
                                 game_state.execute_command(std::move(cmd));
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
                            PendingEffect pe(EffectType::AT_END_OF_TURN, card.instance_id, active_player.id);
                            auto cmd = std::make_unique<game_command::MutateCommand>(-1, game_command::MutateCommand::MutationType::ADD_PENDING_EFFECT);
                            cmd->pending_effect = pe;
                            game_state.execute_command(std::move(cmd));
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
