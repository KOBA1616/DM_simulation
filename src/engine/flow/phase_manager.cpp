#include "phase_manager.hpp"
#include "../mana/mana_system.hpp"
#include "../action_gen/action_generator.hpp"
#include "../../core/constants.hpp"
#include <iostream>

namespace dm::engine {

    using namespace dm::core;

    // Helper to move card from one vector to another
    static void move_card(std::vector<CardInstance>& from, std::vector<CardInstance>& to) {
        if (from.empty()) return;
        to.push_back(from.back());
        from.pop_back();
    }

    void PhaseManager::start_game(GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        game_state.turn_number = 1;
        game_state.active_player_id = 0;
        // Initial setup (shields, hand) would be done elsewhere or here?
        // Usually setup is done before start_game loop.
        // Let's assume decks are shuffled and ready.
        
        // Setup Shields (5 cards)
        for (auto& player : game_state.players) {
            for (int i = 0; i < 5; ++i) {
                if (player.deck.empty()) break; // Should not happen
                move_card(player.deck, player.shield_zone);
            }
            // Draw Hand (5 cards)
            for (int i = 0; i < 5; ++i) {
                if (player.deck.empty()) break;
                move_card(player.deck, player.hand);
            }
        }

        start_turn(game_state, card_db);
    }

    void PhaseManager::start_turn(GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        Player& active_player = game_state.get_active_player();
        
        // Untap
        ManaSystem::untap_all(active_player);

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
        
        // Draw Phase
        // First turn, first player doesn't draw? (Standard rules: First player skips draw on turn 1)
        // But let's implement standard flow.
        bool skip_draw = (game_state.turn_number == 1 && game_state.active_player_id == 0);
        
        if (!skip_draw) {
            draw_card(game_state, active_player);
        }
        
        // Move to Mana Phase (or Main if we want to be granular, but usually Mana is next decision point)
        // Actually, the "Phase" in GameState isn't defined yet. 
        // We should probably add `current_phase` to GameState if we want to track it there.
        // For now, let's assume the external loop handles the phase state or we add it to GameState.
    }

    void PhaseManager::draw_card(GameState& game_state, Player& player) {
        if (player.deck.empty()) {
            // Deck out check will happen in check_game_over
            return;
        }
        move_card(player.deck, player.hand);
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
        // Check Winner Flag (Direct Attack)
        if (game_state.winner != GameResult::NONE) {
            result = game_state.winner;
            return true;
        }

        // Check Deck Out
        // "Deck Out (Hard): 山札の最後の1枚を引いた瞬間に敗北 [Q27]"
        // This means if deck is empty when trying to draw, or just empty?
        // Usually it's when you *must* draw but cannot.
        // But spec says "Last card drawn -> Loss". Wait.
        // "山札の最後の1枚を引いた瞬間に敗北" -> Lose the moment you draw the last card?
        // That sounds like "If deck becomes empty, you lose".
        
        bool p1_deck_empty = game_state.players[0].deck.empty();
        bool p2_deck_empty = game_state.players[1].deck.empty();

        if (p1_deck_empty && p2_deck_empty) {
            result = GameResult::DRAW;
            return true;
        }
        if (p1_deck_empty) {
            result = GameResult::P2_WIN;
            return true;
        }
        if (p2_deck_empty) {
            result = GameResult::P1_WIN;
            return true;
        }

        // Check Turn Limit
        if (game_state.turn_number > TURN_LIMIT) {
            result = GameResult::DRAW;
            return true;
        }

        // Check Direct Attack (Shields 0 and hit) - This is handled in Battle logic usually.
        // But we can check if a player has lost flag set?
        // For now, let's assume Battle logic sets a flag or we check shields here if needed.
        // But "Direct Attack" is an event.
        
        result = GameResult::NONE;
        return false;
    }

    void PhaseManager::next_phase(GameState& game_state, const std::map<CardID, CardDefinition>& card_db) {
        switch (game_state.current_phase) {
            case Phase::START_OF_TURN:
                game_state.current_phase = Phase::DRAW;
                // Draw logic is called in start_turn usually, but let's separate it if we want fine control
                break;
            case Phase::DRAW:
                game_state.current_phase = Phase::MANA;
                break;
            case Phase::MANA:
                game_state.current_phase = Phase::MAIN;
                break;
            case Phase::MAIN:
                game_state.current_phase = Phase::ATTACK;
                break;
            case Phase::ATTACK:
                game_state.current_phase = Phase::END_OF_TURN;
                // Trigger Reservation: AT_END_OF_TURN
                {
                    Player& active_player = game_state.get_active_player();
                    for (const auto& card : active_player.battle_zone) {
                        if (card_db.count(card.card_id)) {
                            const auto& def = card_db.at(card.card_id);
                            if (def.keywords.at_end_of_turn) {
                                game_state.pending_effects.emplace_back(EffectType::AT_END_OF_TURN, card.instance_id, active_player.id);
                            }
                        }
                    }
                }
                break;
            case Phase::BLOCK:
                // Should normally be handled by resolve_block/execute_battle, but as a fallback:
                game_state.current_phase = Phase::ATTACK;
                break;
            case Phase::END_OF_TURN:
                // Switch turn
                game_state.active_player_id = 1 - game_state.active_player_id;
                if (game_state.active_player_id == 0) {
                    game_state.turn_number++;
                }
                game_state.current_phase = Phase::START_OF_TURN;
                start_turn(game_state, card_db);
                break;
        }
    }

}
