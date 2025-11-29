#include "phase_manager.hpp"
#include "../mana/mana_system.hpp"
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

    void PhaseManager::start_game(GameState& game_state) {
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

        start_turn(game_state);
    }

    void PhaseManager::start_turn(GameState& game_state) {
        Player& active_player = game_state.get_active_player();
        
        // Untap
        ManaSystem::untap_all(active_player);
        
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

    bool PhaseManager::check_game_over(GameState& game_state, GameResult& result) {
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

    void PhaseManager::next_phase(GameState& game_state) {
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
                break;
            case Phase::END_OF_TURN:
                // Switch turn
                game_state.active_player_id = 1 - game_state.active_player_id;
                if (game_state.active_player_id == 0) {
                    game_state.turn_number++;
                }
                game_state.current_phase = Phase::START_OF_TURN;
                start_turn(game_state);
                break;
        }
    }

}
