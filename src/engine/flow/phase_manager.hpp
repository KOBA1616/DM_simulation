#pragma once
#include "../../core/game_state.hpp"
#include "../../core/types.hpp"
#include "../../core/card_def.hpp"
#include <map>

namespace dm::engine {

    class PhaseManager {
    public:
        static void start_game(dm::core::GameState& game_state, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
        static void next_phase(dm::core::GameState& game_state, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
        
        // Fast forward the game until a decision is needed or game over
        static void fast_forward(dm::core::GameState& game_state, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);

        // Returns true if the game has ended
        static bool check_game_over(dm::core::GameState& game_state, dm::core::GameResult& result);

    private:
        static void start_turn(dm::core::GameState& game_state, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
        static void draw_card(dm::core::GameState& game_state, dm::core::Player& player);
    };

}
