#pragma once
#include "../../core/game_state.hpp"
#include "../../core/types.hpp"

namespace dm::engine {

    class PhaseManager {
    public:
        static void start_game(dm::core::GameState& game_state);
        static void next_phase(dm::core::GameState& game_state);
        
        // Returns true if the game has ended
        static bool check_game_over(dm::core::GameState& game_state, dm::core::GameResult& result);

    private:
        static void start_turn(dm::core::GameState& game_state);
        static void draw_card(dm::core::GameState& game_state, dm::core::Player& player);
    };

}
