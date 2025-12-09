#pragma once
#include "core/game_state.hpp"
#include "core/types.hpp"

namespace dm::engine {

    class DevTools {
    public:
        // Moves 'count' cards from source zone to target zone for the specified player.
        // If card_id_filter is not -1, only moves cards with that ID.
        // Returns the number of cards actually moved.
        static int move_cards(dm::core::GameState& state, int player_id, dm::core::Zone source, dm::core::Zone target, int count, int card_id_filter = -1);

        // Forces loop detection by pushing the current hash to history twice.
        static void trigger_loop_detection(dm::core::GameState& state);
    };

}
