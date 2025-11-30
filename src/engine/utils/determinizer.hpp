#pragma once
#include "../../core/game_state.hpp"
#include "../../core/card_def.hpp"
#include <map>

namespace dm::engine {

    class Determinizer {
    public:
        // Randomize hidden zones of the opponent (relative to observer_player_id)
        // Assumes we know the set of cards in hidden zones (i.e. we know the deck list).
        static void determinize(dm::core::GameState& state, int observer_player_id);
    };

}
