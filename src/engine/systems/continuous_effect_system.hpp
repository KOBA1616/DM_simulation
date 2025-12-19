#pragma once

#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include <map>

namespace dm::engine::systems {

    class ContinuousEffectSystem {
    public:
        // Recalculates all continuous effects (static abilities) and updates GameState active_modifiers/passive_effects.
        // Should be called whenever the board state changes significantly (e.g. after card play, zone change, turn start).
        static void recalculate(core::GameState& state, const std::map<core::CardID, core::CardDefinition>& card_db);
    };

}
