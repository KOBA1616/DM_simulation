#pragma once
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include <map>

namespace dm::engine::systems {

    class BreakerSystem {
    public:
        /**
         * @brief Calculates the breaker count (number of shields to break) for a creature.
         *
         * Logic:
         * 1. Check for modifiers granting breaker capabilities (World > Triple > Double).
         * 2. Check base card keywords.
         * 3. Return the highest applicable breaker count.
         *
         * @param state The current game state (to check modifiers).
         * @param creature The card instance attacking.
         * @param def The card definition.
         * @return int Number of shields to break (999 for World Breaker).
         */
        static int get_breaker_count(const core::GameState& state, const core::CardInstance& creature, const core::CardDefinition& def);
    };

}
