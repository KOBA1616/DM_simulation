#pragma once

#include "../../core/game_state.hpp"
#include "../../core/card_def.hpp"
#include "../../engine/systems/card/target_utils.hpp"
#include <map>
#include <vector>

namespace dm::ai {

    class LethalSolver {
    public:
        /**
         * Checks if the active player has a guaranteed lethal on board.
         *
         * @param game_state Current game state.
         * @param card_db Card database for keyword lookups (Blocker, SA, etc).
         * @return true if lethal is found, false otherwise.
         */
        static bool is_lethal(const dm::core::GameState& game_state,
                              const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);

    private:
        struct AttackerInfo {
            int instance_id;
            int power;
            bool can_be_blocked;
            // Add more properties later (breaks_double, etc.)
        };

        struct BlockerInfo {
            int instance_id;
            int power;
        };
    };

}
