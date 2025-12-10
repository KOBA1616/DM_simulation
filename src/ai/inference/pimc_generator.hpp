#pragma once

#include "core/game_state.hpp"
#include <vector>
#include <map>
#include <random>

namespace dm::ai::inference {

    class PIMCGenerator {
    public:
        /**
         * @brief Generates a determinized GameState by sampling unknown cards from a candidate pool.
         *
         * @param observation The game state as seen by the observer (with hidden info).
         * @param card_db The card database.
         * @param observer_id The player ID of the observer (whose info is known).
         * @param opponent_deck_candidates A list of CardIDs representing the pool of cards
         *                                 believed to be in the opponent's hidden zones (Hand, Shield, Deck).
         * @param seed Random seed for shuffling.
         * @return dm::core::GameState A complete game state with materialized hidden information.
         */
        static dm::core::GameState generate_determinized_state(
            const dm::core::GameState& observation,
            const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db,
            dm::core::PlayerID observer_id,
            const std::vector<dm::core::CardID>& opponent_deck_candidates,
            uint32_t seed
        );
    };

}
