#pragma once
#include "../../core/game_state.hpp"
#include "../../core/card_def.hpp"
#include <vector>
#include <map>

namespace dm::engine {

    class ManaSystem {
    public:
        // Checks if the player has enough mana and correct civilizations to play a card
        static bool can_pay_cost(const dm::core::Player& player, const dm::core::CardDefinition& card_def, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);

        // Automatically taps mana for a card. Returns true if successful.
        // This is a greedy implementation for now, or a simple one.
        static bool auto_tap_mana(dm::core::Player& player, const dm::core::CardDefinition& card_def, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);

        // Untap all cards in mana zone at start of turn
        static void untap_all(dm::core::Player& player);
    };

}
