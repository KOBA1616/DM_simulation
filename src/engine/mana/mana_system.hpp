#pragma once
#include <vector>
#include <map>
#include "../../core/types.hpp"

// Forward declarations
namespace dm::core {
    struct GameState;
    struct Player;
    struct CardDefinition;
}

namespace dm::engine {

    class ManaSystem {
    public:
        // Checks if the player has enough mana and correct civilizations to play a card
        // Now requires GameState to check active cost modifiers
        static bool can_pay_cost(const dm::core::GameState& game_state, const dm::core::Player& player, const dm::core::CardDefinition& card_def, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);

        // Legacy overload for backward compatibility (assumes no game state context, i.e. no modifiers)
        static bool can_pay_cost(const dm::core::Player& player, const dm::core::CardDefinition& card_def, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);


        // Automatically taps mana for a card. Returns true if successful.
        // This is a greedy implementation for now, or a simple one.
        static bool auto_tap_mana(dm::core::GameState& game_state, dm::core::Player& player, const dm::core::CardDefinition& card_def, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);

        // Legacy overload
        static bool auto_tap_mana(dm::core::Player& player, const dm::core::CardDefinition& card_def, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);

        // Untap all cards in mana zone at start of turn
        static void untap_all(dm::core::Player& player);

        // Calculate cost with modifiers
        static int get_adjusted_cost(const dm::core::GameState& game_state, const dm::core::Player& player, const dm::core::CardDefinition& card_def);

        // Virtual cost calculation for AI/Legality checks [PLAN-002]
        // Currently aliases get_adjusted_cost, but reserved for future "Active Reduction" logic (Hyper Energy, etc.)
        static int get_projected_cost(const dm::core::GameState& game_state, const dm::core::Player& player, const dm::core::CardDefinition& card_def);
    };

}
