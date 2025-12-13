#pragma once
#include <vector>
#include <map>
#include "core/types.hpp"
#include "core/game_state.hpp"
#include "core/card_def.hpp"

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

        // Overload with explicit cost (skips get_adjusted_cost calculation)
        static bool auto_tap_mana(dm::core::GameState& game_state, dm::core::Player& player, const dm::core::CardDefinition& card_def, int cost_override, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);

        // Untap all cards in mana zone at start of turn
        static void untap_all(dm::core::Player& player);

        // Calculate cost with modifiers
        static int get_adjusted_cost(const dm::core::GameState& game_state, const dm::core::Player& player, const dm::core::CardDefinition& card_def);

        // Virtual cost calculation for AI/Legality checks [PLAN-002]
        // Currently aliases get_adjusted_cost, but reserved for future "Active Reduction" logic (Hyper Energy, etc.)
        static int get_projected_cost(const dm::core::GameState& game_state, const dm::core::Player& player, const dm::core::CardDefinition& card_def);

        // Helper to get total usable mana (for cost payment checks outside standard flow)
        static int get_usable_mana_count(const dm::core::GameState& game_state, dm::core::PlayerID player_id, const std::vector<dm::core::Civilization>& required_civs, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);

    private:
        // Helper for strict multicolor payment (Rule 817.1a-ish but for payment)
        // Returns indices of mana cards to tap if successful, empty if failed.
        static std::vector<int> solve_payment(const std::vector<dm::core::CardInstance>& mana_zone,
                                              const std::vector<dm::core::Civilization>& required_civs,
                                              int total_cost);
    };

}
