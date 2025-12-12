#pragma once
#include "core/game_state.hpp"
#include "core/card_def.hpp"
#include "payment_types.hpp"

namespace dm::engine {

    class CostCalculator {
    public:
        // Calculate the PaymentRequirement for a specific card play
        // This considers static modifiers (already in GameState) and active modifiers (passed in context or flags)
        static PaymentRequirement calculate_requirement(
            const dm::core::GameState& game_state,
            const dm::core::Player& player,
            const dm::core::CardDefinition& card_def,
            bool use_hyper_energy = false, // User intent
            int hyper_energy_creature_count = 0 // User intent
        );

        // Helper to check standard mana cost
        static int get_base_adjusted_cost(
            const dm::core::GameState& game_state,
            const dm::core::Player& player,
            const dm::core::CardDefinition& card_def
        );
    };

}
