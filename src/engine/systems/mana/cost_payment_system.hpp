#pragma once
#include "core/game_state.hpp"
#include "core/card_def.hpp"

namespace dm::engine {

    class CostPaymentSystem {
    public:
        // Calculates how many units of payment (e.g. number of creatures to tap) can be paid.
        static int calculate_max_units(const dm::core::GameState& state,
                                       dm::core::PlayerID player_id,
                                       const dm::core::CostReductionDef& reduction,
                                       const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);

        // Calculates the maximum cost reduction value achievable.
        static int calculate_potential_reduction(const dm::core::GameState& state,
                                                 dm::core::PlayerID player_id,
                                                 const dm::core::CostReductionDef& reduction,
                                                 const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);

        // Checks if a card can be played considering available mana and reductions.
        static bool can_pay_cost(const dm::core::GameState& state,
                                 dm::core::PlayerID player_id,
                                 const dm::core::CardDefinition& card,
                                 const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);

        // Executes the payment (e.g. taps creatures) for the specified number of units.
        // Returns the total reduction amount applied.
        static int execute_payment(dm::core::GameState& state,
                                   dm::core::PlayerID player_id,
                                   const dm::core::CostReductionDef& reduction,
                                   int units,
                                   const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db);
    };

}
