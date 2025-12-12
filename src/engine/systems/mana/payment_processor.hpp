#pragma once
#include "core/game_state.hpp"
#include "payment_types.hpp"
#include "cost_calculator.hpp"

namespace dm::engine {

    class PaymentProcessor {
    public:
        // Validates and executes the payment
        // Returns true if successful, false otherwise (and rolls back/does nothing)
        static bool process_payment(
            dm::core::GameState& game_state,
            dm::core::Player& player,
            const PaymentRequirement& req,
            const PaymentContext& context
        );

    private:
        static bool pay_mana(
            dm::core::GameState& game_state,
            dm::core::Player& player,
            int cost,
            const std::vector<dm::core::Civilization>& required_civs,
            const std::vector<dm::core::CardID>& mana_ids
        );

        static bool pay_hyper_energy(
            dm::core::GameState& game_state,
            dm::core::Player& player,
            int creature_count,
            const std::vector<dm::core::CardID>& creature_ids
        );
    };

}
