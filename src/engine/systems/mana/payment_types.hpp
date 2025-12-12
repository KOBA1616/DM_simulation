#pragma once
#include <vector>
#include <map>
#include "core/types.hpp"

namespace dm::engine {

    enum class PaymentType {
        MANA,
        G_ZERO,
        HYPER_ENERGY,
        SYMPATHY
    };

    struct PaymentRequirement {
        // Base requirements
        int base_mana_cost = 0;
        int final_mana_cost = 0;
        std::vector<dm::core::Civilization> required_civs;

        // Active Payment Logic
        bool uses_hyper_energy = false;
        int hyper_energy_count = 0; // Number of creatures required to tap (if fixed) or max allowed?
                                  // For now, let's assume this is the *applied* count in a context
                                  // Or is this the *capability*?
                                  // Requirement usually just says "Pay X".
                                  // Hyper Energy is "You MAY tap creatures. Each reduces cost by 2."
                                  // So the Requirement struct after calculation should reflect the chosen path?
                                  // Let's stick to "What needs to be paid".

        // If true, cost is 0 and no mana/civ check needed (except maybe civ if specified?)
        // G-Zero usually ignores civilization requirement too? (Need to verify, usually yes)
        bool is_g_zero = false;
    };

    struct PaymentContext {
        PaymentType type = PaymentType::MANA;

        // Resources selected for payment
        // We use InstanceID (CardID in types.hpp is uint16_t, often used for both)
        // But let's assume for payment we need specific instances.
        // In GameState, instances are vectors. We usually refer to them by ID (index) if we have the list.
        // Wait, CardID is just a number. Instance ID is implicit index in zone vectors?
        // No, GameState usually has `get_card_instance(id)`.
        std::vector<dm::core::CardID> mana_cards_to_tap;
        std::vector<dm::core::CardID> creatures_to_tap; // For Hyper Energy
    };

}
