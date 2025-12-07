#pragma once
#include "../../core/game_state.hpp"
#include "../../core/card_def.hpp"

namespace dm::engine {
    class PassiveEffectSystem {
    public:
        static PassiveEffectSystem& instance() {
            static PassiveEffectSystem instance;
            return instance;
        }

        int get_power_buff(const dm::core::GameState& game_state, const dm::core::CardInstance& creature, const std::map<dm::core::CardID, dm::core::CardDefinition>& card_db) {
            // Stub implementation
            return 0;
        }
    };
}
