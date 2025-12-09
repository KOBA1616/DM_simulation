#pragma once
#include "../effect_system.hpp"
#include "../../../../core/game_state.hpp"
#include "../generic_card_system.hpp"
#include <algorithm>

namespace dm::engine {

    class MoveToUnderCardHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& /*ctx*/) override {
            // Stub implementation
        }

        void resolve_with_targets(const ResolutionContext& /*ctx*/) override {
             // Stub
        }
    };
}
