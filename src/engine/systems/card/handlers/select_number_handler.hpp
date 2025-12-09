#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include <iostream>

namespace dm::engine {

    class SelectNumberHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            int chosen_number = 5; // Heuristic default

            // Store output
            if (!ctx.action.output_value_key.empty()) {
                ctx.execution_vars[ctx.action.output_value_key] = chosen_number;
            }

            // std::cout << "DEBUG: SelectNumberHandler chose " << chosen_number << " (Heuristic)" << std::endl;
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
             resolve(ctx);
        }
    };
}
