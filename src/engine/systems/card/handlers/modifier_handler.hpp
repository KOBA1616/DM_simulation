#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/generic_card_system.hpp"
#include <iostream>

namespace dm::engine {

    class ModifierHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            apply_modifier(ctx, nullptr);
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            apply_modifier(ctx, ctx.targets);
        }

    private:
        void apply_modifier(const ResolutionContext& ctx, const std::vector<int>* targets) {
            using namespace dm::core;

            // Determine value
            int value = ctx.action.value1;
            if (!ctx.action.input_value_key.empty() && ctx.execution_vars.count(ctx.action.input_value_key)) {
                value = ctx.execution_vars.at(ctx.action.input_value_key);
            }

            // Determine type
            // ActionDef.str_val maps to operation type
            // e.g. "COST" -> COST_REDUCTION

            if (ctx.action.str_val == "COST") {
                 CostModifier mod;
                 mod.reduction_amount = value;
                 mod.condition_filter = ctx.action.filter; // Filter determines which cards get reduced
                 mod.source_instance_id = ctx.source_instance_id;
                 mod.controller = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);

                 // Duration from value2
                 // 0 or 1 = this turn
                 mod.turns_remaining = (ctx.action.value2 > 0) ? ctx.action.value2 : 1;

                 ctx.game_state.active_modifiers.push_back(mod);
            }
            // Add other modifier types here if needed (e.g. POWER)
        }
    };
}
