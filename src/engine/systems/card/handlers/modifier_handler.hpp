#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/generic_card_system.hpp"
#include "engine/systems/card/passive_effect_system.hpp"
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

            } else if (ctx.action.str_val == "LOCK_SPELL") {
                // Step 3-1: Lock Ability Generalization
                PassiveEffect eff;
                eff.type = PassiveType::CANNOT_USE_SPELLS;
                // eff.value is not strictly needed if we just lock ALL spells matching filter
                // But if we want to lock "Spells with cost X", we can use filter.
                // Or if we want to use the legacy LOCK_SPELL_BY_COST where X is dynamic...

                // If action.filter is set, we use it.
                // If the user wants to lock spells with cost equal to 'value' (e.g. declared number),
                // they should use a filter that checks cost.
                // However, FilterDef 'cost' is static.
                // If we want dynamic cost lock, we might need to enhance FilterDef or use LOCK_SPELL_BY_COST type.

                // For now, assume generic lock using filter.
                eff.target_filter = ctx.action.filter;

                // If we want to support the legacy behavior where 'value' is the cost to lock:
                if (ctx.action.value > "0" || value > 0) { // Check if value is relevant
                     // If value is set, maybe we interpret it as cost?
                     // But LOCK_SPELL usually implies filter-based lock.
                }

                eff.source_instance_id = ctx.source_instance_id;
                eff.controller = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                eff.turns_remaining = (ctx.action.value2 > 0) ? ctx.action.value2 : 1;

                ctx.game_state.passive_effects.push_back(eff);

            } else if (ctx.action.str_val == "POWER") {
                // Power Modifier logic (future)
            }
        }
    };
}
