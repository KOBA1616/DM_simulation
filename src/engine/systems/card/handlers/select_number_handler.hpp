#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/effect_system.hpp"
#include <iostream>

namespace dm::engine {

    class SelectNumberHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;

            // This is called when the effect is encountered in the chain.
            // We must interrupt execution and push a PendingEffect to ask for input.

            PlayerID controller = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);

            PendingEffect pending(EffectType::SELECT_NUMBER, ctx.source_instance_id, controller);

            // Determine range
            int max_val = ctx.action.value1;

            // Handle Variable Linking for Max Value
             if (!ctx.action.input_value_key.empty() && ctx.execution_vars.count(ctx.action.input_value_key)) {
                max_val = ctx.execution_vars.at(ctx.action.input_value_key);
             }

            // Default to 1 if not specified (though usually it should be specified)
            if (max_val < 1) max_val = 1;

            // Store MAX in num_targets_needed
            pending.num_targets_needed = max_val;

            // Store output key so we know where to put the result
            pending.execution_context = ctx.execution_vars;
            // We add a special entry to context to indicate which key to write to?
            // Or we just look at the ActionDef later?
            // PendingEffect doesn't store ActionDef directly unless we put it in effect_def.
            // But effect_def is usually for CONTINUATION.

            // We need to store the continuation actions.
            EffectDef continuation;
            continuation.trigger = TriggerType::NONE;
            // The current action (SELECT_NUMBER) is done after selection.
            // So we need the REMAINING actions.
            if (ctx.remaining_actions) {
                continuation.actions = *ctx.remaining_actions;
            }

            // We also need to know the output key to write the result to.
            // We can store the current action in the continuation? No, that would re-execute it.
            // We can wrap the continuation in a special way?

            // Simpler: The pending effect stores the *context*.
            // When resolved, we update the context and then execute the continuation.
            // But we need to know the KEY.
            // Let's store the output key in the PendingEffect's filter or a map.
            // FilterDef has no generic string field.
            // We can abuse `filter.owner` or similar, or just add it to `execution_context` with a special temporary key.
            // Or better: The ActionDef that created this pending effect is needed.
            // But `PendingEffect` has `optional<EffectDef> effect_def`.
            // We can create a dummy EffectDef containing just the SELECT_NUMBER action (marked as done) + remaining actions?

            // Actually, we can store the output key in `execution_context` under a reserved name like "__select_number_output_key".
            pending.execution_context["__select_number_output_key_idx"] = 0; // Just a flag
            // Wait, we need the string key. Map is <string, int>. We can't store string value.

            // Let's check `PendingEffect` struct.
            // It has `std::optional<EffectDef> effect_def`.
            // We can put the current action in `effect_def.actions[0]` but mark it or something?

            // Alternative:
            // When `ActionType::SELECT_NUMBER` is resolved in `EffectResolver` (or `GenericCardSystem`),
            // it needs to know where to write the value.
            // If we use the standard `resolve_effect` flow...

            // Let's look at `SelectOptionHandler`.

            // Let's store the output key in `str_val` of the pending effect's `filter`?
            // `FilterDef` doesn't have `str_val`.
            // `ConditionDef` has `str_val`.
            // `EffectDef` has `condition`. `ConditionDef` has `str_val`.
            // We can use `pending.effect_def->condition.str_val` to store the output key!
            // It's a hack, but `ConditionDef` is part of `EffectDef`.

            pending.effect_def = continuation;
            pending.effect_def->condition.str_val = ctx.action.output_value_key; // Store output key here

            ctx.game_state.pending_effects.push_back(pending);

            if (ctx.interrupted) {
                *ctx.interrupted = true;
            }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
             // This is called when we resume?
             // No, `resolve_with_targets` is called by `GenericCardSystem` if it thinks it's a target selection.
             // But `SELECT_NUMBER` is handled specially now.
             resolve(ctx);
        }
    };
}
