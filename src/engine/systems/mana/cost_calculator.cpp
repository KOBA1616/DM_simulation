#include "cost_calculator.hpp"
#include "mana_system.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/card/condition_system.hpp" // Added for G-Zero
#include <algorithm>
#include <iostream>

namespace dm::engine {

    int CostCalculator::get_base_adjusted_cost(
        const dm::core::GameState& game_state,
        const dm::core::Player& player,
        const dm::core::CardDefinition& card_def
    ) {
        int cost = card_def.cost;

        // Use active modifiers from GameState
        for (const auto& mod : game_state.active_modifiers) {
            // Check player
            if (mod.controller != player.id && mod.controller != 255) continue;

            // Check condition using TargetUtils
            dm::core::CardInstance dummy_inst;
            dummy_inst.card_id = card_def.id;
            dummy_inst.instance_id = -1;
            dummy_inst.owner = player.id;

            bool match = TargetUtils::is_valid_target(
                     dummy_inst,
                     card_def,
                     mod.condition_filter,
                     game_state,
                     mod.controller,
                     player.id,
                     true
                 );

            if (match) {
                cost -= mod.reduction_amount;
            }
        }

        if (cost < 1) cost = 1;
        return cost;
    }

    PaymentRequirement CostCalculator::calculate_requirement(
        const dm::core::GameState& game_state,
        const dm::core::Player& player,
        const dm::core::CardDefinition& card_def,
        bool use_hyper_energy,
        int hyper_energy_creature_count
    ) {
        PaymentRequirement req;
        req.base_mana_cost = card_def.cost;
        req.required_civs = card_def.civilizations;

        // G-Zero Logic
        if (card_def.keywords.g_zero) {
             // To evaluate G-Zero, we usually look for a specific effect on the card that defines the condition.
             // In CardDefinition, where is the G-Zero condition stored?
             // It might be in `effects` with a specific trigger/type or a separate field.
             // Currently CardDefinition has `keywords.g_zero` flag.
             // But the *condition* (e.g. "If you have 6 or more fire cards in mana") is usually
             // parsed into a ConditionDef.
             // Let's assume there is a convention or we need to find it.
             // For now, I'll search `effects` for an effect with trigger type `NONE`? No.
             // Actually, `ConditionSystem` evaluates `ConditionDef`.
             // We need to find the `ConditionDef` associated with G-Zero.
             // If `CardDefinition` lacks a specific `g_zero_condition` field, we might need to rely on
             // checking `effects`.
             // Or maybe we assume `ConditionSystem` isn't fully integrated with G-Zero definition yet?
             // The memory says: "CardImplementation System aims to be data-driven...".
             // Let's look at `CardDefinition` in `src/core/card_def.hpp` again.
             // It has `std::vector<EffectDef> effects`.
             // It doesn't have explicit G-Zero condition field.

             // However, `CostHandler` or similar might handle it? No.
             // Let's assume for now that if `keywords.g_zero` is true, we should check if ANY
             // effect is a "Cost Replacement" effect?

             // Actually, usually G-Zero is implemented as a keyword that implies a specific condition based on text,
             // OR in the new system, it should be explicit.
             // Given I cannot change `CardDefinition` schema easily without affecting JSON loader,
             // I will try to find a `ConditionDef` in `effects` that looks like G-Zero.
             // But G-Zero isn't an "Effect" that resolves. It's a static ability.

             // Workaround: We will skip strict G-Zero condition evaluation here unless I find where it's stored.
             // Wait, `TargetUtils` is for targeting. `ConditionSystem` is for `ConditionDef`.
             // If I can't access the condition, I can't evaluate it.

             // But wait, the user asked for "G-Zero Implementation".
             // If I cannot implement it fully due to data structure limitations, I should mark it.
             // But I can implement the *check* if I had the condition.

             // I'll add a placeholder that if `g_zero` flag is set, we check a hypothetical `g_zero_condition`.
             // For now, I will assume it is false unless I can find it.
             // Or maybe I can assume for TEST purposes (and future data) that `modes` or something holds it?

             // Realistically, `JsonLoader` should populate a `g_zero_condition`.
             // Since it doesn't exist, I will leave it as:
             // req.is_g_zero = false; // Until data structure supports it.

             // BUT, to satisfy the requirement "Implement G-Zero", I will add `is_g_zero` to `PaymentRequirement`
             // and let the caller manually override it if they know? No.

             // Let's look at `CardDefinition` again.
             // `std::vector<HandTrigger> hand_triggers`.
             // Maybe G-Zero is a HandTrigger?
             // `HandTrigger` has `ConditionDef`.
             // TriggerType could be `G_ZERO`?
             // `TriggerType` enum doesn't have `G_ZERO`.

             // Okay, I will defer strict G-Zero condition loading.
             // But I will implement the *structure* so that if we pass a boolean `is_g_zero_satisfied` (maybe from context?),
             // it works. But `CostCalculator` signature doesn't take context.

             // I'll just check if cost becomes 0 via G-Zero flag for now (assume satisfied if flag present? NO that breaks game).
             // I will leave it as TODO with a clear comment.
        }

        // 1. Calculate Standard Adjusted Cost
        int cost = get_base_adjusted_cost(game_state, player, card_def);

        // 2. Active Modifiers (Hyper Energy)
        if (use_hyper_energy && card_def.keywords.hyper_energy) {
            req.uses_hyper_energy = true;
            req.hyper_energy_count = hyper_energy_creature_count;

            int reduction = hyper_energy_creature_count * 2;
            cost -= reduction;

            if (cost < 0) cost = 0;
        }

        req.final_mana_cost = cost;
        return req;
    }

}
