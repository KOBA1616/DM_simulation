#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/generic_card_system.hpp"
#include "engine/systems/card/passive_effect_system.hpp"
#include "engine/game_command/commands.hpp"
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

                 // TODO: Use MUTATE or similar primitive if possible?
                 // CostModifiers are global state mutations, not single-card mutations.
                 // Phase 6 primitives: MUTATE is target-specific.
                 // We might need a GLOBAL_MUTATE or handle this via adding a "GameState Effect".
                 // For now, direct manipulation is acceptable as "generalized implementation",
                 // but ideally should be wrapped.
                 // However, the requirement is "generalization of APPLY_MODIFIER to eliminate individual implementations".
                 // This handler IS the generalized implementation.
                 // The key is that `TapHandler` and `UntapHandler` use `MutateCommand`.
                 // `ModifierHandler` uses global state.
                 // Phase 6.3: Migrated to use GameCommand (MutateCommand::ADD_MODIFIER)

                 auto cmd = std::make_shared<game_command::MutateCommand>(-1, game_command::MutateCommand::MutationType::ADD_MODIFIER);
                 cmd->modifier_payload = mod;
                 cmd->execute(const_cast<GameState&>(ctx.game_state));
                 const_cast<GameState&>(ctx.game_state).command_history.push_back(cmd);

            } else if (ctx.action.str_val == "LOCK_SPELL") {
                PassiveEffect eff;
                eff.type = PassiveType::CANNOT_USE_SPELLS;
                eff.target_filter = ctx.action.filter;
                eff.source_instance_id = ctx.source_instance_id;
                eff.controller = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                eff.turns_remaining = (ctx.action.value2 > 0) ? ctx.action.value2 : 1;

                auto cmd = std::make_shared<game_command::MutateCommand>(-1, game_command::MutateCommand::MutationType::ADD_PASSIVE);
                cmd->passive_payload = eff;
                cmd->execute(const_cast<GameState&>(ctx.game_state));
                const_cast<GameState&>(ctx.game_state).command_history.push_back(cmd);

            } else if (ctx.action.str_val == "POWER") {
                // If targeting specific cards (e.g. infinite duration power mod or turn based)
                // If it is a turn-based power mod (Buff), it's a PassiveEffect.
                // If it's a permanent modification (counters), it's a MUTATE command.

                // Usually "APPLY_MODIFIER" "POWER" implies "Give +X until end of turn".
                // This is a PassiveEffect.

                PassiveEffect eff;
                eff.type = PassiveType::POWER_MODIFIER;
                eff.value = value;
                eff.target_filter = ctx.action.filter; // Apply to cards matching filter
                eff.source_instance_id = ctx.source_instance_id;
                eff.controller = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                eff.turns_remaining = (ctx.action.value2 > 0) ? ctx.action.value2 : 1;

                // If targets are explicitly selected (TargetScope::TARGET_SELECT), we need to restrict the filter or ID list.
                if (targets) {
                     // The PassiveEffect structure needs to support list of IDs or we create one effect per target?
                     // Or we create a specific filter "ID in [list]".
                     // Existing PassiveEffect usually uses filter.
                     // Creating one effect per target is safer for specific targets.
                     for(int tid : *targets) {
                         // We can't easily express "List of IDs" in current PassiveEffect Filter.
                         // Skipping specific target logic for passives for now, assuming "All creatures" filter is used.
                         // If we needed specific targets, we would iterate and use PERMANENT POWER_MOD mutation,
                         // but this is temporary power mod.
                         // Future task: Add "target_ids" to PassiveEffect.
                     }
                } else {
                     auto cmd = std::make_shared<game_command::MutateCommand>(-1, game_command::MutateCommand::MutationType::ADD_PASSIVE);
                     cmd->passive_payload = eff;
                     cmd->execute(const_cast<GameState&>(ctx.game_state));
                     const_cast<GameState&>(ctx.game_state).command_history.push_back(cmd);
                }
            }
        }
    };
}
