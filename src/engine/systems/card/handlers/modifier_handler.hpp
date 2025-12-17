#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/effect_system.hpp"
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
            using namespace dm::engine::game_command;

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
                 mod.controller = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);

                 // Duration from value2
                 // 0 or 1 = this turn
                 mod.turns_remaining = (ctx.action.value2 > 0) ? ctx.action.value2 : 1;

                 // Use MutateCommand for GameState mutation
                 auto cmd = std::make_shared<MutateCommand>(-1, MutateCommand::MutationType::ADD_COST_MODIFIER);
                 cmd->cost_modifier = mod;
                 cmd->execute(ctx.game_state);
                 ctx.game_state.command_history.push_back(cmd);

            } else if (ctx.action.str_val == "LOCK_SPELL") {
                PassiveEffect eff;
                eff.type = PassiveType::CANNOT_USE_SPELLS;
                eff.target_filter = ctx.action.filter;
                eff.source_instance_id = ctx.source_instance_id;
                eff.controller = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                eff.turns_remaining = (ctx.action.value2 > 0) ? ctx.action.value2 : 1;

                auto cmd = std::make_shared<MutateCommand>(-1, MutateCommand::MutationType::ADD_PASSIVE_EFFECT);
                cmd->passive_effect = eff;
                cmd->execute(ctx.game_state);
                ctx.game_state.command_history.push_back(cmd);

            } else if (ctx.action.str_val == "POWER") {
                PassiveEffect eff;
                eff.type = PassiveType::POWER_MODIFIER;
                eff.value = value;
                eff.target_filter = ctx.action.filter; // Apply to cards matching filter
                eff.source_instance_id = ctx.source_instance_id;
                eff.controller = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                eff.turns_remaining = (ctx.action.value2 > 0) ? ctx.action.value2 : 1;

                // TODO: Handle explicit targets (ID list) better in PassiveEffect structure.
                // For now, if targets are provided, we should ideally craft a filter for those IDs,
                // but standard PassiveEffect relies on filters.
                // If the action scope was TARGET_SELECT, we might need to issue separate commands or improve PassiveEffect.

                auto cmd = std::make_shared<MutateCommand>(-1, MutateCommand::MutationType::ADD_PASSIVE_EFFECT);
                cmd->passive_effect = eff;
                cmd->execute(ctx.game_state);
                ctx.game_state.command_history.push_back(cmd);
            }
        }
    };
}
