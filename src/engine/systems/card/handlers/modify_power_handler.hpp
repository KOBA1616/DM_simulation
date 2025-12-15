#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/game_command/commands.hpp"
#include <iostream>

namespace dm::engine {

    class ModifyPowerHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            modify_power(ctx, nullptr);
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            modify_power(ctx, ctx.targets);
        }

    private:
        void modify_power(const ResolutionContext& ctx, const std::vector<int>* targets) {
            using namespace dm::core;
            using namespace dm::engine::game_command;

            int value = ctx.action.value1;
            if (!ctx.action.input_value_key.empty() && ctx.execution_vars.count(ctx.action.input_value_key)) {
                value = ctx.execution_vars.at(ctx.action.input_value_key);
            }

            // MODIFY_POWER action usually implies permanent modification if not specified otherwise via APPLY_MODIFIER.
            // However, most "Power +X" effects are "until end of turn".
            // If the JSON definition uses "MODIFY_POWER", we assume it maps to "Power Mod" (MutateCommand::POWER_MOD)
            // which is currently implemented as a persistent change on the CardInstance struct.
            // If the user intends "until end of turn", they should use "APPLY_MODIFIER" with "POWER".

            // To support generic "Power +X" that might be permanent (e.g. counters), we use MutateCommand.

            // We need targets.
            std::vector<int> final_targets;
            if (targets) {
                final_targets = *targets;
            } else {
                 // Implicit targets? Usually requires SELECT or targets passed in context.
                 // If filter is present and scope is ALL_FILTERED or similar?
                 // But GenericCardSystem usually handles selection before calling handler if scope is SELECT.
                 // If scope is ALL_FILTERED, we iterate.
            }

            for (int target_id : final_targets) {
                auto cmd = std::make_shared<MutateCommand>(target_id, MutateCommand::MutationType::POWER_MOD, value);
                cmd->execute(ctx.game_state);
                ctx.game_state.command_history.push_back(cmd);
            }
        }
    };
}
