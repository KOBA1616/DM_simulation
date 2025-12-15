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
            using namespace dm::engine::game_command;

            // Determine value
            int value = ctx.action.value1;
            if (!ctx.action.input_value_key.empty() && ctx.execution_vars.count(ctx.action.input_value_key)) {
                value = ctx.execution_vars.at(ctx.action.input_value_key);
            }

            int duration = (ctx.action.value2 > 0) ? ctx.action.value2 : 1;
            PlayerID controller = GenericCardSystem::get_controller(ctx.game_state, ctx.source_instance_id);

            // Using GameCommand architecture
            // CommandType: MUTATE (extended for Global Modifiers)

            if (ctx.action.str_val == "COST") {
                 auto cmd = std::make_shared<MutateCommand>(-1, MutateCommand::MutationType::ADD_GLOBAL_MODIFIER, value);
                 cmd->set_modifier_data(ctx.action.filter, duration, controller, ctx.source_instance_id);

                 // Execute directly via GameState? Or push to history?
                 // GenericCardSystem usually executes actions.
                 // We should push to history AND execute.
                 // But currently GameState doesn't automatically execute when pushed.
                 // We must manually execute.
                 // NOTE: If we want undo support, we should use a command manager.
                 // For now, we manually do it as per other handlers (if any use GameCommand).
                 // `MoveCardHandler` in Phase 6.3 should be using GameCommand.

                 cmd->execute(ctx.game_state);
                 ctx.game_state.command_history.push_back(cmd);

            } else {
                 // LOCK_SPELL, POWER, etc. -> PassiveEffect
                 // Map str_val to Passive Type string expected by MutateCommand
                 std::string passive_key = ctx.action.str_val;
                 // If POWER, we pass 'value' as int_value.
                 // If KEYWORD, 'value' might be ignored or used differently.

                 auto cmd = std::make_shared<MutateCommand>(-1, MutateCommand::MutationType::ADD_PASSIVE_EFFECT, value, passive_key);
                 cmd->set_modifier_data(ctx.action.filter, duration, controller, ctx.source_instance_id);

                 cmd->execute(ctx.game_state);
                 ctx.game_state.command_history.push_back(cmd);
            }
        }
    };
}
