#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/game_command/commands.hpp"
#include "core/modifiers.hpp"
#include <memory>

namespace dm::engine {

    class ModifierHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
             if (ctx.action.scope == dm::core::TargetScope::TARGET_SELECT || ctx.action.target_choice == "SELECT") {
                 dm::core::EffectDef ed;
                 ed.actions = { ctx.action };
                 GenericCardSystem::select_targets(ctx.game_state, ctx.action, ctx.source_instance_id, ed, ctx.execution_vars);
                 return;
            }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            apply_modifier(ctx, ctx.targets);
        }

    private:
        void apply_modifier(const ResolutionContext& ctx, const std::vector<int>* targets) {
            using namespace dm::core;
            using namespace dm::engine::game_command;

            if (!targets) return;

            if (ctx.action.type == EffectActionType::APPLY_MODIFIER) {
                 int val = ctx.action.value1;
                 if (val == 0 && !ctx.action.value.empty()) {
                     try { val = std::stoi(ctx.action.value); } catch(...) {}
                 }

                 for (int tid : *targets) {
                     auto cmd = std::make_shared<MutateCommand>(tid, MutateCommand::MutationType::ADD_MODIFIER, val, ctx.action.value2);
                     ctx.game_state.execute_command(cmd);
                 }
            }
            else if (ctx.action.type == EffectActionType::GRANT_KEYWORD) {
                 int val = ctx.action.value1;
                 if (val == 0 && !ctx.action.value.empty()) {
                     try { val = std::stoi(ctx.action.value); } catch(...) {}
                 }
                 for (int tid : *targets) {
                     auto cmd = std::make_shared<MutateCommand>(tid, MutateCommand::MutationType::ADD_PASSIVE, val, ctx.action.value2);
                     ctx.game_state.execute_command(cmd);
                 }
            }
        }
    };
}
