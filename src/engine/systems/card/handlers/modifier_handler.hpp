#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/systems/card/passive_effect_system.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/pipeline_executor.hpp"
#include <iostream>
#include <memory>

namespace dm::engine {

    class ModifierHandler : public IActionHandler {
    public:
        void compile_action(const ResolutionContext& ctx) override {
            using namespace dm::core;

            // Map ActionType to InstructionOp::MODIFY parameters

            int value = ctx.action.value1;
            if (!ctx.action.input_value_key.empty() && ctx.execution_vars.count(ctx.action.input_value_key)) {
                value = ctx.execution_vars.at(ctx.action.input_value_key);
            }

            nlohmann::json args;
            args["value"] = value;
            args["str_value"] = ctx.action.str_val;
            args["duration"] = (ctx.action.value2 > 0) ? ctx.action.value2 : 1;

            // Resolve dynamic filter refs (e.g., cost_ref -> exact_cost) from execution context
            FilterDef resolved_filter = ctx.action.filter; // Filter is used for Global modifiers (ADD_PASSIVE) or Cost Modifiers
            if (resolved_filter.cost_ref.has_value()) {
                const auto& key = resolved_filter.cost_ref.value();
                if (ctx.execution_vars.count(key)) {
                    resolved_filter.exact_cost = ctx.execution_vars.at(key);
                }
            }
            args["filter"] = resolved_filter;

            // Serialize condition if present
            // ActionDef has 'condition' field (std::optional<ConditionDef>)
            if (ctx.action.condition.has_value() && 
                ctx.action.condition->type != "NONE" && 
                !ctx.action.condition->type.empty()) {
                // We need to serialize ConditionDef to JSON.
                // Since nlohmann/json is available and to_json is defined for ConditionDef in card_json_types.hpp
                args["condition"] = ctx.action.condition.value();
            }

            // Determine the type for PipelineExecutor::handle_modify
            if (ctx.action.str_val == "COST") {
                args["type"] = "ADD_COST_MODIFIER";
            } else {
                // LOCK_SPELL, POWER, CANNOT_ATTACK, etc. are handled as ADD_PASSIVE
                args["type"] = "ADD_PASSIVE";
            }

            // Note: If 'targets' were selected in previous steps and we want to apply SPECIFIC modifiers (not global),
            // we should pass the targets.
            // If ctx.targets is set, it means we have specific targets.
            if (ctx.targets && !ctx.targets->empty()) {
                // Serialize the vector of targets directly to ensure accuracy
                args["target"] = *ctx.targets;
            }

            ctx.instruction_buffer->emplace_back(InstructionOp::MODIFY, args);
        }

        void resolve(const ResolutionContext& ctx) override {
            // Fallback to legacy implementation for immediate execution
            apply_modifier(ctx, ctx.targets);
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
             apply_modifier(ctx, ctx.targets);
        }

    private:
        void apply_modifier(const ResolutionContext& ctx, const std::vector<int>* targets) {
            using namespace dm::core;
            using namespace dm::engine::game_command;

            int value = ctx.action.value1;
            if (!ctx.action.input_value_key.empty() && ctx.execution_vars.count(ctx.action.input_value_key)) {
                value = ctx.execution_vars.at(ctx.action.input_value_key);
            }

            auto create_passive = [&](PassiveType type, int val = 0) {
                // If targets are provided, apply to specific targets.
                // Otherwise, it's a global passive (using filter).

                if (targets && !targets->empty()) {
                    for (int id : *targets) {
                        PassiveEffect eff;
                        eff.type = type;
                        eff.value = val;
                        eff.specific_targets = std::vector<int>{id};
                        eff.source_instance_id = ctx.source_instance_id;
                        eff.controller = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                        eff.turns_remaining = (ctx.action.value2 > 0) ? ctx.action.value2 : 1;

                        auto cmd = std::make_unique<MutateCommand>(-1, MutateCommand::MutationType::ADD_PASSIVE_EFFECT);
                        cmd->passive_effect = eff;
                        cmd->execute(ctx.game_state);
                        ctx.game_state.command_history.push_back(std::move(cmd));
                    }
                } else {
                    PassiveEffect eff;
                    eff.type = type;
                    eff.value = val;
                    eff.target_filter = ctx.action.filter;
                    eff.source_instance_id = ctx.source_instance_id;
                    eff.controller = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                    eff.turns_remaining = (ctx.action.value2 > 0) ? ctx.action.value2 : 1;

                    auto cmd = std::make_unique<MutateCommand>(-1, MutateCommand::MutationType::ADD_PASSIVE_EFFECT);
                    cmd->passive_effect = eff;
                    cmd->execute(ctx.game_state);
                    ctx.game_state.command_history.push_back(std::move(cmd));
                }
            };

            if (ctx.action.str_val == "COST") {
                 CostModifier mod;
                 mod.reduction_amount = value;
                 // Resolve dynamic filter refs for cost modifiers
                 FilterDef resolved_filter = ctx.action.filter;
                 if (resolved_filter.cost_ref.has_value()) {
                     const auto& key = resolved_filter.cost_ref.value();
                     if (ctx.execution_vars.count(key)) {
                         resolved_filter.exact_cost = ctx.execution_vars.at(key);
                     }
                 }
                 mod.condition_filter = resolved_filter;
                 mod.source_instance_id = ctx.source_instance_id;
                 mod.controller = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                 mod.turns_remaining = (ctx.action.value2 > 0) ? ctx.action.value2 : 1;

                 auto cmd = std::make_unique<MutateCommand>(-1, MutateCommand::MutationType::ADD_COST_MODIFIER);
                 cmd->cost_modifier = mod;
                 cmd->execute(ctx.game_state);
                 ctx.game_state.command_history.push_back(std::move(cmd));

            } else if (ctx.action.str_val == "LOCK_SPELL") {
                create_passive(PassiveType::CANNOT_USE_SPELLS);
            } else if (ctx.action.str_val == "POWER") {
                // Resolve filter references for all paths (specific targets or global)
                FilterDef resolved_filter = ctx.action.filter;
                if (resolved_filter.cost_ref.has_value()) {
                    const auto& key = resolved_filter.cost_ref.value();
                    if (ctx.execution_vars.count(key)) {
                        resolved_filter.exact_cost = ctx.execution_vars.at(key);
                    }
                }
                
                if (targets && !targets->empty()) {
                    // Apply to specific targets
                    for (int id : *targets) {
                        PassiveEffect eff;
                        eff.type = PassiveType::POWER_MODIFIER;
                        eff.value = value;
                        eff.specific_targets = std::vector<int>{id};
                        eff.source_instance_id = ctx.source_instance_id;
                        eff.controller = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                        eff.turns_remaining = (ctx.action.value2 > 0) ? ctx.action.value2 : 1;
                        eff.target_filter = resolved_filter;

                        auto cmd = std::make_unique<MutateCommand>(-1, MutateCommand::MutationType::ADD_PASSIVE_EFFECT);
                        cmd->passive_effect = eff;
                        cmd->execute(ctx.game_state);
                        ctx.game_state.command_history.push_back(std::move(cmd));
                    }
                } else {
                    // Global passive with resolved filter
                    PassiveEffect eff;
                    eff.type = PassiveType::POWER_MODIFIER;
                    eff.value = value;
                    eff.source_instance_id = ctx.source_instance_id;
                    eff.controller = EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                    eff.turns_remaining = (ctx.action.value2 > 0) ? ctx.action.value2 : 1;
                    eff.target_filter = resolved_filter;

                    auto cmd = std::make_unique<MutateCommand>(-1, MutateCommand::MutationType::ADD_PASSIVE_EFFECT);
                    cmd->passive_effect = eff;
                    cmd->execute(ctx.game_state);
                    ctx.game_state.command_history.push_back(std::move(cmd));
                }
            } else if (ctx.action.str_val == "CANNOT_ATTACK") {
                create_passive(PassiveType::CANNOT_ATTACK);
            } else if (ctx.action.str_val == "CANNOT_BLOCK") {
                create_passive(PassiveType::CANNOT_BLOCK);
            } else if (ctx.action.str_val == "CANNOT_ATTACK_OR_BLOCK") {
                create_passive(PassiveType::CANNOT_ATTACK);
                create_passive(PassiveType::CANNOT_BLOCK);
            } else if (ctx.action.str_val == "FORCE_ATTACK") {
                create_passive(PassiveType::FORCE_ATTACK);
            }
        }
    };
}
