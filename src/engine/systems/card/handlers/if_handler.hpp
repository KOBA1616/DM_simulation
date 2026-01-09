#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/card/condition_system.hpp"

namespace dm::engine {

    class IfHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            // resolve() is usually for direct execution, but IF logic is best handled by pipeline instructions.
            // If called directly, we might need to manually check condition and execute branches?
            // Since we can't easily execute "blocks" of actions without recursive resolve calls or pipeline,
            // we should rely on compile_action mainly.

            // However, if we MUST resolve here (legacy):
            bool cond_result = false;

            // Check condition using ConditionSystem or special logic
            if (ctx.action.condition) {
                // If special types are used:
                if (ctx.action.condition->type == "OPTIONAL_EFFECT_EXECUTED") {
                    // Check context variable "$optional_executed" or similar if we tracked it?
                    // Currently optional execution is not strictly tracked in context unless we do it.
                    // But usually optional execution result might be in ctx.execution_vars if linked.
                    // Let's assume ConditionSystem handles it if we implement evaluator for it.
                    cond_result = EffectSystem::instance().check_condition(ctx.game_state, *ctx.action.condition, ctx.source_instance_id, ctx.card_db, ctx.execution_vars);
                }
                else if (ctx.action.condition->type == "INPUT_VALUE_MATCH") {
                    int val = 0;
                     if (!ctx.action.input_value_key.empty() && ctx.execution_vars.count(ctx.action.input_value_key)) {
                        val = ctx.execution_vars.at(ctx.action.input_value_key);
                     }
                     // Compare val with ctx.action.condition->value
                     cond_result = (val == ctx.action.condition->value);
                }
                else {
                    cond_result = EffectSystem::instance().check_condition(ctx.game_state, *ctx.action.condition, ctx.source_instance_id, ctx.card_db, ctx.execution_vars);
                }
            }

            // const auto& branch = cond_result ? ctx.action.if_true : ctx.action.if_false;
            // Execute branch actions - force pipeline compilation via resolution
            // This fallback is incomplete because we don't have easy access to execute pipeline from here without compilation.
            // But since this is new action, usage will be via compile_action flow.
        }

        void compile_action(const ResolutionContext& ctx) override {
            if (!ctx.instruction_buffer) return;

            using namespace dm::core;

            Instruction if_inst(InstructionOp::IF);

            // Condition
            if (ctx.action.condition) {
                if (ctx.action.condition->type == "INPUT_VALUE_MATCH") {
                     // Custom condition compilation for variable match
                     // IF (var == value)
                     nlohmann::json cond;
                     cond["type"] = "MATH_CMP";
                     cond["lhs"] = "$" + ctx.action.input_value_key; // variable
                     cond["op"] = "==";
                     cond["rhs"] = ctx.action.condition->value;
                     if_inst.args["cond"] = cond;
                }
                else if (ctx.action.condition->type == "OPTIONAL_EFFECT_EXECUTED") {
                     // Check if last optional action was taken?
                     nlohmann::json cond;
                     cond["type"] = "VAR_TRUE";
                     cond["var"] = "$" + (ctx.action.input_value_key.empty() ? "optional_result" : ctx.action.input_value_key);
                     if_inst.args["cond"] = cond;
                }
                else {
                    if_inst.args["cond"] = ConditionSystem::instance().compile_condition(*ctx.action.condition);
                }
            } else {
                // Default true?
                if_inst.args["cond"]["type"] = "TRUE";
            }

            // Compile True/False Branches recursively
            EffectSystem& effect_system = EffectSystem::instance();

            for (const auto& act : ctx.action.if_true) {
                effect_system.compile_action(ctx.game_state, act, ctx.source_instance_id, const_cast<std::map<std::string, int>&>(ctx.execution_vars), ctx.card_db, if_inst.then_block);
            }

            for (const auto& act : ctx.action.if_false) {
                effect_system.compile_action(ctx.game_state, act, ctx.source_instance_id, const_cast<std::map<std::string, int>&>(ctx.execution_vars), ctx.card_db, if_inst.else_block);
            }

            ctx.instruction_buffer->push_back(if_inst);
        }
    };
}
