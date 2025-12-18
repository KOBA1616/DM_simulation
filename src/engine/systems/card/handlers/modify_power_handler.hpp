#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/pipeline_executor.hpp"
#include <iostream>

namespace dm::engine {

    class ModifyPowerHandler : public IActionHandler {
    public:
        void compile(const ResolutionContext& ctx) override {
            using namespace dm::core;

            int value = ctx.action.value1;
            if (!ctx.action.input_value_key.empty() && ctx.execution_vars.count(ctx.action.input_value_key)) {
                value = ctx.execution_vars.at(ctx.action.input_value_key);
            }

            std::vector<int> final_targets;
            if (ctx.targets) {
                final_targets = *ctx.targets;
            } else {
                 // Implicit targets logic could be added here if needed
            }

            if (final_targets.empty()) return;

            for (int target_id : final_targets) {
                 nlohmann::json args;
                 args["type"] = "POWER_ADD";
                 args["value"] = value;
                 args["target"] = target_id;
                 ctx.instruction_buffer->emplace_back(InstructionOp::MODIFY, args);
            }
        }

        void resolve(const ResolutionContext& ctx) override {
            std::vector<dm::core::Instruction> instructions;
            ResolutionContext compile_ctx = ctx;
            compile_ctx.instruction_buffer = &instructions;

            compile(compile_ctx);

            if (instructions.empty()) return;

            dm::engine::systems::PipelineExecutor pipeline;
            pipeline.execute(instructions, ctx.game_state, ctx.card_db);
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            // resolve() handles logic with ctx, including ctx.targets.
            // But base IActionHandler::resolve_with_targets is distinct?
            // Yes, base calls resolve_with_targets.
            // Wait, GenericCardSystem calls resolve_with_targets directly sometimes.
            // So we must implement it too.
            resolve(ctx);
        }
    };
}
