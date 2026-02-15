#pragma once
#include "core/game_state.hpp"
#include "core/action.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/infrastructure/pipeline/pipeline_executor.hpp"

namespace dm::engine {

    class ResolveBattleHandler : public IActionHandler {
    public:
        void compile_action(const ResolutionContext& ctx) override {
            using namespace dm::core;

            // RESOLVE_BATTLE requires an attacker (Source) and a defender (Target).
            // Usually the source of the effect is the attacker (e.g., "This creature battles target creature").
            // If the effect was "Target creature battles another target creature", the logic would be more complex,
            // but for now we assume Source vs Target.

            if (!ctx.selection_var.empty()) {
                // If we have a selection variable (from TARGET_SELECT), iterate over it.
                Instruction loop(InstructionOp::LOOP);
                loop.args["in"] = ctx.selection_var;
                loop.args["as"] = "$target";

                nlohmann::json args;
                args["type"] = "RESOLVE_BATTLE";
                args["attacker"] = ctx.source_instance_id;
                args["defender"] = "$target";

                loop.then_block.emplace_back(InstructionOp::GAME_ACTION, args);
                ctx.instruction_buffer->push_back(loop);
            } else {
                // No selection variable.
                // Check if specific target is provided in action (rare for effects, but possible in legacy paths)
                // or if we should use existing targets from context if any.

                int target_id = -1;
                // target_instance_id is NOT in ActionDef, it has filter, value1, value2, etc.
                // Legacy Action struct had target_instance_id, but ActionDef is from JSON.
                // If specific target is needed without selection, it's usually implicit or passed via other means.
                // However, ResolutionContext::targets might be populated if this was called via resolve_with_targets.

                if (ctx.targets && !ctx.targets->empty()) {
                    // Legacy targets passed directly
                     target_id = ctx.targets->front();
                }

                if (target_id != -1) {
                    nlohmann::json args;
                    args["type"] = "RESOLVE_BATTLE";
                    args["attacker"] = ctx.source_instance_id;
                    args["defender"] = target_id;
                    ctx.instruction_buffer->emplace_back(InstructionOp::GAME_ACTION, args);
                } else {
                    // Fallback: If no target found, do nothing or log warning.
                    // Effectively a no-op if no target.
                }
            }
        }

        void resolve(const ResolutionContext& ctx) override {
            std::vector<dm::core::Instruction> instructions;
            ResolutionContext compile_ctx = ctx;
            compile_ctx.instruction_buffer = &instructions;

            compile_action(compile_ctx);

            if (instructions.empty()) return;

            dm::engine::systems::PipelineExecutor pipeline;
            pipeline.execute(instructions, ctx.game_state, ctx.card_db);
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
             resolve(ctx);
        }
    };
}
