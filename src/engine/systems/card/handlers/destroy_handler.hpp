#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/utils/zone_utils.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/pipeline_executor.hpp"
#include <algorithm>

namespace dm::engine {

    class DestroyHandler : public IActionHandler {
    public:
        void resolve(const ResolutionContext& ctx) override {
            using namespace dm::core;

            // Migrate to compile() -> Pipeline
            std::vector<Instruction> insts;
            ResolutionContext temp_ctx = ctx;
            temp_ctx.instruction_buffer = &insts;
            compile_action(temp_ctx);

            if (!insts.empty()) {
                auto pipeline = std::make_shared<dm::engine::systems::PipelineExecutor>();
                ctx.game_state.active_pipeline = pipeline;
                pipeline->execute(insts, ctx.game_state, ctx.card_db);
            }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            using namespace dm::core;
            // Legacy/Callback support
            if (!ctx.targets) return;

            Instruction move(InstructionOp::MOVE);
            move.args["to"] = "GRAVEYARD";
            move.args["target"] = "$targets";

            dm::engine::systems::PipelineExecutor pipeline;
            pipeline.set_context_var("$targets", *ctx.targets);
            pipeline.execute({move}, ctx.game_state, ctx.card_db);

            if (!ctx.action.output_value_key.empty()) {
                ctx.execution_vars[ctx.action.output_value_key] = (int)ctx.targets->size();
            }
        }

        void compile_action(const ResolutionContext& ctx) override {
            using namespace dm::core;
            if (!ctx.instruction_buffer) return;

            // Determine Targets
            std::string target_var = "";
            bool selection_needed = false;

            if (!ctx.action.input_value_key.empty()) {
                target_var = "$" + ctx.action.input_value_key;
            }
            else if (ctx.action.scope == TargetScope::TARGET_SELECT) {
                 // Selection handled by prior steps or implied?
                 // Usually compile() is called when we encounter the action.
                 // If it requires selection, we must generate SELECT instruction.

                 Instruction select(InstructionOp::SELECT);
                 select.args["filter"] = ctx.action.filter;
                 select.args["out"] = "$destroy_selection";

                 // Count
                 int count = ctx.action.value1;
                 if (count == 0) count = 1;
                 select.args["count"] = count;

                 ctx.instruction_buffer->push_back(select);
                 target_var = "$destroy_selection";
            }
            else {
                 // Auto Select (All/Any matching filter)
                 Instruction select(InstructionOp::SELECT);
                 select.args["filter"] = ctx.action.filter;
                 select.args["out"] = "$auto_destroy_selection";
                 select.args["count"] = 999;

                 ctx.instruction_buffer->push_back(select);
                 target_var = "$auto_destroy_selection";
            }

            // Move to Graveyard
            Instruction move(InstructionOp::MOVE);
            move.args["to"] = "GRAVEYARD";
            move.args["target"] = target_var;
            ctx.instruction_buffer->push_back(move);

            // Output value (Count destroyed)
            if (!ctx.action.output_value_key.empty()) {
                 Instruction count_inst(InstructionOp::COUNT);
                 count_inst.args["in"] = target_var;
                 count_inst.args["out"] = ctx.action.output_value_key;
                 ctx.instruction_buffer->push_back(count_inst);
            }
        }
    };
}
