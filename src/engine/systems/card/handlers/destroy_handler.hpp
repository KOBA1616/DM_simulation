#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/card/effect_system.hpp"
#include "engine/utils/zone_utils.hpp"
#include "engine/game_command/commands.hpp"
#include "engine/systems/pipeline_executor.hpp" // Added include
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
            compile(temp_ctx);

            dm::engine::systems::PipelineExecutor pipeline;
            pipeline.execute(insts, ctx.game_state, ctx.card_db);
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            using namespace dm::core;

            // Use Pipeline for targets
            if (!ctx.targets) return;

            Instruction loop(InstructionOp::LOOP);
            loop.args["in"] = "$targets"; // We will set this var
            loop.args["as"] = "$id";

            Instruction move(InstructionOp::MOVE);
            move.args["to"] = "GRAVEYARD";
            move.args["target"] = "$id"; // inside loop
            loop.then_block.push_back(move);

            dm::engine::systems::PipelineExecutor pipeline;
            pipeline.set_context_var("$targets", *ctx.targets);
            pipeline.execute({loop}, ctx.game_state, ctx.card_db);

            if (!ctx.action.output_value_key.empty()) {
                ctx.execution_vars[ctx.action.output_value_key] = (int)ctx.targets->size();
            }
        }

        void compile(const ResolutionContext& ctx) override {
            using namespace dm::core;
            if (!ctx.instruction_buffer) return;

            Instruction move(InstructionOp::MOVE);
            move.args["to"] = "GRAVEYARD";

            if (ctx.action.scope == TargetScope::TARGET_SELECT) {
                 move.args["target"] = "$selection";
            }
            else if (!ctx.action.input_value_key.empty()) {
                 move.args["target"] = "$" + ctx.action.input_value_key;
            }
            else {
                 Instruction select(InstructionOp::SELECT);
                 select.args["filter"] = ctx.action.filter;
                 select.args["out"] = "$auto_destroy_selection";
                 select.args["count"] = 999;

                 ctx.instruction_buffer->push_back(select);
                 move.args["target"] = "$auto_destroy_selection";
            }

            ctx.instruction_buffer->push_back(move);

            // Output value
            if (!ctx.action.output_value_key.empty()) {
                 Instruction count_inst(InstructionOp::COUNT);
                 count_inst.args["in"] = move.args["target"];
                 count_inst.args["out"] = ctx.action.output_value_key;
                 ctx.instruction_buffer->push_back(count_inst);
            }
        }
    };
}
