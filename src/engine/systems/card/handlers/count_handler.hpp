#pragma once
#include "engine/systems/effects/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/systems/effects/effect_system.hpp"
#include "engine/infrastructure/data/card_registry.hpp"
#include "engine/utils/target_utils.hpp"
#include <set>
#include <string>
#include <algorithm>

namespace dm::engine {

    class CountHandler : public IActionHandler {
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

                // Sync Input Variables
                for (const auto& [k, v] : ctx.execution_vars) {
                    pipeline->set_context_var("$" + k, v);
                }

                // Pass controller info if needed by GET_STAT default logic
                PlayerID controller_id = dm::engine::effects::EffectSystem::get_controller(ctx.game_state, ctx.source_instance_id);
                pipeline->set_context_var("$controller", (int)controller_id);
                pipeline->set_context_var("$source", (int)ctx.source_instance_id);

                // ctx.game_state.active_pipeline = pipeline; // Removed
                pipeline->execute(insts, ctx.game_state, ctx.card_db);

                // Sync Output Variables
                if (!ctx.action.output_value_key.empty()) {
                    auto val = pipeline->get_context_var("$" + ctx.action.output_value_key);
                    if (std::holds_alternative<int>(val)) {
                        ctx.execution_vars[ctx.action.output_value_key] = std::get<int>(val);
                    }
                }
            }
        }

        void compile_action(const ResolutionContext& ctx) override {
             using namespace dm::core;
             if (!ctx.instruction_buffer) return;

             std::string out_key = ctx.action.output_value_key;
             if (out_key.empty()) out_key = "$count_result";

             if (ctx.action.type == EffectPrimitive::COUNT_CARDS) {
                 Instruction count_inst(InstructionOp::COUNT);
                 count_inst.args["filter"] = ctx.action.filter;

                 Instruction select(InstructionOp::SELECT);
                 select.args["filter"] = ctx.action.filter;
                 select.args["out"] = "$temp_selection_for_count";
                 select.args["count"] = 999;
                 ctx.instruction_buffer->push_back(select);

                 count_inst.args["in"] = "$temp_selection_for_count";
                 count_inst.args["out"] = out_key;
                 ctx.instruction_buffer->push_back(count_inst);
             }
             else if (ctx.action.type == EffectPrimitive::GET_GAME_STAT) {
                 Instruction stat(InstructionOp::GET_STAT);
                 stat.args["stat"] = ctx.action.str_val;
                 stat.args["out"] = out_key;
                 ctx.instruction_buffer->push_back(stat);
             }
        }
    };
}
