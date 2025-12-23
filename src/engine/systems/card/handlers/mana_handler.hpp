#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/utils/tap_in_utils.hpp"
#include "engine/utils/zone_utils.hpp"
#include "core/card_def.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/pipeline_executor.hpp"

namespace dm::engine {

    class ManaChargeHandler : public IActionHandler {
    public:
        // Use compile() -> PipelineExecutor pattern
        void resolve(const ResolutionContext& ctx) override {
             using namespace dm::core;

             std::vector<Instruction> insts;
             ResolutionContext temp_ctx = ctx;
             temp_ctx.instruction_buffer = &insts;

             compile_action(temp_ctx);

             if (!insts.empty()) {
                 // Use persistent pipeline to support pausing (e.g. SELECT)
                 auto pipeline = std::make_shared<dm::engine::systems::PipelineExecutor>();
                 ctx.game_state.active_pipeline = pipeline;
                 pipeline->execute(insts, ctx.game_state, ctx.card_db);
             }
        }

        void resolve_with_targets(const ResolutionContext& ctx) override {
            using namespace dm::core;
            // Legacy support if called directly (though deprecated in favor of pipeline)
            // Use compile logic or manual pipeline

            Instruction move(InstructionOp::MOVE);
            move.args["to"] = "MANA";

            dm::engine::systems::PipelineExecutor pipeline;
            if (ctx.targets) {
                pipeline.set_context_var("$targets", *ctx.targets);
                move.args["target"] = "$targets";
                pipeline.execute({move}, ctx.game_state, ctx.card_db);
            }
        }

        void compile_action(const ResolutionContext& ctx) override {
            using namespace dm::core;
            if (!ctx.instruction_buffer) return;

            if (ctx.action.type == EffectPrimitive::SEND_TO_MANA) {
                // If the input is from a previous step (variable linking)
                if (!ctx.action.input_value_key.empty()) {
                     Instruction move(InstructionOp::MOVE);
                     move.args["to"] = "MANA";
                     move.args["target"] = "$" + ctx.action.input_value_key;
                     ctx.instruction_buffer->push_back(move);
                }
                // If the action requires target selection (interactive)
                else if (ctx.action.scope == TargetScope::TARGET_SELECT) {
                     Instruction select(InstructionOp::SELECT);
                     select.args["filter"] = ctx.action.filter;
                     select.args["out"] = "$mana_selection";
                     select.args["count"] = 1; // Or variable?
                     // Currently SEND_TO_MANA usually implies 1 target unless specified.
                     // But action doesn't specify count in scope TARGET_SELECT?
                     // If value1 > 0, use it.
                     if (ctx.action.value1 > 0) select.args["count"] = ctx.action.value1;

                     ctx.instruction_buffer->push_back(select);

                     Instruction move(InstructionOp::MOVE);
                     move.args["to"] = "MANA";
                     move.args["target"] = "$mana_selection";
                     ctx.instruction_buffer->push_back(move);
                }
                // Auto Select (e.g. "Put all your creatures into mana")
                else {
                     Instruction select(InstructionOp::SELECT);
                     select.args["filter"] = ctx.action.filter;
                     select.args["out"] = "$auto_mana_selection";
                     select.args["count"] = 999;

                     ctx.instruction_buffer->push_back(select);

                     Instruction move(InstructionOp::MOVE);
                     move.args["to"] = "MANA";
                     move.args["target"] = "$auto_mana_selection";

                     ctx.instruction_buffer->push_back(move);
                }
            }
            else { // ADD_MANA (Deck Top)
                int count = ctx.action.value1;
                if (count == 0) count = 1;

                Instruction move(InstructionOp::MOVE);
                move.args["to"] = "MANA";
                move.args["target"] = "DECK_TOP";
                move.args["count"] = count;

                ctx.instruction_buffer->push_back(move);
            }
        }
    };
}
