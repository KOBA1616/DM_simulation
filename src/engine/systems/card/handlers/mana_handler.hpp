#pragma once
#include "engine/systems/card/effect_system.hpp"
#include "core/game_state.hpp"
#include "engine/utils/tap_in_utils.hpp"
#include "engine/utils/zone_utils.hpp"
#include "core/card_def.hpp"
#include "engine/systems/card/target_utils.hpp"
#include "engine/systems/pipeline_executor.hpp" // Added include

namespace dm::engine {

    class ManaChargeHandler : public IActionHandler {
    public:
        // Case 1: ADD_MANA (Top of deck to Mana)
        void resolve(const ResolutionContext& ctx) override {
             using namespace dm::core;

             std::vector<Instruction> insts;
             ResolutionContext temp_ctx = ctx;
             temp_ctx.instruction_buffer = &insts;

             compile(temp_ctx);

             // Execute
             dm::engine::systems::PipelineExecutor pipeline;
             pipeline.execute(insts, ctx.game_state, ctx.card_db);
        }

        // Case 2: SEND_TO_MANA (Target to Mana)
        void resolve_with_targets(const ResolutionContext& ctx) override {
            using namespace dm::core;

            Instruction move(InstructionOp::MOVE);
            move.args["to"] = "MANA";

            dm::engine::systems::PipelineExecutor pipeline;
            if (ctx.targets) {
                pipeline.set_context_var("$targets", *ctx.targets);
                move.args["target"] = "$targets";
                pipeline.execute({move}, ctx.game_state, ctx.card_db);
            }
        }

        void compile(const ResolutionContext& ctx) override {
            using namespace dm::core;
            if (!ctx.instruction_buffer) return;

            if (ctx.action.type == EffectActionType::SEND_TO_MANA) {
                Instruction move(InstructionOp::GAME_ACTION);
                move.args["type"] = "MANA_CHARGE"; // Use GameAction for proper Tap-In/Untap logic handling

                if (!ctx.action.input_value_key.empty()) {
                    move.args["source_id"] = "$" + ctx.action.input_value_key;
                } else if (ctx.action.scope == TargetScope::TARGET_SELECT) {
                     // Let's use LOOP.
                     Instruction loop(InstructionOp::LOOP);
                     loop.args["in"] = "$selection";
                     loop.args["as"] = "$id";

                     Instruction charge(InstructionOp::GAME_ACTION);
                     charge.args["type"] = "MANA_CHARGE";
                     charge.args["source_id"] = "$id";

                     loop.then_block.push_back(charge);
                     ctx.instruction_buffer->push_back(loop);
                     return;
                } else {
                     // Auto Select
                     Instruction select(InstructionOp::SELECT);
                     select.args["filter"] = ctx.action.filter;
                     select.args["out"] = "$auto_mana_selection";
                     select.args["count"] = 999;

                     ctx.instruction_buffer->push_back(select);

                     Instruction loop(InstructionOp::LOOP);
                     loop.args["in"] = "$auto_mana_selection";
                     loop.args["as"] = "$id";

                     Instruction charge(InstructionOp::GAME_ACTION);
                     charge.args["type"] = "MANA_CHARGE";
                     charge.args["source_id"] = "$id";

                     loop.then_block.push_back(charge);
                     ctx.instruction_buffer->push_back(loop);
                     return;
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
